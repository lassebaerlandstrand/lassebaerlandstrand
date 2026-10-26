"""Shading for the traced black hole: disk texture, Doppler/gravitational shift,
lensed star field, bloom and colour grade.  Works on the per-subpixel output of
trace.py so any number of animation frames can be shaded cheaply.
"""
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

# ---------------------------------------------------------------- colour
def _g(x, mu, s1, s2):
    s = np.where(x < mu, s1, s2)
    return np.exp(-0.5 * ((x - mu) / s) ** 2)


def _cie(lam):  # Wyman, Sloan & Shirley 2013 multi-lobe fit, lam in nm
    x = 1.056 * _g(lam, 599.8, 37.9, 31.0) + 0.362 * _g(lam, 442.0, 16.0, 26.7) - 0.065 * _g(lam, 501.1, 20.4, 26.2)
    y = 0.821 * _g(lam, 568.8, 46.9, 40.5) + 0.286 * _g(lam, 530.9, 16.3, 31.1)
    z = 1.217 * _g(lam, 437.0, 11.8, 36.0) + 0.681 * _g(lam, 459.0, 26.0, 13.8)
    return x, y, z


_XYZ2RGB = np.array([[3.2406, -1.5372, -0.4986],
                     [-0.9689, 1.8758, 0.0415],
                     [0.0557, -0.2040, 1.0570]])


def _bb_table():
    lam = np.linspace(380, 780, 401)
    X, Y, Z = _cie(lam)
    Ts = np.geomspace(800, 60000, 600)
    rgb = []
    for T in Ts:
        l = lam * 1e-9
        B = 1.0 / (l ** 5 * (np.exp(1.4388e-2 / (l * T)) - 1.0))
        xyz = np.array([np.sum(B * X), np.sum(B * Y), np.sum(B * Z)])
        c = _XYZ2RGB @ (xyz / xyz[1])
        c = np.maximum(c, 0)
        rgb.append(c / (0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]))
    return Ts, np.array(rgb)


_TS, _TRGB = _bb_table()


def blackbody(T):
    """Linear sRGB chromaticity (unit luminance) of a blackbody at T kelvin."""
    lt = np.log(np.clip(T, _TS[0], _TS[-1]))
    lts = np.log(_TS)
    return np.stack([np.interp(lt, lts, _TRGB[:, c]) for c in range(3)], -1)


# ---------------------------------------------------------------- disk texture
RHO0, RHO1 = np.log(2.6), np.log(60.0)


def _field(rng, n_rho, n_psi, a_rho, a_psi, beta):
    w = rng.standard_normal((n_rho, n_psi))
    kr = np.fft.fftfreq(n_rho) * n_rho
    kp = np.fft.rfftfreq(n_psi) * n_psi
    KR, KP = np.meshgrid(kr, kp, indexing='ij')
    k = np.sqrt((KR / a_rho) ** 2 + (KP / a_psi) ** 2)
    amp = (1.0 + k * k) ** (-beta / 2) * np.exp(-(KR / (n_rho * 0.3)) ** 2)
    amp[0, 0] = 0
    f = np.fft.irfft2(np.fft.rfft2(w) * amp, s=(n_rho, n_psi))
    return (f - f.mean()) / f.std()


class DiskTexture:
    """Turbulent, streaky disk texture made of overlapping radial bands.

    Band j rotates rigidly by 2*pi*n_j/m_j per loop and its pattern is
    m_j-fold periodic in azimuth, so every band (and hence the whole texture)
    returns exactly to its start after one loop, while the angular speeds still
    follow Kepler's law Omega ~ r^-3/2 band by band.
    """

    def __init__(self, seed=7, n_bands=16, turns_inner=1.0, r_in=3.0, r_out=45.0):
        rng = np.random.default_rng(seed)
        self.centres = np.geomspace(r_in * 0.95, r_out, n_bands)
        self.lc = np.log(self.centres)
        self.width = np.diff(self.lc).mean() * 0.62
        self.bands = []
        for rc in self.centres:
            q = turns_inner * (rc / r_in) ** -1.5           # Keplerian turns per loop
            best = None
            for m in range(1, 64):
                for n_ in range(1, 6):
                    err = abs(np.log((n_ / m) / q))
                    if best is None or err < best[0] - 1e-9:
                        best = (err, n_, m)
                if best[0] < 0.06:
                    break
            _, n_, m = best
            # angular wavelength target (physical length along the orbit)
            L = 5.0 + 0.35 * rc
            kpsi = max(2 * np.pi * rc / (m * L), 0.9)
            f1 = _field(rng, 512, 512, 22.0, kpsi, 1.9)        # broad clumps
            f2 = _field(rng, 512, 512, 110.0, kpsi * 1.6, 1.6)  # fine streaks
            f = 0.85 * f1 + 0.42 * f2
            f = (f - f.mean()) / f.std()
            self.bands.append(dict(n=n_, m=m, f=f.astype(np.float32),
                                   off=rng.uniform(0, 2 * np.pi)))
        self.rates = [b['n'] / b['m'] for b in self.bands]

    def __call__(self, r, phi, phase):
        """phase in [0,1): fraction of the loop. Returns ~N(0,1) field."""
        rho = np.log(r)
        num = np.zeros_like(r); den = np.zeros_like(r)
        for lc, b in zip(self.lc, self.bands):
            w = np.exp(-0.5 * ((rho - lc) / self.width) ** 2)
            sel = w > 1e-3
            if not sel.any():
                continue
            theta = 2 * np.pi * phase * b['n'] / b['m']
            psi = np.mod(b['m'] * (phi[sel] - theta) + b['off'], 2 * np.pi)
            f = b['f']
            ir = (rho[sel] - RHO0) / (RHO1 - RHO0) * (f.shape[0] - 1)
            ip = psi / (2 * np.pi) * f.shape[1]
            v = map_coordinates(f, [ir, ip], order=1, mode='grid-wrap')
            num[sel] += w[sel] * v
            den[sel] += w[sel] ** 2
        return num / np.sqrt(np.maximum(den, 1e-6))


# ---------------------------------------------------------------- stars
# Stars are point sources.  Rather than painting them onto a sky texture (which
# lensing would smear into scratch-like arcs), we find every lensed *image* of
# every star by inverting the traced sky map cell by cell, then draw each image
# as a round point whose flux is scaled by the magnification |mu|.
def sky_coords(tr, f_sub):
    """Tangent-plane sky coordinates (in subpixel units) seen by each ray."""
    C, fwd, right, up = tr['cam']
    shp = tr['status'].shape
    e = tr['esc'].reshape(3, -1).astype(np.float64)
    dz = fwd @ e
    u = (right @ e) / np.maximum(dz, 1e-6)
    v = (up @ e) / np.maximum(dz, 1e-6)
    ok = (dz > 0.3) & (tr['status'].ravel() == 2)
    return (u * f_sub).reshape(shp), (-v * f_sub).reshape(shp), ok.reshape(shp)


def random_stars(seed, x0, x1, y0, y1, density):
    rng = np.random.default_rng(seed)
    n = int(density * (x1 - x0) * (y1 - y0))
    xs = rng.uniform(x0, x1, n); ys = rng.uniform(y0, y1, n)
    mag = rng.pareto(1.5, n) + 1.0
    flux = 0.04 * np.minimum(mag, 40) ** 1.45
    temp = np.exp(rng.normal(np.log(6500), 0.35, n))
    col = 0.5 * blackbody(temp) + 0.5
    return xs, ys, flux, col


def star_images(SX, SY, ok, sx, sy):
    """All lensed images of point sources at sky (sx, sy).

    Returns image positions in subpixel index coordinates, |mu|, star index."""
    from scipy.ndimage import binary_erosion
    from scipy.spatial import cKDTree
    gXy, gXx = np.gradient(SX)
    gYy, gYx = np.gradient(SY)
    det = gXx * gYy - gXy * gYx
    m = binary_erosion(ok, iterations=2) & np.isfinite(det) & (np.abs(det) > 1e-6)
    iy, ix = np.nonzero(m)
    tree = cKDTree(np.c_[sx, sy])
    _, j = tree.query(np.c_[SX[iy, ix], SY[iy, ix]], workers=-1)
    dx = sx[j] - SX[iy, ix]; dy = sy[j] - SY[iy, ix]
    a, b, c, d, dt = gXx[iy, ix], gXy[iy, ix], gYx[iy, ix], gYy[iy, ix], det[iy, ix]
    ex = (d * dx - b * dy) / dt
    ey = (-c * dx + a * dy) / dt
    ins = (ex >= -0.5) & (ex < 0.5) & (ey >= -0.5) & (ey < 0.5)
    return ix[ins] + ex[ins], iy[ins] + ey[ins], 1.0 / np.abs(dt[ins]), j[ins]


def splat_points(H, W, x, y, flux, col, sigma=0.55):
    """Round Gaussian points (x, y in output pixel coords) -> (H, W, 3)."""
    img = np.zeros((H, W, 3), np.float32)
    R_ = 3
    k = np.arange(-R_, R_ + 1)
    for xi, yi, f, c in zip(x, y, flux, col):
        cx, cy = int(np.floor(xi)), int(np.floor(yi))
        gx = np.exp(-0.5 * ((cx + k + 0.5 - xi) / sigma) ** 2)
        gy = np.exp(-0.5 * ((cy + k + 0.5 - yi) / sigma) ** 2)
        wgt = gy[:, None] * gx[None, :]
        wgt *= f / wgt.sum()
        ys_, xs_ = cy + k, cx + k
        vy = (ys_ >= 0) & (ys_ < H); vx = (xs_ >= 0) & (xs_ < W)
        img[np.ix_(ys_[vy], xs_[vx])] += wgt[np.ix_(vy, vx)][..., None] * c
    return img


# ---------------------------------------------------------------- shading
P = dict(r_in=3.0, r_out=25.0, T_in=5400.0, T_slope=0.38, beam=4.0, gcol=0.45,
         spin=1.0, tex_amp=0.62, tau0=2.2, exposure=0.84, falloff=2.0,
         ring_boost=4.0)


def disk_shade(r, phi, Lz, tex, phase, p):
    """Return (emission rgb, alpha) for disk hits at (r, phi)."""
    r_in, r_out = p['r_in'], p['r_out']
    x = np.clip(r, r_in, None)
    inner = np.clip(1 - np.sqrt(r_in / x), 0, 1) ** 0.45
    inner *= np.clip((r - r_in * 0.97) / (r_in * 0.12), 0, 1)
    taper = np.clip(1 - ((r - r_out * 0.35) / (r_out * 0.65)), 0, 1) ** 2
    prof = inner * (x / r_in) ** -p['falloff'] * taper
    n = tex(r, phi, phase)
    amp = p['tex_amp'] / (1 + 0.04 * (x - r_in))
    mod = np.exp(amp * n - 0.5 * amp ** 2)
    om = np.sqrt(0.5 / x ** 3)
    g = np.sqrt(1 - 1.5 / x) / (1 - p['spin'] * om * Lz)
    T = p['T_in'] * (x / r_in) ** -p['T_slope']
    col = blackbody(T * g ** p['gcol'])
    I = prof * mod * g ** p['beam']
    tau = p['tau0'] * prof ** 0.5 * mod
    a = 1 - np.exp(-tau)
    a = np.where((r >= r_in * 0.97) & (r <= r_out), a, 0)
    return col * (I * p['exposure'])[..., None], a


def shade(tr, tex, phase, p=P):
    """Disk emission (rgb) and remaining transmittance for every ray."""
    cr, cphi, Lz = tr['cr'], tr['cphi'], tr['Lz']
    shp = Lz.shape
    out = np.zeros(shp + (3,), np.float32)
    T = np.ones(shp, np.float32)
    for k in range(cr.shape[0]):
        rk = cr[k]
        m = np.isfinite(rk) & (rk >= p['r_in'] * 0.97) & (rk <= p['r_out'])
        if not m.any():
            continue
        e, a = disk_shade(rk[m].astype(np.float64), cphi[k][m].astype(np.float64),
                          Lz[m].astype(np.float64), tex, phase, p)
        if k >= 2:   # higher-order images: the photon ring, given a little extra
            e = e * p['ring_boost']; a = np.maximum(a, 0.9)
        out[m] += (T[m] * a)[:, None] * e
        T[m] *= (1 - a)
    return out, T


def downsample(img, ss):
    H, W = img.shape[0] // ss, img.shape[1] // ss
    sh = img.shape[2:]
    return img[:H * ss, :W * ss].reshape(H, ss, W, ss, *sh).mean((1, 3))
