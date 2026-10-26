#!/usr/bin/env python3
"""Black hole banner: ray-traced Schwarzschild black hole with a lensed,
Doppler-beamed accretion disk, packaged as an animated SVG.

Pipeline
  1. trace.py   - integrate null geodesics for every subpixel (cached), plus a
                  much denser trace of the pixels around the photon ring.
  2. render.py  - shade disk hits (texture, Keplerian motion, Doppler beaming,
                  gravitational redshift, blackbody colour), lensed stars.
  3. here       - bloom + filmic grade, choose the region that moves, encode a
                  static backdrop and N loop frames, write SVGs whose frames
                  crossfade with SMIL opacity animations.

Needs numpy, scipy, pillow.  Run:  python generate.py
"""
import base64, hashlib, io, os, sys, tempfile, time
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import trace as T   # noqa: E402
import render as R  # noqa: E402

CACHE = os.environ.get('BH_CACHE', os.path.join(tempfile.gettempdir(), 'bh-cache'))

# ------------------------------------------------------------------ settings
W, H = 1200, 340
SS = 3                 # supersampling of the main trace
SS_RING = 12           # supersampling around the photon ring
CX, CY = 830.0, 170.0  # where the hole sits in the frame
SCALE = 20.0           # pixels per Schwarzschild radius at the hole
DIST = 150.0           # camera distance (r_s)
INC = 84.0             # inclination (deg from the disk axis)
ROLL = 0.0

N_FRAMES = 32
LOOP_S = 48.0
FMT = os.environ.get('BH_FMT', 'webp')   # 'webp' (about half the size) or 'jpeg'
Q_BG = 90
Q_FR = 86
DITHER = 0.0   # flat space must encode identically in backdrop and frames (no seam)

BG = np.array([0.0016, 0.0021, 0.0036])   # linear ambient added before tone mapping
SPACE = np.array([5, 7, 11]) / 255.0       # display-space floor (#05070b)

STAR_DENSITY = 300 / (3600 * 1020)   # per sky subpixel^2
BRIGHT_STARS = [  # (x, y) where they appear, in output px; luminance; temperature
    (138, 62, 2.4, 9000), (312, 262, 1.2, 5200), (430, 40, 0.9, 7000),
    (95, 286, 0.8, 4600), (226, 150, 0.5, 6500), (1010, 300, 0.7, 5600),
    (1176, 14, 0.6, 8000), (612, 318, 0.6, 6000),
]


def cached(name, fn):
    os.makedirs(CACHE, exist_ok=True)
    key = hashlib.md5(repr((W, H, SS, SS_RING, CX, CY, SCALE, DIST, INC, ROLL,
                            open(os.path.join(HERE, 'trace.py')).read())).encode()).hexdigest()[:10]
    path = os.path.join(CACHE, f'{name}-{key}.npz')
    if os.path.exists(path):
        return dict(np.load(path))
    d = fn()
    np.savez(path, **d)
    return d


def ring_pixels():
    b_c = 1.5 * np.sqrt(3.0) * SCALE
    yy, xx = np.mgrid[0:H, 0:W]
    d = np.hypot(xx + 0.5 - CX, yy + 0.5 - CY)
    m = np.abs(d - b_c) < 7.0
    return xx[m], yy[m]


class Scene:
    def __init__(self):
        t = time.time()
        self.tr = cached('main', lambda: T.trace_image(W, H, SS, CX, CY, SCALE, DIST, INC, ROLL))
        self.rpx, self.rpy = ring_pixels()
        self.trr = cached('ring', lambda: T.trace_pixels(self.rpx.astype(float), self.rpy.astype(float),
                                                         SS_RING, CX, CY, SCALE, DIST, INC, ROLL))
        print(f'trace ready ({time.time() - t:.1f}s)')
        self.tex = R.DiskTexture()
        f_sub = SCALE * DIST * SS
        SX, SY, ok = R.sky_coords(self.tr, f_sub)
        xs, ys, flux, col = R.random_stars(3, SX[ok].min(), SX[ok].max(),
                                           SY[ok].min(), SY[ok].max(), STAR_DENSITY)
        # hand-placed brighter stars, positioned where they should *appear*
        bx = [SX[int(y * SS) + 1, int(x * SS) + 1] for x, y, *_ in BRIGHT_STARS]
        by = [SY[int(y * SS) + 1, int(x * SS) + 1] for x, y, *_ in BRIGHT_STARS]
        bf = [f for *_, f, _ in BRIGHT_STARS]
        bc = 0.5 * R.blackbody(np.array([t for *_, t in BRIGHT_STARS])) + 0.5
        xs = np.r_[xs, bx]; ys = np.r_[ys, by]; flux = np.r_[flux, bf]; col = np.r_[col, bc]
        ix, iy, mu, j = R.star_images(SX, SY, ok, xs, ys)
        f = flux[j] * np.minimum(mu, 2.5)
        keep = f > 0.004
        self.stars = R.splat_points(H, W, (ix[keep] + 0.5) / SS, (iy[keep] + 0.5) / SS,
                                    f[keep], col[j[keep]])
        print(f'{len(xs)} stars -> {keep.sum()} images')

    def hdr(self, phase):
        e, t = R.shade(self.tr, self.tex, phase)
        e, t = R.downsample(e, SS), R.downsample(t, SS)
        er, tr_ = R.shade(self.trr, self.tex, phase)
        e[self.rpy, self.rpx] = er.mean(1)
        t[self.rpy, self.rpx] = tr_.mean(1)
        return e + self.stars * t[..., None]


def blur(img, s):
    return np.stack([gaussian_filter(img[..., c], s) for c in range(3)], -1)


def wide_glow(mean_hdr):
    """Broad halo, computed once from the loop-averaged image so it is static
    (keeps everything outside the animated box exactly constant)."""
    return 0.07 * blur(mean_hdr, 32) + 0.05 * blur(mean_hdr, 80)


def post(hdr, glow, seed=0):
    b = 0.10 * blur(hdr, 1.2) + 0.08 * blur(hdr, 4) + 0.07 * blur(hdr, 12)
    x = hdr + b + glow + BG
    a, bb, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14          # ACES (Narkowicz)
    y = np.clip((x * (a * x + bb)) / (x * (c * x + d) + e), 0, 1)
    y = np.where(y <= 0.0031308, 12.92 * y, 1.055 * np.power(y, 1 / 2.4) - 0.055)
    y = y + (1 - y) * SPACE                                    # deep blue-black floor
    rng = np.random.default_rng(seed)
    y = y * 255 + DITHER * rng.uniform(-0.5, 0.5, y.shape)     # optional dither
    return np.clip(y + 0.5, 0, 255).astype(np.uint8)


def encode(arr, q):
    buf = io.BytesIO()
    if FMT == 'webp':
        Image.fromarray(arr).save(buf, 'WEBP', quality=q, method=6)
    else:
        Image.fromarray(arr).save(buf, 'JPEG', quality=q, optimize=True, progressive=True,
                                  subsampling=0 if q >= 90 else 2)
    return buf.getvalue()


MIME = 'image/webp' if FMT == 'webp' else 'image/jpeg'


def b64(data, mime):
    return f'data:{mime};base64,' + base64.b64encode(data).decode()


def build_svg(bg_jpg, frames, box, theme):
    x0, y0, x1, y1 = box
    N = len(frames)
    eps = 1e-4
    border = '#30363d' if theme == 'dark' else '#d0d7de'
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
           f'viewBox="0 0 {W} {H}" width="{W}" height="{H}">',
           '<defs><clipPath id="c"><rect width="1200" height="340" rx="16"/></clipPath></defs>',
           '<rect width="1200" height="340" rx="16" fill="#05070b"/>',
           '<g clip-path="url(#c)">',
           f'<image width="{W}" height="{H}" href="{b64(bg_jpg, MIME)}"/>',
           f'<g transform="translate({x0} {y0})">']
    w, h = x1 - x0, y1 - y0
    dur = f'{LOOP_S:g}s'
    for k, fr in enumerate(frames):
        attrs = f'id="f{k}" width="{w}" height="{h}" href="{b64(fr, MIME)}"'
        if k == 0:
            out.append(f'<image {attrs}/>')
            continue
        a, b_, c = (k - 1) / N, k / N, (k + 1) / N
        if k < N - 1:
            kt = f'0;{a:.5f};{b_:.5f};{c:.5f};{c + eps:.5f};1'
            vals = '0;0;1;1;0;0'
        else:
            kt = f'0;{a:.5f};{b_:.5f};1'
            vals = '0;0;1;1'
        if k == 1:
            kt = kt.replace('0;0.00000;', '0;', 1); vals = vals[2:]
        out.append(f'<image {attrs} opacity="0"><animate attributeName="opacity" dur="{dur}" '
                   f'repeatCount="indefinite" keyTimes="{kt}" values="{vals}"/></image>')
    out.append(f'<use href="#f0" opacity="0"><animate attributeName="opacity" dur="{dur}" '
               f'repeatCount="indefinite" keyTimes="0;{(N - 1) / N:.5f};1" values="0;0;1"/></use>')
    out.append('</g></g>')
    out.append(f'<rect x=".5" y=".5" width="1199" height="339" rx="15.5" fill="none" stroke="{border}"/>')
    out.append('</svg>')
    return '\n'.join(out)


def main():
    sc = Scene()
    t = time.time()
    phases = np.arange(N_FRAMES) / N_FRAMES
    hdrs = [sc.hdr(p) for p in phases]
    glow = wide_glow(np.mean(hdrs, 0))
    imgs = [post(h, glow) for h in hdrs]
    print(f'rendered {N_FRAMES} frames ({time.time() - t:.1f}s)')
    # region that actually changes over the loop
    ref = imgs[0].astype(int)
    diff = np.zeros((H, W), bool)
    for im in imgs[1:]:
        diff |= np.abs(im.astype(int) - ref).max(-1) > 0
    ys, xs = np.nonzero(diff)
    pad = 6
    x0 = max(int(xs.min()) - pad, 0) // 8 * 8; x1 = min(-(-(int(xs.max()) + pad) // 8) * 8, W)
    y0 = max(int(ys.min()) - pad, 0) // 8 * 8; y1 = min(-(-(int(ys.max()) + pad) // 8) * 8, H)
    box = (x0, y0, x1, y1)
    print('animated box', box, (x1 - x0, y1 - y0))
    bg = encode(imgs[0], Q_BG)
    frames = [encode(im[y0:y1, x0:x1], Q_FR) for im in imgs]
    print(f'backdrop {len(bg) / 1024:.0f} KB, frames {sum(map(len, frames)) / 1024:.0f} KB '
          f'({np.mean(list(map(len, frames))) / 1024:.1f} KB each)')
    for theme in ('dark', 'light'):
        svg = build_svg(bg, frames, box, theme)
        p = os.path.join(HERE, f'banner-{theme}.svg')
        open(p, 'w').write(svg)
        print(p, f'{len(svg) / 1024:.0f} KB')
    Image.fromarray(imgs[0]).save(os.path.join(CACHE, 'still.png'))


if __name__ == '__main__':
    main()
