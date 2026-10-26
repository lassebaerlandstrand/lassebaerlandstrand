"""Schwarzschild null-geodesic tracer (vectorised numpy).

Units: Schwarzschild radius r_s = 1 (so M = 1/2, photon sphere r = 1.5,
critical impact parameter b_c = 3*sqrt(3)/2 ~ 2.598, ISCO r = 3).

Rays are traced backwards from a pinhole camera using the exact orbit equation
written in Cartesian form, x'' = -3/2 h^2 x / r^5 with h = |x cross x'|, which
reproduces the Schwarzschild photon orbit r(phi) (the trick popularised by
rantonels' "Starless").  For every ray we record up to NCROSS crossings of the
equatorial plane (the accretion disk), its conserved angular momentum about the
disk axis (for the Doppler factor), and either its asymptotic escape direction
or capture by the horizon.
"""
import numpy as np
import multiprocessing as mp

NCROSS = 4


def _accel(p, h2):
    r2 = np.einsum('ij,ij->j', p, p)
    return p * (-1.5 * h2 / (r2 * r2 * np.sqrt(r2)))


def trace_chunk(args):
    pos, vel, max_steps = args
    pos = pos.copy()
    vel = vel / np.linalg.norm(vel, axis=0)
    n = pos.shape[1]
    Lz = -np.cross(pos.T, vel.T)[:, 2]          # photon (forward) L_z, E = 1
    h2 = np.sum(np.cross(pos.T, vel.T) ** 2, axis=1)

    cr = np.full((NCROSS, n), np.nan)            # crossing radius
    cphi = np.full((NCROSS, n), np.nan)          # crossing azimuth
    ncr = np.zeros(n, np.int32)
    status = np.zeros(n, np.int8)                # 0 active, 1 captured, 2 escaped
    esc = np.zeros((3, n))

    idx = np.arange(n)
    p, v, hh = pos, vel, h2
    for _ in range(max_steps):
        if idx.size == 0:
            break
        r = np.sqrt(np.einsum('ij,ij->j', p, p))
        dt = np.minimum(0.025 * r + 0.0015 * r * r, 60.0)
        # RK4
        k1v = _accel(p, hh); k1x = v
        k2v = _accel(p + 0.5 * dt * k1x, hh); k2x = v + 0.5 * dt * k1v
        k3v = _accel(p + 0.5 * dt * k2x, hh); k3x = v + 0.5 * dt * k2v
        k4v = _accel(p + dt * k3x, hh); k4x = v + dt * k3v
        pn = p + dt / 6 * (k1x + 2 * k2x + 2 * k3x + k4x)
        vn = v + dt / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)

        z0, z1 = p[2], pn[2]
        hit = (z0 * z1 < 0)
        if hit.any():
            hi = np.nonzero(hit)[0]
            t = z0[hi] / (z0[hi] - z1[hi])
            q = p[:, hi] + (pn[:, hi] - p[:, hi]) * t
            gid = idx[hi]
            slot = ncr[gid]
            ok = slot < NCROSS
            gid, slot, q = gid[ok], slot[ok], q[:, ok]
            cr[slot, gid] = np.hypot(q[0], q[1])
            cphi[slot, gid] = np.arctan2(q[1], q[0])
            ncr[gid] += 1

        p, v = pn, vn
        rn = np.sqrt(np.einsum('ij,ij->j', p, p))
        cap = rn < 1.0005
        out = (rn > 3000.0) & (np.einsum('ij,ij->j', p, v) > 0)
        if cap.any():
            status[idx[cap]] = 1
        if out.any():
            status[idx[out]] = 2
            esc[:, idx[out]] = v[:, out] / np.linalg.norm(v[:, out], axis=0)
        keep = ~(cap | out)
        if not keep.all():
            idx, p, v, hh = idx[keep], p[:, keep], v[:, keep], hh[keep]
    # anything left: treat as captured (lingering on the photon sphere)
    status[idx] = np.where(status[idx] == 0, 1, status[idx])
    return cr, cphi, Lz, status, esc


def camera(D, inc_deg, roll_deg=0.0):
    inc = np.radians(inc_deg)
    C = D * np.array([np.sin(inc), 0.0, np.cos(inc)])
    fwd = -C / np.linalg.norm(C)
    up0 = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, up0); right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    a = np.radians(roll_deg)
    right, up = np.cos(a) * right + np.sin(a) * up, -np.sin(a) * right + np.cos(a) * up
    return C, fwd, right, up


def trace_points(X, Y, cx, cy, px_per_rs, D, inc_deg, roll_deg=0.0,
                 procs=18, max_steps=3000):
    """Trace rays through screen points (X, Y) given in output-pixel units."""
    C, fwd, right, up = camera(D, inc_deg, roll_deg)
    f = px_per_rs * D
    u = (np.ravel(X) - cx) / f
    v = (np.ravel(Y) - cy) / f
    dirs = fwd[:, None] + u[None] * right[:, None] - v[None] * up[:, None]
    dirs /= np.linalg.norm(dirs, axis=0)
    n = dirs.shape[1]
    pos = np.repeat(C[:, None], n, axis=1)
    perm = np.random.default_rng(0).permutation(n)       # balance chunk cost
    chunks = np.array_split(perm, procs * 8)
    jobs = [(pos[:, c], dirs[:, c], max_steps) for c in chunks]
    with mp.Pool(procs) as pool:
        res = pool.map(trace_chunk, jobs)
    cr = np.full((NCROSS, n), np.nan); cphi = np.full((NCROSS, n), np.nan)
    Lz = np.zeros(n); status = np.zeros(n, np.int8); esc = np.zeros((3, n))
    for c, (a, b, l, s_, e) in zip(chunks, res):
        cr[:, c] = a; cphi[:, c] = b; Lz[c] = l; status[c] = s_; esc[:, c] = e
    shp = np.shape(X)
    return dict(cr=cr.reshape(NCROSS, *shp).astype(np.float32),
                cphi=cphi.reshape(NCROSS, *shp).astype(np.float32),
                Lz=Lz.reshape(shp).astype(np.float32),
                status=status.reshape(shp),
                esc=esc.reshape(3, *shp).astype(np.float32),
                cam=np.stack([C, fwd, right, up]))


def trace_image(W, H, SS, cx, cy, px_per_rs, D, inc_deg, roll_deg=0.0, **kw):
    xs = (np.arange(W * SS) + 0.5) / SS
    ys = (np.arange(H * SS) + 0.5) / SS
    X, Y = np.meshgrid(xs, ys)
    return trace_points(X, Y, cx, cy, px_per_rs, D, inc_deg, roll_deg, **kw)


def trace_pixels(px, py, SS, cx, cy, px_per_rs, D, inc_deg, roll_deg=0.0, **kw):
    """Heavily supersample a sparse set of output pixels -> arrays (npix, SS*SS)."""
    o = (np.arange(SS) + 0.5) / SS
    ox, oy = np.meshgrid(o, o)
    X = px[:, None] + ox.ravel()[None]
    Y = py[:, None] + oy.ravel()[None]
    return trace_points(X, Y, cx, cy, px_per_rs, D, inc_deg, roll_deg, **kw)
