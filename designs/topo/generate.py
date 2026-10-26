#!/usr/bin/env python3
"""Animated topographic-contour banner generator.

Approach
--------
A smooth static scalar field (random Fourier terrain + a few Gaussian massifs,
lightly domain-warped) is contoured once with marching squares.  Motion comes
from a smooth, time-periodic *diffeomorphic* warp of the plane: every contour
point p is carried to p + E(p, t).  Because the warp is invertible, the
topology of the map can never change (no merging/splitting/popping contours),
and point correspondence between keyframes is exact, so the SMIL `d` morph is
glitch-free.  The warp's normal component makes contours advance, retreat,
bunch and spread, which reads as the terrain slowly breathing.

Each contour is written as a closed quadratic B-spline (M .. q .. t ...).
Control points are quantised to 1/Q px and the path is written in integer
1/(2Q) px units, so the on-curve midpoints the `t` chain encodes are exact and
cannot drift.  Control points are placed curvature-adaptively (spacing ~
sqrt(radius)) and very sparsely outside the view, which keeps 8 keyframes
around 200 KB.

Colour: every line is stroked with one repeating linear gradient, mostly the
neutral line colour with a soft warm band; the gradient is translated by
exactly one period per loop, so a slanted band of light drifts across the map.

Usage:  python generate.py       -> banner-dark.svg, banner-light.svg
"""
import math
import os

import numpy as np
from skimage import measure

HERE = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------- parameters
W, H = 1200, 340            # viewBox
PAD = 150                   # extended domain margin (field is closed off there)
GRID = 2.5                  # marching-squares grid step (px)
SEED = int(os.environ.get('TOPO_SEED', 5))
NFEAT = 11                  # random Fourier features in the terrain
LEVEL_STEP = 0.066          # contour interval (field units)
LEVEL_OFFSET = 0.012
INDEX_EVERY = 5             # every n-th line is an index contour
MIN_AREA = 160.0            # drop specks smaller than this (px^2)

KEYFRAMES = 8               # unique keyframes per loop
DUR = 48                    # seconds per loop
SPACING_MAX = float(os.environ.get('SMAX', 44))   # spacing on straight runs (px)
SPACING_MIN = 6.0           # ... and in the tightest bends
CURV_C = float(os.environ.get('CURVC', 4.0))      # spacing = sqrt(CURV_C * radius)
SPACING_OUT = float(os.environ.get('SOUT', 250))         # control-point spacing far outside the view (px)
Q = int(os.environ.get('Q', 2))                      # control points quantised to 1/Q px; the path is
                            # written in 1/(2Q) px units so on-curve midpoints
                            # are exact and the `t` chain cannot drift
SWEEP = 2600                # period (px) of the travelling light band

WARP_AMP = 1.0              # global scale of the animated warp


# ------------------------------------------------------------------- terrain
def terrain_params(seed):
    rng = np.random.default_rng(seed)
    feats = []
    for _ in range(NFEAT):
        lam = math.exp(rng.uniform(math.log(320), math.log(1150)))
        ang = rng.uniform(0, math.pi)
        k = 2 * math.pi / lam
        feats.append((k * math.cos(ang), k * math.sin(ang) * 1.35,
                      (lam / 1000.0) ** 1.15, rng.uniform(0, 2 * math.pi)))
        rng.uniform(0, 1), rng.uniform(0, 1)       # keep stream stable
    warp = []
    for _ in range(4):
        lam = rng.uniform(380, 700)
        ang = rng.uniform(0, 2 * math.pi)
        k = 2 * math.pi / lam
        warp.append((k * math.cos(ang), k * math.sin(ang),
                     rng.uniform(0, 2 * math.pi)))
        rng.uniform(0, 1)
    return feats, warp


FEATS, TWARP = terrain_params(SEED)

# Gaussian massifs anchoring the composition: (cx, cy, sx, sy, rot, amp)
MASSIFS = [
    (835, 150, 210, 125, 0.35, 1.05),
    (255, 250, 230, 140, -0.2, 0.55),
    (560, 20, 160, 110, 0.0, -0.45),
]


def terrain(X, Y):
    wx = np.zeros_like(X)
    wy = np.zeros_like(X)
    for i, (kx, ky, ph) in enumerate(TWARP):
        s = np.sin(kx * X + ky * Y + ph)
        if i % 2 == 0:
            wx += 26 * s
        else:
            wy += 18 * s
    Xw, Yw = X + wx, Y + wy
    f = np.zeros_like(X)
    for kx, ky, amp, ph in FEATS:
        f += amp * np.cos(kx * Xw + ky * Yw + ph)
    f *= 0.42
    for cx, cy, sx, sy, rot, amp in MASSIFS:
        c, s = math.cos(rot), math.sin(rot)
        dx, dy = Xw - cx, Yw - cy
        u = (c * dx + s * dy) / sx
        v = (-s * dx + c * dy) / sy
        f += amp * np.exp(-0.5 * (u * u + v * v))
    return f


XS = np.arange(-PAD, W + PAD + 1e-6, GRID)
YS = np.arange(-PAD, H + PAD + 1e-6, GRID)
GX, GY = np.meshgrid(XS, YS)


def closed_terrain():
    """Terrain inside view(+margin), smoothly sinking below every contour
    level towards the extended border so all contours are closed loops."""
    f = terrain(GX, GY)
    inner = (GX >= 0) & (GX <= W) & (GY >= 0) & (GY <= H)
    lo, hi = f[inner].min(), f[inner].max()
    m_in, m_out = 70.0, PAD - 10.0
    dx = np.maximum(np.maximum(-GX, GX - W), 0)
    dy = np.maximum(np.maximum(-GY, GY - H), 0)
    s = np.clip((np.hypot(dx, dy) - m_in) / (m_out - m_in), 0, 1)
    win = 1 - s * s * (3 - 2 * s)
    return win * f + (1 - win) * (lo - 1.0), lo, hi


# --------------------------------------------------------------------- warp
def warp_params(seed):
    rng = np.random.default_rng(seed + 1000)
    comps = []
    # (kx, ky, ax, ay, phase, temporal harmonic)
    for lam, amp, n in [(900, 13, 1), (700, 11, -1), (560, 9, 1),
                        (480, 7, 1), (1300, 12, -1)]:
        ang = rng.uniform(0, 2 * math.pi)
        k = 2 * math.pi / lam
        d = rng.uniform(0, 2 * math.pi)
        comps.append((k * math.cos(ang), k * math.sin(ang) * 1.6,
                      amp * math.cos(d), amp * math.sin(d) * 0.75,
                      rng.uniform(0, 2 * math.pi), n))
    return comps


WARP = warp_params(SEED)


def displacement(P, t):
    """Smooth periodic displacement E(p, t) (period 1)."""
    tau = 2 * math.pi * t
    x, y = P[:, 0], P[:, 1]
    ex = np.zeros_like(x)
    ey = np.zeros_like(y)
    for kx, ky, ax, ay, ph, n in WARP:
        c = np.cos(kx * x + ky * y + ph + n * tau)
        ex += ax * c
        ey += ay * c
    return WARP_AMP * np.stack([ex, ey], axis=1)


def check_diffeo():
    """Minimum Jacobian determinant of p -> p + E(p, t) over the view."""
    xs = np.arange(-60, W + 60, 6.0)
    ys = np.arange(-60, H + 60, 6.0)
    X, Y = np.meshgrid(xs, ys)
    P = np.stack([X.ravel(), Y.ravel()], axis=1)
    worst = 9.0
    h = 0.5
    for t in np.arange(32) / 32:
        ex1 = displacement(P + [h, 0], t) - displacement(P - [h, 0], t)
        ey1 = displacement(P + [0, h], t) - displacement(P - [0, h], t)
        J11 = 1 + ex1[:, 0] / (2 * h)
        J21 = ex1[:, 1] / (2 * h)
        J12 = ey1[:, 0] / (2 * h)
        J22 = 1 + ey1[:, 1] / (2 * h)
        worst = min(worst, float((J11 * J22 - J12 * J21).min()))
    return worst


# ----------------------------------------------------------------- geometry
def to_xy(c):
    return np.stack([XS[0] + c[:, 1] * GRID, YS[0] + c[:, 0] * GRID], axis=1)


def poly_area(p):
    x, y = p[:, 0], p[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


def curvature(q):
    """|curvature| at the segment midpoints of closed polyline q (q[-1]==q[0]),
    from a lightly smoothed copy so marching-squares jitter is ignored."""
    p = q[:-1]
    k = np.arange(-6, 7)
    g = np.exp(-0.5 * (k / 2.5) ** 2)
    g /= g.sum()
    p = sum(g[i] * np.roll(p, -k[i], axis=0) for i in range(len(k)))
    d1 = (np.roll(p, -1, axis=0) - np.roll(p, 1, axis=0)) / 2
    d2 = np.roll(p, -1, axis=0) - 2 * p + np.roll(p, 1, axis=0)
    num = np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0])
    kap = num / np.maximum(np.hypot(*d1.T) ** 3, 1e-9)
    kap = np.convolve(np.r_[kap[-8:], kap, kap[:8]], np.ones(17) / 17, 'same')[8:-8]
    return 0.5 * (kap + np.roll(kap, -1))


def density(p, kap):
    """Control-point density: curvature-adaptive inside the view, sparse far
    outside it."""
    dx = np.maximum(np.maximum(-p[:, 0], p[:, 0] - W), 0)
    dy = np.maximum(np.maximum(-p[:, 1], p[:, 1] - H), 0)
    s = np.clip((np.hypot(dx, dy) - 45) / 50, 0, 1)
    inside = np.clip(np.sqrt(kap / CURV_C), 1 / SPACING_MAX, 1 / SPACING_MIN)
    return (1 - s) * inside + s / SPACING_OUT


def resample(p):
    """Density-weighted arclength resample of a closed polyline, then a
    light circular smoothing (in index space) of the control polygon."""
    q = np.vstack([p, p[:1]])
    seg = np.hypot(*np.diff(q, axis=0).T)
    w = seg * density(0.5 * (q[1:] + q[:-1]), curvature(q))
    s = np.concatenate([[0], np.cumsum(w)])
    n = max(8, int(round(s[-1])))
    u = np.arange(n) / n * s[-1]
    r = np.stack([np.interp(u, s, q[:, 0]), np.interp(u, s, q[:, 1])], axis=1)
    k = np.arange(-3, 4)
    g = np.exp(-0.5 * (k / 0.6) ** 2)
    g /= g.sum()
    return sum(g[i] * np.roll(r, -k[i], axis=0) for i in range(len(k)))


def in_view(p, margin=4):
    return np.any((p[:, 0] > -margin) & (p[:, 0] < W + margin)
                  & (p[:, 1] > -margin) & (p[:, 1] < H + margin))


def build_contours():
    F, lo, hi = closed_terrain()
    levels = np.arange(lo + LEVEL_OFFSET, hi, LEVEL_STEP)
    ts = np.arange(KEYFRAMES) / KEYFRAMES
    tracks = []
    for li, L in enumerate(levels):
        for c in measure.find_contours(F, L):
            p = to_xy(c)
            assert np.hypot(*(p[0] - p[-1])) < 1e-6, 'open contour'
            p = p[:-1]
            a = poly_area(p)
            if abs(a) < MIN_AREA:
                continue
            if a < 0:
                p = p[::-1]
            ctrl = resample(p)
            frames = [ctrl + displacement(ctrl, t) for t in ts]
            if not any(in_view(f) for f in frames):
                continue
            tracks.append(dict(li=li, level=L, frames=frames,
                               index=(li % INDEX_EVERY) == 2))
    return tracks, levels


# ---------------------------------------------------------------- svg output
def num_list(vals):
    s = ''
    for i, v in enumerate(vals):
        v = int(v)
        s += str(v) if (i == 0 or v < 0) else ' ' + str(v)
    return s


def path_d(ctrl):
    """Closed quadratic B-spline with control polygon `ctrl`:
    M m[n-1] q p0 m0 t m1 m2 ... m[n-1] z   (relative, quantised)."""
    P = 2 * np.round(ctrl * Q).astype(np.int64)
    M = (P + np.roll(P, -1, axis=0)) // 2          # exact: P is even
    start = M[-1]
    out = f'M{start[0]} {start[1]}'
    out += 'q' + num_list([*(P[0] - start), *(M[0] - start)])
    deltas = np.diff(M, axis=0).ravel()
    out += 't' + num_list(deltas) + 'z'
    return out


THEMES = {
    'dark': dict(
        bg='#0d1117', border='#30363d',
        line='#8796ab', accent='#f3b37a',
        line_op=0.33, index_op=0.60, base=0.62,
    ),
    'light': dict(
        bg='#ffffff', border='#d0d7de',
        line='#4f5b6a', accent='#c8642a',
        line_op=0.36, index_op=0.52, base=0.62,
    ),
}


def build_svg(tracks, theme):
    th = THEMES[theme]
    U = 2 * Q                      # path units per px
    body = []
    for tr in tracks:
        frames = [path_d(p) for p in tr['frames']]
        vals = ';'.join(frames + [frames[0]])
        cls = ' class="i"' if tr['index'] else ''
        body.append(f'<path{cls} d="{frames[0]}"><animate attributeName="d" '
                    f'dur="{DUR}s" repeatCount="indefinite" values="{vals}"/>'
                    f'</path>')
    # A soft band of warm light travels slowly across the map: a repeating
    # linear gradient (period SWEEP px) translated by exactly one period per
    # loop, skewed so the band is slanted.
    P = SWEEP * U
    L, A, B = th['line'], th['accent'], th['base']
    grad = (
        f'<linearGradient id="g" gradientUnits="userSpaceOnUse" x1="0" y1="0" '
        f'x2="{P}" y2="0" spreadMethod="repeat" gradientTransform="skewX(-28)">'
        f'<stop offset="0" stop-color="{L}" stop-opacity="{B}"/>'
        f'<stop offset=".32" stop-color="{L}" stop-opacity="{B}"/>'
        f'<stop offset=".45" stop-color="{A}"/>'
        f'<stop offset=".5" stop-color="{A}"/>'
        f'<stop offset=".63" stop-color="{L}" stop-opacity="{B}"/>'
        f'<stop offset="1" stop-color="{L}" stop-opacity="{B}"/>'
        f'<animateTransform attributeName="gradientTransform" type="translate" '
        f'from="{-P // 2}" to="{P // 2}" dur="{DUR}s" additive="sum" '
        f'repeatCount="indefinite"/>'
        f'</linearGradient>'
    )
    css = (
        f"#c path{{fill:none;stroke:url(#g);stroke-linejoin:round;"
        f"stroke-width:{0.85 * U:g};stroke-opacity:{th['line_op'] / B:.3g}}}"
        f"#c path.i{{stroke-width:{1.4 * U:g};stroke-opacity:{min(1, th['index_op'] / B):.3g}}}"
    )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">
<style>{css}</style>
<defs>
<clipPath id="clip"><rect width="{W}" height="{H}" rx="16"/></clipPath>
</defs>
<rect width="{W}" height="{H}" rx="16" fill="{th['bg']}"/>
<g clip-path="url(#clip)">
<g id="c" transform="scale({1 / U:g})">
<defs>{grad}</defs>
{chr(10).join(body)}
</g>
</g>
<rect x=".5" y=".5" width="{W - 1}" height="{H - 1}" rx="15.5" fill="none" stroke="{th['border']}"/>
</svg>
"""
    return svg


def main():
    jd = check_diffeo()
    print(f'min Jacobian det of warp: {jd:.3f}')
    assert jd > 0.2, 'warp folds over - reduce WARP_AMP'
    tracks, levels = build_contours()
    npts = sum(len(t['frames'][0]) for t in tracks)
    print(f'{len(levels)} levels, {len(tracks)} paths, {npts} pts/frame')
    for theme in THEMES:
        svg = build_svg(tracks, theme)
        fn = os.path.join(HERE, f'banner-{theme}.svg')
        with open(fn, 'w') as fh:
            fh.write(svg)
        print(f'{fn}: {len(svg.encode()) / 1024:.1f} KB')


if __name__ == '__main__':
    main()
