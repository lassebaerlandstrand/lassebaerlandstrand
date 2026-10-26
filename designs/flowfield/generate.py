#!/usr/bin/env python3
"""Animated flow-field banner generator.

Approach
--------
A steady, incompressible 2D flow is defined by a stream function
    psi = Im(w(z)) + noise,
where w(z) is the complex potential of a slightly inclined uniform stream past
a few invisible circular obstacles (doublets, one with a little circulation),
and `noise` is a low-amplitude sum of random Fourier modes that is damped near
each obstacle so the bodies stay clean.  Velocity = (dpsi/dy, -dpsi/dx) is
evaluated on a 1 px grid.

Streamlines are placed with the Jobard-Lefer evenly-spaced algorithm: every
accepted line spawns candidate seeds at +-DSEP along its normal; a new line is
integrated (RK4, unit speed) in both directions until it gets closer than
DTEST to any other line, enters an obstacle or leaves the domain.  The result
is a calm, plotter-like hatching with constant line spacing.

Motion: the lines are static.  A subset of them carries short "comet" dashes
(three stacked stroke-dash layers: long faint tail, mid glow, bright head)
that travel downstream via SMIL stroke-dashoffset animation.  Each line's
dash period P is chosen so that an integer number of periods passes per loop
at a speed proportional to the mean flow speed along that line, so the loop
is exact and seamless.

Usage:  python generate.py        -> banner-dark.svg, banner-light.svg
"""
import heapq
import math
import os
import random

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------- parameters
W, H = 1200, 340
PAD = 24                    # integration domain extends past the view
SEED = int(os.environ.get('FF_SEED', 4))

DSEP = float(os.environ.get('DSEP', 7.0))      # line spacing (px)
DTEST = 0.55 * DSEP         # a line stops when this close to another one
STEP = 1.0                  # RK4 step (px, unit-speed integration)
MIN_LEN = 40.0              # drop lines shorter than this (px)
RDP_TOL = 0.12              # polyline simplification tolerance (px)
SCALE = 4                   # path coords are integers in 1/SCALE px

ANGLE = math.radians(float(os.environ.get('ANG', -6.0)))  # inclination of the free stream (y is down)
# obstacles: (cx, cy, radius, circulation)
BODIES = [
    (290.0, 214.0, 34.0, 0.0),
    (720.0, 104.0, 24.0, 0.0),
    (1000.0, 246.0, 15.0, 0.0),
]
BODY_GAP = 3.0              # keep lines this far off the surface
NOISE_AMP = float(os.environ.get('NAMP', 0.65))           # rough |v_noise| / |U|
NMODES = 8
LAM = (300.0, 1000.0)       # noise wavelengths (px)
RAMP = 2.4                  # obstacle influence radius, in units of R

DUR = 30                    # loop length (s)
ANIM_FRAC = 0.6            # fraction of (long enough) lines with dashes
SPEED = 34.0                # px/s at free-stream speed
P_RANGE = (800.0, 1500.0)    # allowed dash period (px)
GROUP_OP = (0.62, 0.82, 1.0)  # static-line opacity factor: slow / mid / fast


# ---------------------------------------------------------------------- field
def make_field(seed):
    rng = np.random.default_rng(seed)
    x0, y0 = -PAD, -PAD
    nx, ny = W + 2 * PAD + 1, H + 2 * PAD + 1
    xs = np.arange(nx, dtype=float) + x0
    ys = np.arange(ny, dtype=float) + y0
    X, Y = np.meshgrid(xs, ys)
    Z = X + 1j * Y

    # free stream + random Fourier modes (divergence free by construction)
    psi = math.cos(ANGLE) * Y - math.sin(ANGLE) * X
    for _ in range(NMODES):
        lam = math.exp(rng.uniform(math.log(LAM[0]), math.log(LAM[1])))
        a = rng.uniform(0, 2 * math.pi)
        k = 2 * math.pi / lam
        kx, ky = k * math.cos(a), k * math.sin(a) * 1.3
        amp = NOISE_AMP / math.sqrt(NMODES) / k * rng.uniform(0.6, 1.0)
        psi = psi + amp * np.sin(kx * X + ky * Y + rng.uniform(0, 2 * math.pi))

    # obstacles (Bridson-style): blend psi to a constant across a smooth
    # ramp so each circle is a streamline and the flow slides around it.
    for cx, cy, R, G in BODIES:
        r = np.hypot(X - cx, Y - cy)
        th = np.linspace(0, 2 * math.pi, 96, endpoint=False)
        px = np.clip(np.round(cx + R * np.cos(th)) - x0, 0, nx - 1).astype(int)
        py = np.clip(np.round(cy + R * np.sin(th)) - y0, 0, ny - 1).astype(int)
        psib = psi[py, px].mean()
        s = np.clip((r - R) / (RAMP * R), 0, 1)
        ramp = 15 / 8 * s - 10 / 8 * s ** 3 + 3 / 8 * s ** 5
        psi = psib + (psi - psib) * ramp + G * np.log(np.maximum(r, R) / R) * (1 - ramp)

    gy, gx = np.gradient(psi)
    u, v = gy, -gx
    return dict(x0=x0, y0=y0, nx=nx, ny=ny, u=u, v=v,
                speed=np.hypot(u, v))


class Sampler:
    def __init__(self, f):
        self.f = f
        self.u = f['u'].tolist()
        self.v = f['v'].tolist()
        self.x0, self.y0 = f['x0'], f['y0']
        self.nx, self.ny = f['nx'], f['ny']

    def vel(self, x, y):
        gx, gy = x - self.x0, y - self.y0
        if gx < 0 or gy < 0 or gx >= self.nx - 1 or gy >= self.ny - 1:
            return None
        i, j = int(gx), int(gy)
        fx, fy = gx - i, gy - j
        u, v = self.u, self.v
        a = u[j][i] * (1 - fx) + u[j][i + 1] * fx
        b = u[j + 1][i] * (1 - fx) + u[j + 1][i + 1] * fx
        uu = a * (1 - fy) + b * fy
        a = v[j][i] * (1 - fx) + v[j][i + 1] * fx
        b = v[j + 1][i] * (1 - fx) + v[j + 1][i + 1] * fx
        vv = a * (1 - fy) + b * fy
        return uu, vv

    def dir(self, x, y, sgn):
        q = self.vel(x, y)
        if q is None:
            return None
        s = math.hypot(*q)
        if s < 0.04:
            return None
        return sgn * q[0] / s, sgn * q[1] / s


def inside_body(x, y, gap=BODY_GAP):
    for cx, cy, R, _ in BODIES:
        if (x - cx) ** 2 + (y - cy) ** 2 < (R + gap) ** 2:
            return True
    return False


# ------------------------------------------------------------ Jobard-Lefer
class Grid:
    def __init__(self, cell):
        self.cell = cell
        self.d = {}

    def key(self, x, y):
        return int(math.floor(x / self.cell)), int(math.floor(y / self.cell))

    def add(self, x, y, lid, idx):
        self.d.setdefault(self.key(x, y), []).append((x, y, lid, idx))

    def near(self, x, y, r, lid=None, idx=None, selfgap=0):
        kx, ky = self.key(x, y)
        n = int(math.ceil(r / self.cell))
        r2 = r * r
        for i in range(kx - n, kx + n + 1):
            for j in range(ky - n, ky + n + 1):
                for (px, py, l, k) in self.d.get((i, j), ()):
                    if l == lid and abs(k - idx) < selfgap:
                        continue
                    if (px - x) ** 2 + (py - y) ** 2 < r2:
                        return True
        return False


def integrate(S, grid, x, y, sgn, lid, maxn=4000):
    pts = []
    own = Grid(DSEP)
    selfgap = int(3 * DSEP / STEP)
    for n in range(maxn):
        k1 = S.dir(x, y, sgn)
        if k1 is None:
            break
        h = STEP
        k2 = S.dir(x + 0.5 * h * k1[0], y + 0.5 * h * k1[1], sgn)
        if k2 is None:
            break
        k3 = S.dir(x + 0.5 * h * k2[0], y + 0.5 * h * k2[1], sgn)
        if k3 is None:
            break
        k4 = S.dir(x + h * k3[0], y + h * k3[1], sgn)
        if k4 is None:
            break
        nx_ = x + h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        ny_ = y + h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        if inside_body(nx_, ny_):
            break
        # proximity to other lines (and to far-away parts of this one)
        sidx = n + 1 if sgn > 0 else -(n + 1)
        if grid.near(nx_, ny_, DTEST, lid, sidx, selfgap):
            break
        # self-proximity for the line being built (loops / eddies)
        if own.near(nx_, ny_, DTEST, -1, n, selfgap):
            break
        x, y = nx_, ny_
        pts.append((x, y))
        own.add(x, y, -1, n)
    return pts


def streamline(S, grid, x, y, lid):
    fwd = integrate(S, grid, x, y, +1, lid)
    bwd = integrate(S, grid, x, y, -1, lid)
    return bwd[::-1] + [(x, y)] + fwd


def seed_ok(S, grid, x, y):
    if not (-PAD < x < W + PAD and -PAD < y < H + PAD):
        return False
    if inside_body(x, y, BODY_GAP + 1):
        return False
    if S.dir(x, y, 1) is None:
        return False
    return not grid.near(x, y, DSEP * 0.98)


def place_streamlines(S, seed):
    rnd = random.Random(seed)
    grid = Grid(DSEP)
    lines = []
    queue = []

    def accept(pts):
        lid = len(lines)
        arr = np.array(pts)
        if len(arr) < 2:
            return False
        L = np.hypot(*np.diff(arr, axis=0).T).sum()
        if L < MIN_LEN:
            return False
        for i, (px, py) in enumerate(pts):
            grid.add(px, py, lid, 0)   # other-line check only
        lines.append(arr)
        queue.append(lid)
        return True

    # first line through the centre of the view
    accept(streamline(S, grid, W / 2, H / 2 + 3, 0))
    qi = 0
    while True:
        while qi < len(queue):
            arr = lines[queue[qi]]
            qi += 1
            for i in range(0, len(arr), 2):
                if i == 0 or i == len(arr) - 1:
                    continue
                d = arr[min(i + 1, len(arr) - 1)] - arr[max(i - 1, 0)]
                nrm = math.hypot(*d)
                if nrm < 1e-9:
                    continue
                nx_, ny_ = -d[1] / nrm, d[0] / nrm
                for s in (+1, -1):
                    sx = arr[i][0] + s * DSEP * nx_
                    sy = arr[i][1] + s * DSEP * ny_
                    if seed_ok(S, grid, sx, sy):
                        accept(streamline(S, grid, sx, sy, len(lines)))
        # fill any remaining holes with random seeds
        added = False
        for _ in range(4000):
            sx = rnd.uniform(-PAD + 1, W + PAD - 1)
            sy = rnd.uniform(-PAD + 1, H + PAD - 1)
            if seed_ok(S, grid, sx, sy):
                if accept(streamline(S, grid, sx, sy, len(lines))):
                    added = True
                    break
        if not added:
            break
    return lines


# ------------------------------------------------------------- geometry/out
def join_lines(lines, reach=2.4 * DSEP):
    """Jobard-Lefer leaves small gaps where one line stops and another one
    starts a little further along the same track.  Join such end->start
    pairs when they are close and well aligned (all lines run downstream)."""
    def tang(p, at_end):
        a, b = (p[-6], p[-1]) if at_end else (p[0], p[5])
        d = b - a
        return d / max(np.hypot(*d), 1e-9)

    lines = [p for p in lines]
    changed = True
    while changed:
        changed = False
        starts = np.array([p[0] for p in lines])
        for i in range(len(lines)):
            p = lines[i]
            if p is None or len(p) < 6:
                continue
            e, te = p[-1], tang(p, True)
            d = starts - e
            dist = np.hypot(d[:, 0], d[:, 1])
            best, bj = None, -1
            for j in np.nonzero(dist < reach)[0]:
                q = lines[j]
                if j == i or q is None or len(q) < 6 or dist[j] < 1e-6:
                    continue
                g = d[j] / dist[j]
                if g @ te > 0.97 and tang(q, False) @ te > 0.97:
                    if best is None or dist[j] < best:
                        best, bj = dist[j], j
            if bj >= 0:
                lines[i] = np.vstack([p, lines[bj]])
                lines[bj] = None
                starts[bj] = (1e9, 1e9)
                changed = True
        lines = [p for p in lines if p is not None]
    return lines


def rdp(p, tol):
    keep = np.zeros(len(p), bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(p) - 1)]
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        A, B = p[a], p[b]
        d = B - A
        L = math.hypot(*d)
        seg = p[a + 1:b]
        if L < 1e-9:
            dist = np.hypot(*(seg - A).T)
        else:
            dist = np.abs(d[0] * (seg[:, 1] - A[1]) - d[1] * (seg[:, 0] - A[0])) / L
        k = int(np.argmax(dist))
        if dist[k] > tol:
            m = a + 1 + k
            keep[m] = True
            stack += [(a, m), (m, b)]
    return p[keep]


def clip_to_view(p, m=6):
    """Trim the parts of a line that run far outside the view."""
    ins = (p[:, 0] > -m) & (p[:, 0] < W + m) & (p[:, 1] > -m) & (p[:, 1] < H + m)
    if not ins.any():
        return None
    idx = np.nonzero(ins)[0]
    a, b = max(idx[0] - 1, 0), min(idx[-1] + 2, len(p))
    return p[a:b]


def path_d(p):
    q = np.round(p * SCALE).astype(np.int64)
    out = f'M{q[0][0]} {q[0][1]}l'
    d = np.diff(q, axis=0)
    s = ''
    first = True
    for dx, dy in d:
        for v in (dx, dy):
            v = int(v)
            if first or v < 0:
                s += str(v)
            else:
                s += ' ' + str(v)
            first = False
    return out + s


THEMES = {
    'dark': dict(
        bg='#0d1117', border='#30363d',
        line='#7d8ba0', line_op=0.52,
        accent='#f2c38f',
        # comet layers: (length px, width px, opacity), back to front
        comet=[(150, 1.0, 0.12), (64, 1.25, 0.28), (16, 1.7, 0.95)],
    ),
    'light': dict(
        bg='#ffffff', border='#d0d7de',
        line='#5a6676', line_op=0.48,
        accent='#c8642a',
        comet=[(150, 1.0, 0.13), (64, 1.25, 0.30), (16, 1.7, 0.95)],
    ),
}


def line_speed(p, S):
    seg = np.hypot(*np.diff(p, axis=0).T)
    mid = 0.5 * (p[1:] + p[:-1])
    sp = []
    for x, y in mid:
        q = S.vel(x, y)
        sp.append(math.hypot(*q) if q else 1.0)
    return float(np.dot(sp, seg) / max(seg.sum(), 1e-9))


def plan_motion(lines, S, seed):
    rnd = random.Random(seed + 101)
    plans = []
    for li, p in enumerate(lines):
        seg = np.hypot(*np.diff(p, axis=0).T)
        L = seg.sum()
        if L < 140 or rnd.random() > ANIM_FRAC:
            plans.append(None)
            continue
        vmean = line_speed(p, S)
        v = SPEED * min(max(vmean, 0.55), 1.6) * rnd.uniform(0.85, 1.15)
        travel = v * DUR                       # px per loop
        # integer number of periods per loop, period within P_RANGE
        k = max(1, round(travel / rnd.uniform(*P_RANGE)))
        P = travel / k
        phase = rnd.uniform(0, P)
        plans.append(dict(P=P, k=k, phase=phase))
    return plans


def anim(attr_from, attr_to):
    return (f'<animate attributeName="stroke-dashoffset" from="{attr_from}" '
            f'to="{attr_to}" dur="{DUR}s" repeatCount="indefinite"/>')


def build_svg(lines, plans, speeds, theme):
    th = THEMES[theme]
    U = SCALE
    ds = [path_d(p) for p in lines]
    # static lines in three brightness groups by mean flow speed: the
    # faster (squeezed) lines read slightly brighter
    q1, q2 = np.quantile(speeds, [1 / 3, 2 / 3])
    groups = ['', '', '']
    for d, v in zip(ds, speeds):
        groups[0 if v < q1 else 1 if v < q2 else 2] += d
    layers = [[] for _ in th['comet']]
    Lmax = th['comet'][0][0]
    for d, pl in zip(ds, plans):
        if pl is None:
            continue
        # integers throughout so the dash period and the travel per loop
        # (k periods) match exactly -> seamless loop
        P, k, ph = round(pl['P'] * U), pl['k'], round(pl['phase'] * U)
        # every layer's dash ends at the same leading point
        for li, (ln, _, _) in enumerate(th['comet']):
            t = round(ln * U)
            o = -ph - (round(Lmax * U) - t)
            layers[li].append(
                f'<path d="{d}" stroke-dasharray="{t} {P - t}">'
                + anim(o, o - k * P) + '</path>')

    sw = 0.8 * U
    vign = (
        '<radialGradient id="v" gradientUnits="userSpaceOnUse" '
        f'cx="{W/2*U:.0f}" cy="{H/2*U:.0f}" r="{W*0.62*U:.0f}" '
        f'gradientTransform="translate({W/2*U:.0f} {H/2*U:.0f}) scale(1 0.55) '
        f'translate({-W/2*U:.0f} {-H/2*U:.0f})">'
        f'<stop offset="0" stop-color="{th["line"]}" stop-opacity="{th["line_op"]}"/>'
        f'<stop offset=".7" stop-color="{th["line"]}" stop-opacity="{th["line_op"]*0.8:.3f}"/>'
        f'<stop offset="1" stop-color="{th["line"]}" stop-opacity="{th["line_op"]*0.45:.3f}"/>'
        '</radialGradient>')
    css = (
        f'#s path{{fill:none;stroke:url(#v);stroke-width:{sw:.1f};stroke-linecap:round;stroke-linejoin:round}}'
        f'#s .a{{stroke-opacity:{GROUP_OP[0]}}}#s .b{{stroke-opacity:{GROUP_OP[1]}}}'
    )
    for li, (_, w, op) in enumerate(th['comet']):
        css += (f'#c{li} path{{fill:none;stroke:{th["accent"]};stroke-opacity:{op};'
                f'stroke-width:{w*U:.1f};stroke-linecap:round}}')
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">',
        f'<style>{css}</style>',
        f'<defs><clipPath id="clip"><rect width="{W}" height="{H}" rx="16"/></clipPath>{vign}</defs>',
        f'<rect width="{W}" height="{H}" rx="16" fill="{th["bg"]}"/>',
        '<g clip-path="url(#clip)">',
        f'<g transform="scale({1/U})">',
        '<g id="s">' + ''.join(f'<path class="{c}" d="{g}"/>' for c, g in zip('abc', groups)) + '</g>',
        *[f'<g id="c{li}">' + ''.join(L) + '</g>' for li, L in enumerate(layers)],
        '</g></g>',
        f'<rect x=".5" y=".5" width="{W-1}" height="{H-1}" rx="15.5" fill="none" stroke="{th["border"]}"/>',
        '</svg>',
    ]
    return '\n'.join(out)


def main():
    f = make_field(SEED)
    S = Sampler(f)
    raw = place_streamlines(S, SEED)
    n0 = len(raw)
    raw = join_lines(raw)
    print(f'joined {n0 - len(raw)} gaps')
    lines = []
    for p in raw:
        c = clip_to_view(p)
        if c is None or len(c) < 2:
            continue
        lines.append(rdp(c, RDP_TOL))
    # rebuild full-resolution copies for speed sampling via the simplified pts
    plans = plan_motion(lines, S, SEED)
    speeds = [line_speed(p, S) for p in lines]
    print(f'{len(raw)} raw lines, {len(lines)} kept, '
          f'{sum(p is not None for p in plans)} animated, '
          f'{sum(len(p) for p in lines)} vertices')
    for theme in THEMES:
        svg = build_svg(lines, plans, speeds, theme)
        fn = os.path.join(os.environ.get('OUT', HERE), f'banner-{theme}.svg')
        with open(fn, 'w') as fh:
            fh.write(svg)
        print(fn, len(svg.encode()) // 1024, 'KB')


if __name__ == '__main__':
    main()
