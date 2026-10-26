#!/usr/bin/env python3
"""Animated fjord + Viking longship banner, drawn as a copperplate engraving.

A line-art variant of ../longship: the same procedural fjord, the same
Gokstad/Oseberg longship and the same motion, but every tone is built from
engraved lines instead of fills.

Scene (back to front)
---------------------
* sky: ruled horizontal lines whose weight follows the sky's tone; they swell
  into the glow around the moon (dark) or thin away around the sun (light);
* seven fjord walls (glacial U-valley profiles), each hatched with lines that
  follow the terrain: level sets of a smoothed distance-to-skyline field, so
  the hatching runs parallel to the ridge and wraps around the cliff edges.
  Line weight carries the shading (aerial perspective, faces turned to the
  light, haze at the foot); hidden parts are cut exactly in Python, so there
  are no occluding fills;
* drifting mist: a soft veil in the background colour (the hatching thins
  under it) carrying a few fine horizontal wisps;
* water: horizontal ripple lines in perspective whose weight encodes the
  mirrored mountains and the light's glade; each line drifts in its own small
  orbit (a slow swell), and glints twinkle on the glade;
* the ship's reflection, written as heavier strokes on the ripple rows that
  sway like the water; the Kelvin wake, oar puddles and stern trail as lines;
* the ship: ink outlines, strake hatching on the hull, the sail hatched along
  its cloths in an accent ink, rigging as hairlines, rowing oars.

Motion (all SMIL, seamless): the ship crosses once per LOOP seconds and
re-enters off-screen, the oars row at a stroke matched to the ship speed,
puddles stay put in the world, the ripple rows orbit, mist drifts, glints
twinkle, a waterfall runs.

Usage: python generate.py  -> banner-dark.svg, banner-light.svg
Needs numpy, scipy, scikit-image.
"""
import math
import os
from collections import defaultdict

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d, map_coordinates
from scipy.spatial import cKDTree
from skimage import measure

HERE = os.path.dirname(os.path.abspath(__file__))

W, H = 1200, 340
YH = 226.0          # horizon (water line of the distant shore)
YW = 276.0          # ship's water line
LIGHT_X, LIGHT_Y, LIGHT_R = 668.0, 150.0, 15.0

STROKE = 3.2        # seconds per oar stroke
X_START_VIS = 360.0 # ship centre at t = 0
WAKE_LEN = 240.0
SHIP_HALF = 108.0
X_A = -SHIP_HALF - 12
X_B = W + SHIP_HALF + 175
V = 23.0                 # px / s
LOOP = round((X_B - X_A) / V, 1)   # seconds per crossing
K_PERSP = 0.46           # screen-y per unit of depth toward the viewer


def f(v):
    s = f'{v:.1f}'
    if s.endswith('.0'):
        s = s[:-2]
    if s == '-0':
        s = '0'
    if s.startswith('0.'):
        s = s[1:]
    elif s.startswith('-0.'):
        s = '-' + s[2:]
    return s


def poly(pts, close=True):
    out = 'M' + ' '.join(f'{f(x)} {f(y)}' for x, y in pts[:1])
    out += 'L' + ' '.join(f'{f(x)} {f(y)}' for x, y in pts[1:])
    return out + ('Z' if close else '')


def smoothstep(a, b, x):
    t = np.clip((np.asarray(x, float) - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


# ------------------------------------------------------------------ line pen
def _i10(v):
    """Format an integer number of tenths."""
    if v % 10 == 0:
        return str(v // 10)
    s = f'{abs(v) / 10:.1f}'.lstrip('0')
    return ('-' if v < 0 else '') + s


def _join(nums):
    out = ''
    for i, n in enumerate(nums):
        s = _i10(int(n))
        out += s if (i == 0 or s[0] == '-') else ' ' + s
    return out


def rdp(p, eps):
    """Ramer-Douglas-Peucker simplification of an open polyline."""
    if len(p) < 3:
        return p
    keep = np.zeros(len(p), bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(p) - 1)]
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        d = p[b] - p[a]
        n = math.hypot(*d)
        q = p[a + 1:b] - p[a]
        if n < 1e-9:
            dist = np.hypot(q[:, 0], q[:, 1])
        else:
            dist = np.abs(q[:, 0] * d[1] - q[:, 1] * d[0]) / n
        k = int(np.argmax(dist))
        if dist[k] > eps:
            m = a + 1 + k
            keep[m] = True
            stack += [(a, m), (m, b)]
    return p[keep]


def encode(pts):
    P = np.round(np.asarray(pts) * 10).astype(np.int64)
    if len(P) > 1:
        k = np.r_[True, np.any(np.diff(P, axis=0) != 0, axis=1)]
        P = P[k]
    if len(P) < 2:
        return ''
    d = np.diff(P, axis=0)
    out = f'M{_i10(int(P[0, 0]))} {_i10(int(P[0, 1]))}'
    if np.all(d[:, 1] == 0):
        return out + 'h' + _i10(int(d[:, 0].sum()))
    return out + 'l' + _join(d.ravel())


class Pen:
    """Collects polylines of varying weight into one <path> per weight class."""

    def __init__(self, ratio=1.2, wmin=0.18, wmax=2.0, eps=0.12, hyst=0.8):
        self.ratio, self.wmin, self.wmax, self.eps, self.hyst = ratio, wmin, wmax, eps, hyst
        self.b = defaultdict(list)

    def line(self, pts, w):
        pts = np.asarray(pts, float)
        w = np.broadcast_to(np.asarray(w, float), (len(pts),))
        # quantise to weight classes with hysteresis, so a slowly varying
        # weight does not flicker between two classes
        # classes are geometric: w_k = 0.1 * RATIO**k (finer steps for hairlines)
        t = np.log(np.clip(w, 0.1, self.wmax) / 0.1) / math.log(self.ratio)
        q = np.zeros(len(t), int)
        cur = 0
        for k in range(len(t)):
            if w[k] < self.wmin:
                cur = 0
            elif cur == 0 or abs(t[k] - cur) > self.hyst:
                cur = max(1, int(round(t[k])))
            q[k] = cur
        n = len(pts)
        i = 0
        while i < n:
            if q[i] == 0:
                i += 1
                continue
            j = i
            while j + 1 < n and q[j + 1] == q[i]:
                j += 1
            seg = pts[i:j + 1]
            if i > 0 and q[i - 1] != 0:
                seg = np.vstack([0.5 * (pts[i - 1] + pts[i]), seg])
            if j + 1 < n and q[j + 1] != 0:
                seg = np.vstack([seg, 0.5 * (pts[j] + pts[j + 1])])
            if len(seg) >= 2:
                s = encode(rdp(seg, self.eps))
                if s:
                    self.b[q[i]].append(s)
            i = j + 1

    def svg(self):
        return ''.join(f'<path stroke-width="{0.1 * self.ratio ** k:.2f}" d="{"".join(v)}"/>'.replace('"0.', '".')
                       for k, v in sorted(self.b.items()))

    def empty(self):
        return not self.b


# ------------------------------------------------------------------ terrain
def value_noise(x, scale, seed):
    r = np.random.default_rng(seed)
    n = int(x.max() / scale) + 4
    lat = r.uniform(-1, 1, n)
    u = x / scale
    i = np.floor(u).astype(int)
    t = u - i
    t = t * t * t * (t * (t * 6 - 15) + 10)
    return lat[i] * (1 - t) + lat[i + 1] * t


def fbm(x, scale, seed, octaves=6, gain=0.5):
    out = np.zeros_like(x)
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        out += amp * value_noise(x, scale / 2 ** o, seed * 31 + o)
        tot += amp
        amp *= gain
    return out / tot


def cliff(x, c, w, side):
    """1 on the wall side, 0 on the water side; glacial U profile."""
    s = (x - c) / w * (1 if side == 'L' else -1)
    s = np.clip(s, -1.6, 1.6)
    return 0.5 * (1 - np.tanh(2.2 * s))


LAYERS = [
    dict(kind='range', top=166, seed=3),
    dict(side='R', c=712, w=34, top=148, rough=8, seed=11),
    dict(side='L', c=566, w=36, top=118, rough=10, seed=5),
    dict(side='R', c=872, w=34, top=86, rough=12, seed=21),
    dict(side='L', c=372, w=32, top=62, rough=13, seed=8),
    dict(side='R', c=1062, w=30, top=46, rough=14, seed=33, trend=0.1),
    dict(side='L', c=150, w=28, top=26, rough=14, seed=41, trend=0.08),
]


def layer_profile(L, x):
    x = np.asarray(x, float)
    xx = np.maximum(x, 0.0)          # noise lattice starts at 0
    if L.get('kind') == 'range':
        n = fbm(xx, 120, L['seed'], 7, 0.55)
        hump = np.exp(-((x - 700) / 230) ** 2)
        return YH - (YH - L['top']) * (0.45 + 0.55 * hump) * (0.7 + 0.3 * n)
    sd = L['seed']
    xw = x + 1.4 * L['w'] * fbm(xx, 110, sd + 7, 4)
    env = cliff(xw, L['c'], L['w'], L['side'])
    plate = L['top'] + L['rough'] * fbm(xx, 190, sd, 7, 0.55)
    r = np.random.default_rng(sd)
    inland = np.clip((L['c'] - x) if L['side'] == 'L' else (x - L['c']), 0, 500)
    plate -= L.get('trend', 0.05) * inland - 0.05 * 60
    for _ in range(4):
        px = L['c'] + (-1 if L['side'] == 'L' else 1) * r.uniform(1.5, 11) * L['w']
        plate -= r.uniform(4, 14) * np.exp(-((x - px) / r.uniform(14, 38)) ** 2)
    nx = L['c'] + (-1 if L['side'] == 'L' else 1) * r.uniform(3, 7) * L['w']
    plate += r.uniform(4, 9) * np.exp(-((x - nx) / r.uniform(10, 18)) ** 2)
    y = YH - (YH - plate) * env
    face = 4 * env * (1 - env)
    y += 7 * face * fbm(xx, 22, sd + 99, 4)
    y += 1.4 * fbm(xx, 9, sd + 5, 3) * env
    return np.minimum(y, YH)


# fine grid on which all terrain work is done
GX = np.arange(-8, W + 8.01, 1.0)
PROFILES = [layer_profile(L, GX) for L in LAYERS]
NLAY = len(LAYERS)
# occluding skyline in front of layer i (min over nearer layers)
FRONT = []
for i in range(NLAY):
    fr = np.full_like(GX, 1e9)
    for j in range(i + 1, NLAY):
        fr = np.minimum(fr, PROFILES[j])
    FRONT.append(fr)
SKYLINE = np.minimum.reduce(PROFILES)


def prof_at(i, x):
    return np.interp(x, GX, PROFILES[i])


def front_at(i, x):
    return np.interp(x, GX, FRONT[i])


def layer_at(x, y):
    """Index of the nearest layer covering screen point(s) (x, y), -1 = sky."""
    out = np.full(np.shape(x), -1)
    for i in range(NLAY):
        out = np.where(y >= prof_at(i, x), i, out)
    return out


# ------------------------------------------------------------------ ship
def bez(p0, p1, p2, p3, n):
    t = np.linspace(0, 1, n)[:, None]
    p0, p1, p2, p3 = map(np.array, (p0, p1, p2, p3))
    return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3


NRISE = 60


def stem_centerline(sign, base, c1, c2, centre, r0, turns):
    """Stem: raked sweep out of the water, then an Oseberg spiral. sign=+1 bow."""
    cx, cy = centre
    S = (cx + sign * r0, cy)
    rise = bez(base, (base[0] + sign * c1[0], base[1] - c1[1]),
               (S[0] + sign * c2[0], S[1] + c2[1]), S, NRISE)
    th = np.linspace(0, 2 * math.pi * turns, 90)
    r = r0 * (1 - 0.8 * th / th[-1]) ** 1.2
    sp = np.stack([cx + sign * r * np.cos(th), cy - r * np.sin(th)], 1)
    return np.vstack([rise, sp[1:]])


def tapered(cl, w0, w1):
    d = np.gradient(cl, axis=0)
    d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
    nrm = np.stack([-d[:, 1], d[:, 0]], 1)
    s = np.linspace(0, 1, len(cl))
    w = w0 + (w1 - w0) * s ** 0.7
    a = cl + nrm * (w / 2)[:, None]
    b = cl - nrm * (w / 2)[:, None]
    return np.vstack([a, b[::-1]])


def build_ship():
    bow = stem_centerline(+1, (70, 1.5), (18, 3), (1, 15), (91, -41), 8.6, 1.2)
    stern = stem_centerline(-1, (-70, 1.5), (17, 3), (1, 14), (-89, -38), 7.6, 1.15)
    HJ = -26.0

    def at_h(cl):
        i = int(np.argmin(np.abs(cl[:NRISE, 1] - HJ)))
        return i, cl[i]

    ib, pf = at_h(bow)
    ia, pa = at_h(stern)
    xs = np.linspace(pa[0], pf[0], 160)
    mid = (pa[0] + pf[0]) / 2
    u = (xs - mid) / ((pf[0] - pa[0]) / 2)
    sheer_y = -10.0 - (-HJ - 10.0) * np.abs(u) ** 2.5
    sheer = np.stack([xs, sheer_y], 1)
    hull = np.vstack([
        [[-70, 1.5], [70, 1.5]],
        bow[1:ib + 1],
        sheer[::-1][1:-1],
        stern[:ia + 1][::-1],
    ])
    return dict(bow=bow, stern=stern, hull=hull,
                bow_post=tapered(bow, 4.6, 1.4), stern_post=tapered(stern, 4.4, 1.3),
                sheer=sheer, xs=xs, sheer_at=lambda x: np.interp(x, xs, sheer_y))


SHIP = build_ship()

MAST_X = 4.0
MAST_TOP = -112.0
YARD_Y = -101.0
SAIL_HALF = 42.0
SAIL_BOT = -30.0

N_OARS = 14
OAR_X = np.linspace(-54, 56, N_OARS)
SHIELD_X = np.linspace(-63, 65, 21)
OAR_R = 44.0

SX0, SX1 = MAST_X - SAIL_HALF, MAST_X + SAIL_HALF
TOP_L, TOP_R = (SX0 - 1, YARD_Y + 1.2), (SX1 + 1, YARD_Y + 1.2)
BOT_L, BOT_R = (SX0 + 3, SAIL_BOT), (SX1 + 1, SAIL_BOT + 1)
SAIL_D = (f'M{f(TOP_L[0])} {f(TOP_L[1])}L{f(TOP_R[0])} {f(TOP_R[1])}'
          f'C{f(SX1 + 6)} -86 {f(SX1 + 6)} -52 {f(BOT_R[0])} {f(BOT_R[1])}'
          f'Q{f(MAST_X + 2)} {f(SAIL_BOT - 9)} {f(BOT_L[0])} {f(BOT_L[1])}'
          f'C{f(SX0 + 1)} -56 {f(SX0 - 3)} -86 {f(TOP_L[0])} {f(TOP_L[1])}Z')


def sail_patch(u, v):
    """Coons patch of the billowing sail; u across (0 = left), v down."""
    u = np.asarray(u, float)[..., None]
    v = np.asarray(v, float)[..., None]
    TL, TR, BL, BR = map(np.array, (TOP_L, TOP_R, BOT_L, BOT_R))

    def cub(p0, p1, p2, p3, t):
        return ((1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1
                + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3)

    R = cub(TR, np.array([SX1 + 6, -86]), np.array([SX1 + 6, -52]), BR, v)
    L = cub(TL, np.array([SX0 - 3, -86]), np.array([SX0 + 1, -56]), BL, v)
    Q = np.array([MAST_X + 2, SAIL_BOT - 9])
    B = (1 - u) ** 2 * BL + 2 * (1 - u) * u * Q + u ** 2 * BR
    T = (1 - u) * TL + u * TR
    return ((1 - v) * T + v * B + (1 - u) * L + u * R
            - ((1 - u) * (1 - v) * TL + u * (1 - v) * TR + (1 - u) * v * BL + u * v * BR))


def sail_outline():
    n = 60
    t = np.linspace(0, 1, n)
    top = sail_patch(t, 0 * t)
    right = sail_patch(1 + 0 * t, t)
    bot = sail_patch(t[::-1], 1 + 0 * t)
    left = sail_patch(0 * t, t[::-1])
    return np.vstack([top, right, bot, left])


SAIL_POLY = sail_outline()


def chords(polys, y):
    """Horizontal runs [x0, x1] where y crosses the union of polygons."""
    iv = []
    for P in polys:
        a, b = P, np.roll(P, -1, axis=0)
        m = (a[:, 1] - y) * (b[:, 1] - y) < 0
        if not m.any():
            continue
        xs = a[m, 0] + (y - a[m, 1]) * (b[m, 0] - a[m, 0]) / (b[m, 1] - a[m, 1])
        xs = np.sort(xs)
        for k in range(0, len(xs) - 1, 2):
            iv.append([xs[k], xs[k + 1]])
    iv.sort()
    out = []
    for a, b in iv:
        if out and a <= out[-1][1] + 0.5:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def oar_frames(n=28):
    """(t, tip_dx, tip_dy) keyframes for one stroke, relative to the oar port."""
    drive = 0.45
    a0, a1 = 30.0, -25.0
    out = []
    for i in range(n + 1):
        p = i / n
        if p <= drive:
            s = p / drive
            s = 0.5 - 0.5 * math.cos(math.pi * s)
            th = math.radians(a0 + (a1 - a0) * s)
            lift = 0.0
        else:
            s = (p - drive) / (1 - drive)
            e = 0.5 - 0.5 * math.cos(math.pi * s)
            th = math.radians(a1 + (a0 - a1) * e)
            lift = 8.0 * math.sin(math.pi * s) ** 0.6
        dx = OAR_R * math.sin(th)
        depth = OAR_R * math.cos(th)
        dy = 5.0 + K_PERSP * depth - lift
        out.append((p, dx, dy))
    return out


def ship_svg(th, uid):
    S = SHIP
    p = []
    ink, acc, bg = th['ink'], th['accent'], th['bg']
    hull_d = encode(rdp(S['hull'], 0.04)) + 'z'
    posts_d = encode(rdp(S['bow_post'], 0.04)) + 'z' + encode(rdp(S['stern_post'], 0.04)) + 'z'
    mast_d = (f'M{f(MAST_X - 1.4)} {f(S["sheer_at"](MAST_X))}L{f(MAST_X - 0.9)} {f(MAST_TOP)}'
              f'L{f(MAST_X + 0.9)} {f(MAST_TOP)}L{f(MAST_X + 1.4)} {f(S["sheer_at"](MAST_X))}Z')
    # rigging behind the sail (hairlines)
    bow_top = S['bow'][NRISE + 12]
    stern_top = S['stern'][NRISE + 12]
    p.append(f'<path fill="none" stroke="{ink}" stroke-width="{th["rig_w"]}" opacity="{th["rig_op"]}" '
             f'stroke-linecap="round" d="M{f(MAST_X)} {f(MAST_TOP + 2)}L{f(bow_top[0] - 1)} {f(bow_top[1] + 4)}'
             f'M{f(MAST_X)} {f(MAST_TOP + 2)}L{f(stern_top[0] + 2)} {f(stern_top[1] + 5)}'
             f'M{f(MAST_X)} {f(MAST_TOP + 6)}L{f(MAST_X - 30)} {f(S["sheer_at"](MAST_X - 30))}'
             f'M{f(MAST_X)} {f(MAST_TOP + 6)}L{f(MAST_X - 44)} {f(S["sheer_at"](MAST_X - 44))}'
             f'M{f(MAST_X)} {f(MAST_TOP + 6)}L{f(MAST_X + 34)} {f(S["sheer_at"](MAST_X + 34))}'
             f'M{f(BOT_L[0])} {f(BOT_L[1])}L{f(SX0 - 22)} {f(S["sheer_at"](SX0 - 22))}'
             f'M{f(BOT_R[0])} {f(BOT_R[1])}L{f(SX1 + 20)} {f(S["sheer_at"](SX1 + 20))}"/>')
    # occluder: the ship hides what is behind it (paper / night colour)
    p.append(f'<defs><path id="hull{uid}" d="{hull_d}"/></defs><use href="#hull{uid}" fill="{bg}"/>')
    p.append(f'<path d="{posts_d}{mast_d}{SAIL_D}" fill="{bg}"/>')
    # mast
    p.append(f'<path d="{mast_d}" fill="{ink}"/>')
    # sail: lines along the cloths (u = const), alternate cloths in accent ink
    nst = 7
    pa, pi = Pen(eps=0.08), Pen(eps=0.08)
    nl = 58
    v = np.linspace(0, 1, 40)
    for k in range(nl):
        u = (k + 0.5) / nl
        cl = sail_patch(np.full_like(v, u), v)
        stripe = int(u * nst)
        # shade: the belly of the sail turns from the light low and to one side
        shade = 0.55 + 0.45 * smoothstep(0.2, 1.0, v) * (0.6 + 0.4 * (1 - u))
        if stripe % 2 == 1:
            pa.line(cl, th['sail_acc_w'] * shade)
        elif k % 2 == 0:
            pi.line(cl, th['sail_ink_w'] * (shade - 0.35) / 0.65)
    p.append(f'<g fill="none" stroke="{acc}" stroke-linecap="round" opacity="{th["sail_acc_op"]}">{pa.svg()}</g>')
    p.append(f'<g fill="none" stroke="{ink}" stroke-linecap="round" opacity="{th["sail_ink_op"]}">{pi.svg()}</g>')
    # reef seams and outline
    seams = []
    for vv in (0.17, 0.42, 0.67):
        uu = np.linspace(0, 1, 30)
        seams.append(poly(sail_patch(uu, np.full_like(uu, vv)), False))
    p.append(f'<path d="{"".join(seams)}" fill="none" stroke="{ink}" stroke-width=".35" opacity=".5"/>')
    p.append(f'<path d="{SAIL_D}" fill="none" stroke="{ink}" stroke-width="{th["sail_edge_w"]}"/>')
    # yard
    p.append(f'<path d="M{f(SX0 - 6)} {f(YARD_Y + .6)}Q{f(MAST_X)} {f(YARD_Y - 1.5)} {f(SX1 + 6)} {f(YARD_Y + .6)}" '
             f'stroke="{ink}" stroke-width="2.2" stroke-linecap="round" fill="none"/>')
    # hull: strake hatching following the sheer, heavier towards the bilge
    xs, sy = S['xs'], S['sheer'][:, 1]
    ph = Pen(eps=0.06)
    nstr = th['hull_lines']
    for k in range(1, nstr + 1):
        fr = k / (nstr + 1)
        y = sy * (1 - fr) + 1.5 * fr
        m = y < 1.0
        pts = np.stack([xs[m], y[m]], 1)
        strake = (k % 3 == 0)
        w = th['hull_w'] * (0.55 + 0.45 * fr) * (1.35 if strake else 1.0)
        # fade towards the sunlit ends of the hull
        ww = w * (1 - 0.35 * smoothstep(40, 80, np.abs(pts[:, 0])))
        ph.line(pts, ww)
    p.append(f'<clipPath id="hullclip{uid}"><use href="#hull{uid}"/></clipPath>')
    p.append(f'<g clip-path="url(#hullclip{uid})" fill="none" stroke="{ink}" stroke-linecap="round" '
             f'opacity="{th["hull_op"]}">{ph.svg()}</g>')
    # stems (tapered posts) and hull outline
    p.append(f'<path d="{posts_d}" fill="{th["post_fill"]}" stroke="{ink}" stroke-width="{th["post_w"]}"/>')
    p.append(f'<use href="#hull{uid}" fill="none" stroke="{ink}" stroke-width="{th["hull_edge"]}" stroke-linejoin="round"/>')
    p.append(f'<path d="{poly(S["sheer"][3:-3], False)}" fill="none" stroke="{ink}" stroke-width="{th["gunwale_w"]}"/>')
    # shields along the rail
    sh_a, sh_b = [], []
    for i, x in enumerate(SHIELD_X):
        y = S['sheer_at'](x) + 0.6
        (sh_a if i % 2 == 0 else sh_b).append(f'<circle cx="{f(x)}" cy="{f(y)}" r="3.3"/>')
    p.append(f'<g fill="{bg}" stroke="{ink}" stroke-width=".55">{"".join(sh_a)}</g>')
    p.append(f'<g fill="{th["shield_b"]}" stroke="{ink}" stroke-width=".55">{"".join(sh_b)}</g>')
    p.append(f'<g fill="{ink}">' + ''.join(
        f'<circle cx="{f(x)}" cy="{f(S["sheer_at"](x) + .6)}" r=".85"/>' for i, x in enumerate(SHIELD_X) if i % 2 == 0) + '</g>')
    # oars
    p.append(f'<g>{oar_uses(S, uid)}</g>')
    return ''.join(p)


def oar_uses(S, uid):
    return ''.join(f'<use href="#oar{uid}" x="{f(x)}" y="{f(S["sheer_at"](x) + 5.4)}"/>' for x in OAR_X)


def oar_defs(th, uid):
    fr = oar_frames()
    kt = ';'.join(f'{p:.4f}' for p, _, _ in fr)
    shaft = ';'.join(f'M0 0L{dx:.2f} {dy:.2f}' for _, dx, dy in fr)
    blade = ';'.join(f'M{dx * .72:.2f} {dy * .72:.2f}L{dx:.2f} {dy:.2f}' for _, dx, dy in fr)
    return (f'<g id="oar{uid}" stroke="{th["ink"]}" stroke-linecap="round" fill="none">'
            f'<path stroke-width=".8" d="M0 0L{fr[0][1]:.2f} {fr[0][2]:.2f}">'
            f'<animate attributeName="d" dur="{STROKE}s" repeatCount="indefinite" keyTimes="{kt}" values="{shaft}"/></path>'
            f'<path stroke-width="1.8" d="M0 0L1 1">'
            f'<animate attributeName="d" dur="{STROKE}s" repeatCount="indefinite" keyTimes="{kt}" values="{blade}"/></path>'
            f'</g>')


def mover_anim():
    f0 = (X_START_VIS - X_A) / (X_B - X_A)
    k = 1 - f0
    return (f'<animateTransform attributeName="transform" type="translate" dur="{LOOP}s" '
            f'repeatCount="indefinite" calcMode="linear" '
            f'values="{f(X_START_VIS)} {f(YW)};{f(X_B)} {f(YW)};{f(X_A)} {f(YW)};{f(X_START_VIS)} {f(YW)}" '
            f'keyTimes="0;{k:.5f};{k:.5f};1"/>')


def surge_anim():
    vals = []
    n = 12
    for i in range(n + 1):
        ph = i / n
        vals.append(f'{1.1 * math.sin(2 * math.pi * (ph - 0.12)):.2f} 0')
    return (f'<animateTransform attributeName="transform" type="translate" dur="{STROKE}s" '
            f'repeatCount="indefinite" values="{";".join(vals)}"/>')


def wake_svg(th, uid):
    out = []
    c = th['ink']
    slope = K_PERSP * math.tan(math.radians(19.5))
    B = (73.0, 1.4)
    for side in (+1, -1):
        L = WAKE_LEN if side > 0 else WAKE_LEN * 0.95
        ex, ey = B[0] - L, B[1] + side * slope * L
        out.append(f'<path d="M{f(B[0])} {f(B[1])}L{f(ex)} {f(ey)}" '
                   f'stroke="url(#{"wakenear" if side > 0 else "wakefade"}{uid})" '
                   f'stroke-width="{0.85 if side > 0 else 0.65}" fill="none"/>')
        nfe = 13
        for i in range(1, nfe + 1):
            s = L * (i / (nfe + 1)) ** 1.15
            if side < 0 and B[0] - s > -84:
                continue
            if side > 0 and B[0] - s > -40:
                continue
            px, py = B[0] - s, B[1] + side * slope * s
            ln = 7 + 10 * (s / L)
            dx, dy = -0.92 * ln, side * 0.10 * ln * 0.9
            op = th['wake_op'] * (1 - s / L) ** 1.3
            out.append(f'<path d="M{f(px)} {f(py)}l{f(dx)} {f(dy)}" stroke="{c}" '
                       f'stroke-width=".75" opacity="{op:.2f}" fill="none"/>')
    out.append(f'<path d="M72 2.4q-6 2.2-15 2.4M70 3.8q-8 2-19 1.6" stroke="{c}" stroke-width=".75" '
               f'fill="none" opacity="{th["wake_op"] * .8:.2f}"/>')
    per = 26.0
    dur = per / V
    for j, (dy, op, w) in enumerate(((2.6, .55, .75), (5.2, .35, .65), (-0.6, .3, .55))):
        out.append(f'<path d="M-70 {f(dy)}L{f(-70 - 230)} {f(dy + j * 1.5)}" stroke="url(#trailfade{uid})" '
                   f'stroke-width="{w}" stroke-dasharray="{12 - 3 * j} {14 + 3 * j}" opacity="{op * th["wake_op"] / .7:.2f}" fill="none">'
                   f'<animate attributeName="stroke-dashoffset" values="{j * 7};{j * 7 - per}" dur="{dur:.3f}s" '
                   f'repeatCount="indefinite"/></path>')
    return ''.join(out)


def puddles_svg(th, uid):
    """Ring left by each blade at the catch; stays fixed in the world."""
    fr = oar_frames()
    cdx, cdy = fr[0][1], fr[0][2]
    life = 4.2
    dur = 2 * STROKE
    kt_end = min(life / dur, 0.98)
    back = V * life
    defs = []
    for k in range(2):
        defs.append(
            f'<ellipse id="pud{uid}{k}" cx="0" cy="0" rx="1" ry=".3" fill="none" stroke="{th["ink"]}" '
            f'stroke-width=".6" opacity="0">'
            f'<animate attributeName="rx" values="2;14;14" keyTimes="0;{kt_end:.3f};1" dur="{dur}s" '
            f'begin="{k * STROKE}s" repeatCount="indefinite"/>'
            f'<animate attributeName="ry" values=".6;3.6;3.6" keyTimes="0;{kt_end:.3f};1" dur="{dur}s" '
            f'begin="{k * STROKE}s" repeatCount="indefinite"/>'
            f'<animate attributeName="opacity" values="0;{th["pud_op"]};0;0" keyTimes="0;.05;{kt_end:.3f};1" dur="{dur}s" '
            f'begin="{k * STROKE}s" repeatCount="indefinite"/>'
            f'<animateTransform attributeName="transform" type="translate" values="0 0;{-back:.1f} 0;{-back:.1f} 0" '
            f'keyTimes="0;{kt_end:.3f};1" dur="{dur}s" begin="{k * STROKE}s" repeatCount="indefinite"/>'
            f'</ellipse>')
    uses = []
    for x in OAR_X:
        for k in range(2):
            uses.append(f'<use href="#pud{uid}{k}" x="{f(x + cdx)}" y="{f(cdy + SHIP["sheer_at"](x) + 5.8)}"/>')
    return ''.join(defs), ''.join(uses)


# ------------------------------------------------------------------ sky
def sky_cov(th, x, y, jit=1.0, refl=False):
    """Line coverage of the sky (fraction of the row pitch that is ink)."""
    t = np.clip(y / YH, 0, 1)
    c = th['sky_top'] + (th['sky_hor'] - th['sky_top']) * t ** th['sky_gam']
    # elliptical distance from the light
    r = np.hypot((x - LIGHT_X) / 1.25, (y - LIGHT_Y) / 0.62)
    if th['night']:
        c = c + th['glow'] * np.exp(-(r / 120) ** 2) + th['glow2'] * np.exp(-(r / 42) ** 2)
        if not refl:
            c = c * smoothstep(LIGHT_R + 1.2, LIGHT_R + 4.5, np.hypot(x - LIGHT_X, y - LIGHT_Y))
    else:
        c = c * (0.2 + 0.8 * smoothstep(20 * jit, 170 * jit, r))
    return c


def sky_svg(th):
    pen = Pen(eps=0.05, wmin=0.08)
    r = np.random.default_rng(7)
    xs = np.arange(-4, W + 4.01, 2.0)
    sky_lim = np.interp(xs, GX, SKYLINE)
    y = th['sky_pitch'] * 0.8
    while y < YH:
        c = sky_cov(th, xs, y, 1 + 0.12 * r.uniform(-1, 1))
        # a little irregularity, like a hand-cut ruling
        c = c * (1 + 0.12 * fbm(xs + 4, 160, int(r.integers(1000)), 3))
        w = c * th['sky_pitch']
        w = np.where(y < sky_lim - 0.8, w, 0)
        pen.line(np.stack([xs, np.full_like(xs, y)], 1), w)
        y += th['sky_pitch']
    return pen


def moon_svg(th):
    out = []
    cx, cy, R = LIGHT_X, LIGHT_Y, LIGHT_R
    ink = th['ink']
    if th['night']:
        # engraved disc: close horizontal ruling
        pen = Pen(eps=0.05, wmin=0.15)
        pitch = 1.6
        yy = cy - R + pitch / 2
        while yy < cy + R:
            hw = math.sqrt(max(R * R - (yy - cy) ** 2, 0))
            xs = np.linspace(cx - hw, cx + hw, max(3, int(hw)))
            w = np.full_like(xs, 1.05)
            pen.line(np.stack([xs, np.full_like(xs, yy)], 1), w)
            yy += pitch
        out.append(f'<g stroke="{th["moon"]}" stroke-linecap="butt" fill="none">{pen.svg()}</g>')
        out.append(f'<circle cx="{f(cx)}" cy="{f(cy)}" r="{f(R)}" fill="none" stroke="{th["moon"]}" stroke-width=".7"/>')
    else:
        out.append(f'<circle cx="{f(cx)}" cy="{f(cy)}" r="{f(R)}" fill="none" stroke="{ink}" stroke-width=".7" opacity=".6"/>')
    return ''.join(out)


def stars_svg(th):
    r = np.random.default_rng(91)
    out = [f'<g fill="{th["ink"]}">']
    tw = []
    for i in range(46):
        x, y = r.uniform(20, W - 20), r.uniform(8, 150)
        if math.hypot(x - LIGHT_X, (y - LIGHT_Y) * 1.6) < 150:
            continue
        if y > np.interp(x, GX, SKYLINE) - 6:
            continue
        rad = r.uniform(0.5, 1.0)
        op = r.uniform(0.25, 0.7) * (1 - y / 220)
        if i % 9 == 0:
            dur = r.uniform(5, 9)
            tw.append(f'<circle cx="{f(x)}" cy="{f(y)}" r="{rad:.2f}" opacity="{op:.2f}">'
                      f'<animate attributeName="opacity" values="{op:.2f};{op * .25:.2f};{op:.2f}" dur="{dur:.1f}s" '
                      f'repeatCount="indefinite"/></circle>')
        else:
            out.append(f'<circle cx="{f(x)}" cy="{f(y)}" r="{rad:.2f}" opacity="{op:.2f}"/>')
    return ''.join(out + tw) + '</g>'


# ------------------------------------------------------------------ mountains
GY = np.arange(-2, YH + 6.01, 1.0)
MX, MY = np.meshgrid(GX, GY)


def wave_noise(X, Y, seed, n=6, lam=(30, 90), amp=1.0):
    rng = np.random.default_rng(seed)
    out = np.zeros_like(X)
    for _ in range(n):
        l = rng.uniform(*lam)
        a = rng.uniform(0, 2 * math.pi)
        k = 2 * math.pi / l
        out += np.sin(k * math.cos(a) * X + k * math.sin(a) * Y * 1.6 + rng.uniform(0, 2 * math.pi))
    return amp * out / math.sqrt(n)


def layer_field(i):
    """Hatch field: smoothed distance below the skyline, lightly perturbed."""
    prof = gaussian_filter1d(PROFILES[i], 1.6)
    xd = np.arange(GX[0] - 40, GX[-1] + 40, 0.25)
    yd = np.interp(xd, GX, prof)
    tree = cKDTree(np.stack([xd, yd], 1))
    d, _ = tree.query(np.stack([MX.ravel(), MY.ravel()], 1))
    d = d.reshape(MX.shape)
    inside = MY >= np.interp(MX, GX, prof)
    phi = np.where(inside, d, -d)
    phi = gaussian_filter(phi, 1.2)
    # deeper inside, the lines relax towards the ridge-parallel family and
    # pick up a slow waviness, like rock buttresses
    vert = MY - np.interp(MX, GX, gaussian_filter1d(PROFILES[i], 14))
    b = smoothstep(8, 70, phi) * 0.45
    phi = (1 - b) * phi + b * vert
    phi += smoothstep(4, 24, phi) * wave_noise(MX, MY, 300 + i, lam=(40, 120), amp=1.6)
    return phi


def mountains(th):
    """Returns per-layer (hatch pen, rim pen)."""
    out = []
    sun = np.array([LIGHT_X, LIGHT_Y])
    for i, L in enumerate(LAYERS):
        phi = layer_field(i)
        gy, gx = np.gradient(phi)
        top = float(PROFILES[i][(GX >= 0) & (GX <= W)].min())
        pitch = th['mtn_pitch'][i]
        pen = Pen(eps=0.1, wmin=th['mtn_wmin'])
        lev = pitch * 0.9
        while lev < phi.max():
            for c in measure.find_contours(phi, lev):
                py = GY[0] + c[:, 0]
                px = GX[0] + c[:, 1]
                if len(c) < 4:
                    continue
                gxs = map_coordinates(gx, [c[:, 0], c[:, 1]], order=1)
                gys = map_coordinates(gy, [c[:, 0], c[:, 1]], order=1)
                gn = np.hypot(gxs, gys) + 1e-9
                ox, oy = -gxs / gn, -gys / gn            # outward (towards the sky)
                tx, ty = sun[0] - px, sun[1] - py
                tn = np.hypot(tx, ty) + 1e-9
                lit = np.clip((ox * tx + oy * ty) / tn, 0, 1)
                hz = smoothstep(top + th['haze_start'] * (YH - top), YH + 2, py)
                cov = th['mtn_cov'][i] * (1 + th['lit_k'] * lit) * (1 - th['haze_k'][i] * hz)
                # the line weight swells and thins slowly, as from a burin
                cov *= 1 + 0.18 * np.sin(px / 23.0 + lev * 0.7 + i)
                w = cov * pitch
                vis = (py <= front_at(i, px) - 0.7) & (py >= prof_at(i, px) + 0.5) & (py <= YH - 0.4)
                w = np.where(vis, w, 0)
                pen.line(np.stack([px, py], 1), w)
            lev += pitch
        # skyline outline (the actual ridge, with its fine roughness)
        rim = Pen(eps=0.08)
        xs = GX
        y = PROFILES[i]
        vis = (y <= FRONT[i] - 0.3) & (y < YH - 1.0)
        hz = smoothstep(top + 0.5 * (YH - top), YH, y)
        w = th['rim_w'][i] * (1 - 0.85 * hz)
        rim.line(np.stack([xs, y], 1), np.where(vis, w, 0))
        out.append((pen, rim))
    return out


WATERFALLS = [(3, 896.0)]


def waterfall_svg(th, uid, x0):
    i = 3
    ytop = float(prof_at(i, x0)) + 1.0
    ybot = ytop + 0.62 * (YH - ytop)
    n = 30
    ys = np.linspace(ytop, ybot, n)
    xx = x0 + 1.4 * (1 - np.exp(-(ys - ytop) / 5)) + 0.35 * np.sin(ys / 7)
    d = poly(list(zip(xx, ys)), False)
    gid = f'fall{uid}'
    return (f'<defs><linearGradient id="{gid}" x1="0" y1="{f(ytop)}" x2="0" y2="{f(ybot)}" gradientUnits="userSpaceOnUse">'
            f'<stop offset="0" stop-color="{th["fall"]}" stop-opacity="1"/>'
            f'<stop offset=".55" stop-color="{th["fall"]}" stop-opacity=".6"/>'
            f'<stop offset="1" stop-color="{th["fall"]}" stop-opacity="0"/></linearGradient></defs>'
            f'<path d="{d}" fill="none" stroke="{th["bg"]}" stroke-width="{th["fall_gap"]}" stroke-linecap="round"/>'
            f'<g fill="none" stroke="url(#{gid})" stroke-linecap="round">'
            f'<path d="{d}" stroke-width=".55" opacity=".7"/>'
            f'<path d="{d}" stroke-width=".7" stroke-dasharray="5 4 2 5">'
            f'<animate attributeName="stroke-dashoffset" values="0;-16" dur="1.1s" repeatCount="indefinite"/></path>'
            f'</g>')


MIST = {
    0: dict(y=204, h=30, per=520, amp=7, dur=260, op=.75, seed=2),
    1: dict(y=208, h=26, per=640, amp=8, dur=330, op=.65, seed=4),
    2: dict(y=212, h=24, per=760, amp=6, dur=300, op=.6, seed=6),
    3: dict(y=214, h=22, per=600, amp=6, dur=240, op=.55, seed=9),
    4: dict(y=217, h=20, per=700, amp=5, dur=280, op=.5, seed=10),
    'fg': dict(y=222, h=14, per=900, amp=3, dur=420, op=.35, seed=12),
}


def mist_band(th, uid, key):
    """A veil in the background colour (the hatching thins beneath it) that
    carries a few fine wisps, drifting left by one period per cycle."""
    m = MIST[key]
    per = m['per']
    xs = np.arange(0, W + per + 1, 24.0)
    r = np.random.default_rng(m['seed'])
    ph = r.uniform(0, 2 * np.pi, 3)
    top = m['y'] - m['amp'] * (0.55 * np.sin(2 * np.pi * xs / per + ph[0])
                               + 0.3 * np.sin(4 * np.pi * xs / per + ph[1])
                               + 0.15 * np.sin(6 * np.pi * xs / per + ph[2]))
    bot = m['y'] + m['h'] + 0.4 * m['amp'] * np.sin(2 * np.pi * xs / per + ph[1])
    pts = list(zip(xs, top)) + list(zip(xs[::-1], bot[::-1]))
    gid = f'mist{uid}{key}'
    g = (f'<linearGradient id="{gid}" x1="0" y1="0" x2="0" y2="1">'
         f'<stop offset="0" stop-color="{th["bg"]}" stop-opacity="0"/>'
         f'<stop offset=".5" stop-color="{th["bg"]}" stop-opacity="{m["op"] * th["veil_op"]:.3f}"/>'
         f'<stop offset="1" stop-color="{th["bg"]}" stop-opacity="0"/></linearGradient>')
    # wisps: tapered horizontal strokes, periodic in x with period `per`
    pen = Pen(eps=0.08, wmin=0.12)
    nw = int(per / 110)
    for k in range(nw):
        cx = r.uniform(0, per)
        ln = r.uniform(60, 170)
        yy = m['y'] + m['h'] * r.uniform(0.3, 0.6)
        for rep in range(int((W + per) / per) + 2):
            x0 = cx + rep * per - ln / 2
            if x0 > W + per + 10 or x0 + ln < -10:
                continue
            t = np.linspace(0, 1, 10)
            xx = x0 + ln * t
            yv = yy + 0.8 * np.sin(2 * np.pi * (xx / per) * 3 + ph[2]) * 0.6
            pen.line(np.stack([xx, yv], 1), th['wisp_w'] * np.sin(np.pi * t) ** 0.8)
    return (f'<defs>{g}</defs><g>'
            f'<animateTransform attributeName="transform" type="translate" values="0 0;{-per} 0" '
            f'dur="{m["dur"]}s" repeatCount="indefinite"/>'
            f'<path d="{encode(np.array(pts))}z" fill="url(#{gid})"/>'
            f'<g fill="none" stroke="{th["ink"]}" stroke-linecap="round" opacity="{th["wisp_op"]}">{pen.svg()}</g>'
            f'</g>')


# ------------------------------------------------------------------ water
def water_rows():
    rows = []
    y, gap = YH + 1.6, 1.9
    while y < H + 3:
        rows.append((y, gap))
        y += gap
        gap = min(gap * 1.05, 4.4)
    return rows


ROWS = water_rows()


def glade(x, y):
    depth = (y - YH) / (H - YH)
    halfw = 16 + 120 * depth ** 0.9
    return np.exp(-((x - LIGHT_X) / (halfw * 0.75)) ** 2) * (1 - 0.4 * depth)


def water_cov(th, x, y, mtn_cov):
    """Coverage of a ripple row at depth y: base water tone, mirrored
    mountains / sky, the glade under the light."""
    depth = (y - YH) / (H - YH)
    yr = 2 * YH - y
    li = layer_at(x, np.full_like(x, yr))
    sky = sky_cov(th, x, yr, refl=True)
    refl = np.where(li >= 0, np.take(np.r_[mtn_cov, 0], li), sky * th['refl_sky'])
    # mountains fade into haze at their foot (mirrored: near the horizon)
    near = smoothstep(0, 22, y - YH)
    if th['night']:
        refl = np.where(li >= 0, refl * (0.35 + 0.65 * near) + th['shore'] * (1 - near), refl)
    else:
        refl = np.where(li >= 0, refl * (0.45 + 0.55 * near), refl)
    # ripples smear the mirror image sideways, more so nearer the viewer
    # (x is a uniform 1 px grid here)
    refl = gaussian_filter1d(refl, 1.5 + 9 * depth, mode='nearest')
    base = th['water_base'][0] + (th['water_base'][1] - th['water_base'][0]) * depth
    c = base + th['refl_k'] * refl
    # glade: a column of light under the moon / sun, widening towards us
    g = glade(x, y)
    if th['night']:
        c = c + th['glade_k'] * g
    else:
        c = c * (1 - th['glade_k'] * g)
    return c


def water_svg(th, uid, mtn_cov):
    rng = np.random.default_rng(23)
    xs = np.arange(-12, W + 12.01, 1.0)
    out = []
    for k, (y, gap) in enumerate(ROWS):
        depth = (y - YH) / (H - YH)
        c = water_cov(th, xs, y, mtn_cov)
        c = c * (1 + 0.15 * fbm(xs + 12, 90, 500 + k, 3))
        w = c * gap
        # break the line into long strokes with small gaps
        x = -12.0
        mask = np.ones_like(xs, bool)
        while x < W + 12:
            gx = float(glade(np.array([x]), y)[0])
            ln = rng.uniform(200, 900) * (0.6 + 0.8 * depth) * (1 - 0.97 * gx) + 3
            x += ln
            gp = rng.uniform(1.2, 2.6) * (0.8 + 0.8 * depth)
            mask &= ~((xs > x) & (xs < x + gp))
            x += gp
        w = np.where(mask, w, 0)
        pen = Pen(eps=0.05, wmax=3.0)
        pen.line(np.stack([xs, np.full_like(xs, y)], 1), w)
        # each row orbits slowly (water particles move in small ellipses),
        # with the phase advancing row by row like a long swell
        ax = 0.5 + 1.6 * depth
        ay = 0.12 + 0.25 * depth
        per = 7.5 - 2.0 * depth + 0.6 * math.sin(k * 1.7)
        ph = (k * 0.137) % 1
        n = 12
        vals = ';'.join(f'{ax * math.sin(2 * math.pi * (j / n + ph)):.2f} '
                        f'{ay * math.cos(2 * math.pi * (j / n + ph)):.2f}' for j in range(n + 1))
        out.append(f'<g><animateTransform attributeName="transform" type="translate" values="{vals}" '
                   f'dur="{per:.2f}s" repeatCount="indefinite"/>{pen.svg()}</g>')
    return f'<g fill="none" stroke="{th["ink"]}" stroke-linecap="round">{"".join(out)}</g>'


def glints_svg(th):
    """Short strokes on the glade that twinkle: bright nicks."""
    r = np.random.default_rng(17)
    groups = [[] for _ in range(5)]
    for k, (y, gap) in enumerate(ROWS):
        depth = (y - YH) / (H - YH)
        halfw = 16 + 120 * depth ** 0.9
        n = int(1 + 4 * depth ** 0.5)
        for _ in range(n):
            x = LIGHT_X + r.normal(0, 0.42) * halfw
            ln = r.uniform(3, 9) * (0.6 + 1.4 * depth)
            cen = math.exp(-((x - LIGHT_X) / (halfw * 0.75)) ** 2)
            op = (0.35 + 0.65 * cen) * (1 - 0.5 * depth)
            groups[r.integers(0, 5)].append(
                f'<path d="M{f(x - ln / 2)} {f(y)}h{f(ln)}" stroke-opacity="{op:.2f}"/>')
    out = [f'<g stroke="{th["glint"]}" stroke-width="{th["glint_w"]}" stroke-linecap="round" opacity="{th["glint_op"]}">']
    for i, g in enumerate(groups):
        dur = 3.1 + 0.83 * i
        vals = ';'.join(f'{0.25 + 0.75 * (0.5 + 0.5 * math.sin(2 * math.pi * (j / 6 + i / 5))):.2f}' for j in range(7))
        out.append(f'<g>{"".join(g)}<animate attributeName="opacity" values="{vals}" dur="{dur:.2f}s" '
                   f'repeatCount="indefinite"/></g>')
    out.append('</g>')
    return ''.join(out)


def ship_refl_svg(th, uid):
    """The ship's reflection as heavier strokes on the ripple rows (ship
    frame), each row swaying on its own like the water."""
    S = SHIP
    hull_polys = [S['hull'], S['bow_post'], S['stern_post']]
    mast = np.array([[MAST_X - 1.3, 0], [MAST_X - 1, MAST_TOP], [MAST_X + 1, MAST_TOP], [MAST_X + 1.3, 0]])
    out = []
    # denser rows than the water for a readable reflection
    ys = []
    for (y, gap) in ROWS:
        if y > YW - 0.5:
            ys.append((y, gap))
    for k, (y, gap) in enumerate(ys):
        r = y - YW                      # depth below the ship's water line
        yy = -r                         # mirrored height on the ship
        fade = max(0.0, 1 - r / 110) ** 0.7
        segs = []
        for (a, b) in chords(hull_polys + [mast], yy):
            segs.append((a, b, 'hull'))
        for (a, b) in chords([SAIL_POLY], yy):
            # keep the sail's cloth stripes: accent on the odd cloths
            nst = 7
            us = np.linspace(0, 1, 200)
            vv = np.clip((yy - YARD_Y) / (SAIL_BOT - YARD_Y), 0, 1)
            xu = sail_patch(us, np.full_like(us, vv))[:, 0]
            edges = np.interp(np.arange(1, nst) / nst, us, xu)
            cuts = [a] + [e for e in edges if a < e < b] + [b]
            for c0, c1 in zip(cuts[:-1], cuts[1:]):
                um = np.interp(0.5 * (c0 + c1), xu, us)
                if int(um * nst) % 2 == 1:
                    segs.append((c0, c1, 'sail'))
                else:
                    segs.append((c0, c1, 'cloth'))
        if not segs:
            continue
        d_er, d_h, d_s = [], [], []
        for a, b, kind in segs:
            if b - a < 0.6:
                continue
            seg = f'M{f(a)} {f(r)}H{f(b)}'
            d_er.append(seg)
            if kind == 'hull':
                d_h.append(seg)
            elif kind == 'sail':
                d_s.append(seg)
        amp = 0.6 + 0.16 * k
        per = 3.4 + 0.9 * math.sin(k * 1.7) + 0.05 * k
        ph = (k * 0.37) % 1
        vals = ';'.join(f'{amp * math.sin(2 * math.pi * (i / 8 + ph)):.2f} 0' for i in range(9))
        g = [f'<g><animateTransform attributeName="transform" type="translate" values="{vals}" '
             f'dur="{per:.2f}s" repeatCount="indefinite"/>']
        if th['refl_erase'] > 0:
            g.append(f'<path d="{"".join(d_er)}" stroke="{th["bg"]}" stroke-width="{f(min(gap * .9, 2.6))}" '
                     f'stroke-linecap="round" opacity="{th["refl_erase"] * fade:.2f}"/>')
        if d_h:
            g.append(f'<path d="{"".join(d_h)}" stroke="{th["ink"]}" stroke-width="{f(gap * th["refl_hull_w"])}" opacity="{fade:.2f}"/>')
        if d_s:
            g.append(f'<path d="{"".join(d_s)}" stroke="{th["accent"]}" stroke-width="{f(gap * th["refl_sail_w"])}" opacity="{fade * th["refl_sail_op"]:.2f}"/>')
        g.append('</g>')
        out.append(''.join(g))
    # mirrored oars
    oars = (f'<g transform="scale(1 -1)" opacity="{th["refl_oar_op"]}">{oar_uses(S, uid)}</g>')
    return (f'<g>{mover_anim()}<g fill="none" stroke-linecap="butt">{"".join(out)}</g>'
            f'<g transform="translate(0 0)">{oars}</g></g>')


# ------------------------------------------------------------------ scene
def theme_svg(th, uid):
    parts, defs = [], []
    ink = th['ink']
    defs.append(f'<linearGradient id="wakefade{uid}" x1="77" y1="0" x2="{77 - WAKE_LEN}" y2="0" gradientUnits="userSpaceOnUse">'
                f'<stop offset="0" stop-color="{ink}" stop-opacity="{th["wake_op"]:.2f}"/>'
                f'<stop offset=".55" stop-color="{ink}" stop-opacity="0"/></linearGradient>')
    defs.append(f'<linearGradient id="wakenear{uid}" x1="73" y1="0" x2="{73 - WAKE_LEN}" y2="0" gradientUnits="userSpaceOnUse">'
                f'<stop offset=".3" stop-color="{ink}" stop-opacity="0"/>'
                f'<stop offset=".55" stop-color="{ink}" stop-opacity="{th["wake_op"] * .75:.2f}"/>'
                f'<stop offset="1" stop-color="{ink}" stop-opacity="0"/></linearGradient>')
    defs.append(f'<linearGradient id="trailfade{uid}" x1="-74" y1="0" x2="-304" y2="0" gradientUnits="userSpaceOnUse">'
                f'<stop offset="0" stop-color="{ink}" stop-opacity="1"/>'
                f'<stop offset="1" stop-color="{ink}" stop-opacity="0"/></linearGradient>')

    # sky
    sp = sky_svg(th)
    parts.append(f'<g fill="none" stroke="{ink}" stroke-linecap="round" opacity="{th["sky_op"]}">{sp.svg()}</g>')
    parts.append(moon_svg(th))
    if th['night']:
        parts.append(stars_svg(th))

    # mountains
    mt = mountains(th)
    for i, (pen, rim) in enumerate(mt):
        parts.append(f'<g fill="none" stroke="{ink}" stroke-linecap="round" stroke-linejoin="round" '
                     f'opacity="{th["mtn_op"][i]}">{pen.svg()}</g>')
        parts.append(f'<g fill="none" stroke="{th["rim"]}" stroke-linecap="round" stroke-linejoin="round" '
                     f'opacity="{th["rim_op"][i]}">{rim.svg()}</g>')
        for wl, wx in WATERFALLS:
            if wl == i:
                parts.append(waterfall_svg(th, uid, wx))
        if i in MIST:
            parts.append(mist_band(th, uid, i))

    # water
    mtn_c = np.array(th['mtn_refl'])
    parts.append(f'<g opacity="{th["water_op"]}">{water_svg(th, uid, mtn_c)}</g>')
    parts.append(mist_band(th, uid, 'fg'))
    parts.append(glints_svg(th))

    # ship reflection, wake, puddles, ship
    pud_defs, pud_uses = puddles_svg(th, uid)
    defs.append(oar_defs(th, uid))
    defs.append(pud_defs)
    parts.append(ship_refl_svg(th, uid))
    parts.append(f'<g>{mover_anim()}'
                 f'<g>{wake_svg(th, uid)}</g>'
                 f'<g>{pud_uses}</g>'
                 f'<g>{surge_anim()}{ship_svg(th, uid)}</g>'
                 f'</g>')
    return ''.join(defs), ''.join(parts)


THEMES = {
    'dark': dict(
        night=True, mtn_wmin=.1,
        bg='#0f1726', border='#30363d',
        ink='#c6d0dc', rim='#dfe6ee', accent='#c98a6c', moon='#efe8d6',
        glint='#f1ece0', glint_w=.9, glint_op=.9,
        # sky: pale ruling, faint at the zenith, swelling into the moon's glow
        sky_pitch=2.6, sky_top=.05, sky_hor=.13, sky_gam=1.4, sky_op=.9,
        glow=.20, glow2=.22,
        mtn_pitch=[2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2],
        mtn_cov=[.16, .14, .12, .1, .088, .076, .068],
        mtn_op=[.8, .78, .76, .74, .72, .7, .68],
        lit_k=1.6, haze_start=.5, haze_k=[.6, .65, .7, .75, .8, .82, .85],
        rim_w=[.6, .65, .7, .75, .8, .85, .9], rim_op=[.55, .5, .48, .45, .42, .4, .38],
        fall='#dbe3ec', fall_gap=2.4,
        veil_op=.85, wisp_w=.45, wisp_op=.35,
        water_base=(.05, .02), refl_k=1.0, refl_sky=1.0, shore=.12,
        mtn_refl=[.06, .05, .045, .04, .035, .03, .03], glade_k=.5, water_op=.85,
        rig_w=.45, rig_op=.6, sail_acc_w=.75, sail_acc_op=.95, sail_ink_w=.3, sail_ink_op=.45,
        sail_edge_w=.6, hull_lines=6, hull_w=.35, hull_op=.75, hull_edge=.7, gunwale_w=.9,
        post_fill='#0f1726', post_w=.6, shield_b='#0f1726',
        wake_op=.6, pud_op=.4,
        refl_erase=1.0, refl_hull_w=.0, refl_sail_w=.32, refl_sail_op=.55, refl_oar_op=.25,
    ),
    'light': dict(
        night=False, mtn_wmin=.16,
        bg='#f5f0e6', border='#d8d0c2',
        ink='#26303c', rim='#26303c', accent='#9c4a33', moon='#26303c',
        glint='#f5f0e6', glint_w=1.2, glint_op=1,
        sky_pitch=2.6, sky_top=.15, sky_hor=.045, sky_gam=.8, sky_op=.75,
        glow=0, glow2=0,
        mtn_pitch=[2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2],
        mtn_cov=[.1, .13, .17, .21, .26, .31, .36],
        mtn_op=[.75, .8, .85, .9, .92, .95, 1],
        lit_k=-.45, haze_start=.45, haze_k=[.7, .72, .75, .78, .8, .82, .84],
        rim_w=[.5, .55, .6, .65, .7, .75, .8], rim_op=[.5, .55, .6, .65, .7, .75, .8],
        fall='#26303c', fall_gap=3.4,
        veil_op=.9, wisp_w=.4, wisp_op=.3,
        water_base=(.02, .05), refl_k=1.0, refl_sky=.35, shore=0,
        mtn_refl=[.09, .11, .14, .16, .19, .21, .24], glade_k=.9, water_op=.9,
        rig_w=.45, rig_op=.75, sail_acc_w=1.0, sail_acc_op=.9, sail_ink_w=.35, sail_ink_op=.55,
        sail_edge_w=.7, hull_lines=9, hull_w=.85, hull_op=1, hull_edge=.8, gunwale_w=1.0,
        post_fill='#26303c', post_w=.3, shield_b='#26303c',
        wake_op=.7, pud_op=.45,
        refl_erase=.0, refl_hull_w=.5, refl_sail_w=.4, refl_sail_op=1, refl_oar_op=.35,
    ),
}


def build(name):
    th = THEMES[name]
    uid = name[0]
    defs, body = theme_svg(th, uid)
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">'
           f'<defs><clipPath id="card{uid}"><rect width="{W}" height="{H}" rx="16"/></clipPath>{defs}</defs>'
           f'<g clip-path="url(#card{uid})"><rect width="{W}" height="{H}" fill="{th["bg"]}"/>{body}</g>'
           f'<rect x=".5" y=".5" width="{W - 1}" height="{H - 1}" rx="15.5" fill="none" stroke="{th["border"]}"/>'
           f'</svg>')
    path = os.path.join(HERE, f'banner-{name}.svg')
    with open(path, 'w') as fh:
        fh.write(svg)
    print(path, len(svg.encode()) // 1024, 'KB')


if __name__ == '__main__':
    for n in ('dark', 'light'):
        build(n)
