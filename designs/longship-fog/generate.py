#!/usr/bin/env python3
"""Out of the fog: a Viking longship emerging from fjord fog (variant of ../longship).

Scene (back to front)
---------------------
pale sky, a sun / moon diffused to a soft disc behind high wisps, seven
procedurally generated fjord walls (same glacial U-valley generator as the
original) whose lower faces dissolve into fog, with a rolling fog bank drifting
in front of each wall so only the cliff tops stand clear; the water (faint
mirrored walls, diffuse light column, slow ripples, drizzle rings), the
longship's reflection, the wake, the ship, low wisps sliding past the hull,
two large fog masses that leave a clearer patch mid-fjord, and a fine drizzle.

Fog is built only from precomputed soft shapes: elliptical puffs filled with
a radial gradient (no filters at all). Each bank is two independent puff
layers drifting with the wind at slightly different speeds, so the bank's
outline slowly changes instead of sliding rigidly; banks further away drift
slower (parallax) and gently rise and sink. The fog masses breathe in place.

The ship (unchanged from the original: Gokstad/Oseberg proportions, rowing
oars, puddle rings, Kelvin wake) enters in thick fog as a grey ghost, becomes
crisp and shows the sail's colour in the clear patch, then dissolves again.

Motion: SMIL only (transforms, opacity, stroke-dashoffset, path d), every
animation periodic, so the loop is seamless.

Usage: python generate.py  -> banner-dark.svg, banner-light.svg
"""
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

W, H = 1200, 340
YH = 226.0          # horizon (water line of the distant shore)
YW = 276.0          # ship's water line
LIGHT_X, LIGHT_Y, LIGHT_R = 668.0, 104.0, 15.0

STROKE = 3.2        # seconds per oar stroke
X_START_VIS = 470.0 # ship centre at t = 0 (just emerging into the clear patch)
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
    return s


def poly(pts, close=True):
    out = 'M' + ' '.join(f'{f(x)} {f(y)}' for x, y in pts[:1])
    out += 'L' + ' '.join(f'{f(x)} {f(y)}' for x, y in pts[1:])
    return out + ('Z' if close else '')


def mix(c1, c2, t):
    a = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
    return '#' + ''.join(f'{round(x + (y - x) * t):02x}' for x, y in zip(a, b))


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
    v = 0.5 * (1 - np.tanh(2.2 * s))
    return v


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
    if L.get('kind') == 'range':
        n = fbm(x, 120, L['seed'], 7, 0.55)
        hump = np.exp(-((x - 700) / 230) ** 2)
        y = YH - (YH - L['top']) * (0.45 + 0.55 * hump) * (0.7 + 0.3 * n)
        return y
    sd = L['seed']
    # irregular cliff line (domain-warped edge)
    xw = x + 1.4 * L['w'] * fbm(x, 110, sd + 7, 4)
    env = cliff(xw, L['c'], L['w'], L['side'])
    # fjell plateau with a few tinder
    plate = L['top'] + L['rough'] * fbm(x, 190, sd, 7, 0.55)
    r = np.random.default_rng(sd)
    inland = np.clip((L['c'] - x) if L['side'] == 'L' else (x - L['c']), 0, 500)
    plate -= L.get('trend', 0.05) * inland - 0.05 * 60
    for _ in range(4):
        px = L['c'] + (-1 if L['side'] == 'L' else 1) * r.uniform(1.5, 11) * L['w']
        plate -= r.uniform(4, 14) * np.exp(-((x - px) / r.uniform(14, 38)) ** 2)
    # a hanging-valley notch
    nx = L['c'] + (-1 if L['side'] == 'L' else 1) * r.uniform(3, 7) * L['w']
    plate += r.uniform(4, 9) * np.exp(-((x - nx) / r.uniform(10, 18)) ** 2)
    y = YH - (YH - plate) * env
    # ledges and gullies on the cliff face
    face = 4 * env * (1 - env)
    y += 7 * face * fbm(x, 22, sd + 99, 4)
    # fine crisp roughness everywhere on the skyline
    y += 1.4 * fbm(x, 9, sd + 5, 3) * env
    return np.minimum(y, YH)


def layer_path(y, x):
    pts = [(x[0], YH + 1)] + list(zip(x, y)) + [(x[-1], YH + 1)]
    # drop runs that sit on the horizon
    keep = [pts[0]]
    for i in range(1, len(pts) - 1):
        a, b, c = pts[i - 1], pts[i], pts[i + 1]
        if b[1] >= YH - 0.05 and a[1] >= YH - 0.05 and c[1] >= YH - 0.05:
            continue
        keep.append(b)
    keep.append(pts[-1])
    return poly(keep)


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
    # stems leave the water well inboard and sweep out and up (long overhangs)
    bow = stem_centerline(+1, (70, 1.5), (18, 3), (1, 15), (91, -41), 8.6, 1.2)
    stern = stem_centerline(-1, (-70, 1.5), (17, 3), (1, 14), (-89, -38), 7.6, 1.15)

    HJ = -26.0     # sheer meets the stems at this height

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
        dy = 5.0 + K_PERSP * depth - lift      # port sits 5px above water
        out.append((p, dx, dy))
    return out


def ship_svg(th, uid):
    S = SHIP
    p = []
    ink = th['ship']
    # rigging behind the sail
    sx0, sx1 = MAST_X - SAIL_HALF, MAST_X + SAIL_HALF
    stay_w = th['rig_w']
    p.append(f'<g fill="none" stroke="{th["rig"]}" stroke-width="{stay_w}" stroke-linecap="round">')
    bow_top = S['bow'][NRISE + 12]
    stern_top = S['stern'][NRISE + 12]
    p.append(f'<path d="M{f(MAST_X)} {f(MAST_TOP + 2)}L{f(bow_top[0] - 1)} {f(bow_top[1] + 4)}'
             f'M{f(MAST_X)} {f(MAST_TOP + 2)}L{f(stern_top[0] + 2)} {f(stern_top[1] + 5)}'
             f'M{f(MAST_X)} {f(MAST_TOP + 6)}L{f(MAST_X - 30)} {f(S["sheer_at"](MAST_X - 30))}'
             f'M{f(MAST_X)} {f(MAST_TOP + 6)}L{f(MAST_X - 44)} {f(S["sheer_at"](MAST_X - 44))}'
             f'M{f(MAST_X)} {f(MAST_TOP + 6)}L{f(MAST_X + 34)} {f(S["sheer_at"](MAST_X + 34))}"/>')
    p.append('</g>')
    # mast + yard
    p.append(f'<path d="M{f(MAST_X - 1.4)} {f(S["sheer_at"](MAST_X))}L{f(MAST_X - 0.9)} {f(MAST_TOP)}'
             f'L{f(MAST_X + 0.9)} {f(MAST_TOP)}L{f(MAST_X + 1.4)} {f(S["sheer_at"](MAST_X))}Z" fill="{ink}"/>')
    # sail: braced slightly, billowing; vertical cloth stripes
    top_l, top_r = (sx0 - 1, YARD_Y + 1.2), (sx1 + 1, YARD_Y + 1.2)
    bot_l, bot_r = (sx0 + 3, SAIL_BOT), (sx1 + 1, SAIL_BOT + 1)
    sail = (f'M{f(top_l[0])} {f(top_l[1])}L{f(top_r[0])} {f(top_r[1])}'
            f'C{f(sx1 + 6)} {f(-86)} {f(sx1 + 6)} {f(-52)} {f(bot_r[0])} {f(bot_r[1])}'
            f'Q{f(MAST_X + 2)} {f(SAIL_BOT - 9)} {f(bot_l[0])} {f(bot_l[1])}'
            f'C{f(sx0 + 1)} {f(-56)} {f(sx0 - 3)} {f(-86)} {f(top_l[0])} {f(top_l[1])}Z')
    p.append(f'<clipPath id="sailclip{uid}"><path d="{sail}"/></clipPath>')
    p.append(f'<path d="{sail}" fill="url(#sailg{uid})"/>')
    nst = 7
    stripes = []
    for i in range(nst):
        if i % 2 == 1:
            a = sx0 - 6 + (sx1 - sx0 + 12) * i / nst
            b = sx0 - 6 + (sx1 - sx0 + 12) * (i + 1) / nst
            stripes.append(f'M{f(a)} -112L{f(b)} -112L{f(b + 2)} -20L{f(a + 2)} -20Z')
    p.append(f'<path clip-path="url(#sailclip{uid})" d="{"".join(stripes)}" fill="{th["stripe"]}" '
             f'opacity="{th["stripe_op"]}"/>')
    # reef-band seams (horizontal) for cloth texture
    seams = ''.join(f'M{f(sx0 - 4)} {f(y)}L{f(sx1 + 6)} {f(y + 0.6)}' for y in (-90, -72, -54))
    p.append(f'<path clip-path="url(#sailclip{uid})" d="{seams}" stroke="{th["seam"]}" '
             f'stroke-width=".5" fill="none" opacity=".45"/>')
    p.append(f'<path d="{sail}" fill="none" stroke="{th["sail_edge"]}" stroke-width=".7"/>')
    # yard
    p.append(f'<path d="M{f(sx0 - 6)} {f(YARD_Y + .6)}Q{f(MAST_X)} {f(YARD_Y - 1.5)} {f(sx1 + 6)} {f(YARD_Y + .6)}" '
             f'stroke="{ink}" stroke-width="2.2" stroke-linecap="round" fill="none"/>')
    # sheets from the clews to the hull
    p.append(f'<path d="M{f(bot_l[0])} {f(bot_l[1])}L{f(sx0 - 22)} {f(S["sheer_at"](sx0 - 22))}'
             f'M{f(bot_r[0])} {f(bot_r[1])}L{f(sx1 + 20)} {f(S["sheer_at"](sx1 + 20))}" '
             f'stroke="{th["rig"]}" stroke-width="{stay_w}" fill="none"/>')
    # hull
    p.append(f'<path d="{poly(S["hull"])}" fill="{ink}"/>')
    p.append(f'<path d="{poly(S["bow_post"])}" fill="{ink}"/>')
    p.append(f'<path d="{poly(S["stern_post"])}" fill="{ink}"/>')
    # clinker strakes: lines following the sheer, converging on the stems
    strakes = []
    xs, sy = S['xs'], S['sheer'][:, 1]
    for fr in (0.36, 0.62, 0.84):
        y = sy * (1 - fr) + 1.5 * fr
        m = (y < 0.5)
        pts = np.stack([xs[m], y[m]], 1)
        strakes.append(poly(pts, False))
    p.append(f'<clipPath id="hullclip{uid}"><path d="{poly(S["hull"])}"/></clipPath>')
    p.append(f'<path clip-path="url(#hullclip{uid})" d="{"".join(strakes)}" fill="none" stroke="{th["strake"]}" stroke-width=".55"/>')
    # gunwale highlight + stem edge highlight
    p.append(f'<path d="{poly(S["sheer"][3:-3], False)}" fill="none" stroke="{th["rim"]}" stroke-width=".8"/>')
    bowc = S['bow'][22:NRISE + 55]
    p.append(f'<path d="{poly(bowc + np.array([0.2, -0.4]), False)}" fill="none" stroke="{th["rim"]}" '
             f'stroke-width=".6" opacity=".8"/>')
    sterc = S['stern'][22:NRISE + 55]
    p.append(f'<path d="{poly(sterc + np.array([-0.2, -0.4]), False)}" fill="none" stroke="{th["rim"]}" '
             f'stroke-width=".6" opacity=".8"/>')
    # shields along the rail
    sh = []
    for i, x in enumerate(SHIELD_X):
        y = S['sheer_at'](x) + 0.6
        fill = th['shield_a'] if i % 2 == 0 else th['shield_b']
        sh.append(f'<circle cx="{f(x)}" cy="{f(y)}" r="3.4" fill="{fill}"/>')
    p.append(f'<g stroke="{th["shield_rim"]}" stroke-width=".55">{"".join(sh)}</g>')
    p.append('<g fill="' + th['boss'] + '">' + ''.join(
        f'<circle cx="{f(x)}" cy="{f(S["sheer_at"](x) + .6)}" r=".9"/>' for x in SHIELD_X) + '</g>')
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
    return (f'<g id="oar{uid}" stroke="{th["oar"]}" stroke-linecap="round" fill="none">'
            f'<path stroke-width=".85" d="M0 0L{fr[0][1]:.2f} {fr[0][2]:.2f}">'
            f'<animate attributeName="d" dur="{STROKE}s" repeatCount="indefinite" keyTimes="{kt}" values="{shaft}"/></path>'
            f'<path stroke-width="1.9" d="M0 0L1 1">'
            f'<animate attributeName="d" dur="{STROKE}s" repeatCount="indefinite" keyTimes="{kt}" values="{blade}"/></path>'
            f'</g>')


def ship_refl_svg(th, uid):
    """Simplified silhouette for the reflection."""
    S = SHIP
    c = th['refl_ship']
    sx0, sx1 = MAST_X - SAIL_HALF, MAST_X + SAIL_HALF
    sail = (f'M{f(sx0 - 1)} {f(YARD_Y + 1)}L{f(sx1 + 1)} {f(YARD_Y + 1)}'
            f'C{f(sx1 + 6)} -86 {f(sx1 + 6)} -52 {f(sx1 + 1)} {f(SAIL_BOT + 1)}'
            f'Q{f(MAST_X + 2)} {f(SAIL_BOT - 9)} {f(sx0 + 3)} {f(SAIL_BOT)}'
            f'C{f(sx0 + 1)} -56 {f(sx0 - 3)} -86 {f(sx0 - 1)} {f(YARD_Y + 1)}Z')
    return (f'<g id="shipRefl{uid}">'
            f'<path d="{sail}" fill="{th["refl_sail"]}"/>'
            f'<path d="M{f(MAST_X - 1.2)} -12L{f(MAST_X - 1)} {f(MAST_TOP)}L{f(MAST_X + 1)} {f(MAST_TOP)}L{f(MAST_X + 1.2)} -12Z" fill="{c}"/>'
            f'<path d="{poly(S["hull"])}{poly(S["bow_post"])}{poly(S["stern_post"])}" fill="{c}"/>'
            f'<g opacity=".9">{oar_uses(S, uid)}</g>'
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


def wake_svg(th):
    out = []
    c = th['wake']
    slope = K_PERSP * math.tan(math.radians(19.5))
    B = (73.0, 1.4)
    # continuous arms with fading gradient (ship-frame gradient)
    for side in (+1, -1):
        L = WAKE_LEN if side > 0 else WAKE_LEN * 0.95
        ex, ey = B[0] - L, B[1] + side * slope * L
        out.append(f'<path d="M{f(B[0])} {f(B[1])}L{f(ex)} {f(ey)}" stroke="url(#{"wakenear" if side > 0 else "wakefade"})" '
                   f'stroke-width="{0.9 if side > 0 else 0.7}" fill="none"/>')
        # featherlets: short crests at an angle along the arm
        nfe = 13
        for i in range(1, nfe + 1):
            s = L * (i / (nfe + 1)) ** 1.15
            if side < 0 and B[0] - s > -84:
                continue      # hidden behind the hull
            if side > 0 and B[0] - s > -40:
                continue      # still tucked under the oars
            px, py = B[0] - s, B[1] + side * slope * s
            ln = 7 + 10 * (s / L)
            dx, dy = -0.92 * ln, side * 0.10 * ln * 0.9
            op = 0.7 * (1 - s / L) ** 1.3
            out.append(f'<path d="M{f(px)} {f(py)}l{f(dx)} {f(dy)}" stroke="{c}" '
                       f'stroke-width=".8" opacity="{op:.2f}" fill="none"/>')
    # bow wave curl
    out.append(f'<path d="M72 2.4q-6 2.2-15 2.4M70 3.8q-8 2-19 1.6" stroke="{c}" stroke-width=".8" '
               f'fill="none" opacity=".55"/>')
    # stern trail: dashes that stay put in the water (move back at ship speed)
    per = 26.0
    dur = per / V
    for j, (dy, op, w) in enumerate(((2.6, .55, .8), (5.2, .35, .7), (-0.6, .3, .6))):
        out.append(f'<path d="M-70 {f(dy)}L{f(-70 - 230)} {f(dy + j * 1.5)}" stroke="url(#trailfade)" '
                   f'stroke-width="{w}" stroke-dasharray="{12 - 3 * j} {14 + 3 * j}" opacity="{op}" fill="none">'
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
            f'<ellipse id="pud{uid}{k}" cx="0" cy="0" rx="1" ry=".3" fill="none" stroke="{th["wake"]}" '
            f'stroke-width=".7" opacity="0">'
            f'<animate attributeName="rx" values="2;14;14" keyTimes="0;{kt_end:.3f};1" dur="{dur}s" '
            f'begin="{k * STROKE}s" repeatCount="indefinite"/>'
            f'<animate attributeName="ry" values=".6;3.6;3.6" keyTimes="0;{kt_end:.3f};1" dur="{dur}s" '
            f'begin="{k * STROKE}s" repeatCount="indefinite"/>'
            f'<animate attributeName="opacity" values="0;.38;0;0" keyTimes="0;.05;{kt_end:.3f};1" dur="{dur}s" '
            f'begin="{k * STROKE}s" repeatCount="indefinite"/>'
            f'<animateTransform attributeName="transform" type="translate" values="0 0;{-back:.1f} 0;{-back:.1f} 0" '
            f'keyTimes="0;{kt_end:.3f};1" dur="{dur}s" begin="{k * STROKE}s" repeatCount="indefinite"/>'
            f'</ellipse>')
    uses = []
    for x in OAR_X:
        for k in range(2):
            uses.append(f'<use href="#pud{uid}{k}" x="{f(x + cdx)}" y="{f(cdy - 6 + 6 + SHIP["sheer_at"](x) + 5.4 + 0.4)}"/>')
    return ''.join(defs), ''.join(uses)



# ------------------------------------------------------------------ helpers
def sine_vals(fn, n=24):
    """values= list sampling one period of fn(phase in [0,1))."""
    return ';'.join(fn(i / n) for i in range(n + 1))


def stops_at(pairs, y0, y1):
    """Gradient stops given at absolute y (userSpace gradient from y0 to y1)."""
    out = []
    for y, col, op in pairs:
        o = min(1.0, max(0.0, (y - y0) / (y1 - y0)))
        out.append(f'<stop offset="{o:.3f}" stop-color="{col}"' + (f' stop-opacity="{op:.3f}"' if op < 1 else '') + '/>')
    return ''.join(out)


# ------------------------------------------------------------------ fog
FOG_P = 1560.0      # horizontal period of every drifting fog layer
FOG_RX = 165.0      # largest puff half-width
FOG_D = 170.0       # start margin so puffs never pop at the wrap

# top of the rolling fog bank in front of each wall (screen y); index = LAYERS index
FOG_Y = [190, 182, 170, 160, 152, 146, 142]

# drifting banks: y = top of the billows, depth = body below it, n puffs per period,
# rx range, flat = ry/rx range, op = bank opacity, v = drift speed (px/s, with the
# wind, i.e. rightward), bob = slow rise/sink amplitude (px) and period (s)
BANKS = {
    'sky':   dict(y=96, depth=4, spread=6, n=14, rx=(80, 165), flat=(.06, .11), op=.5, v=2.2, bob=2, bobT=71, wave=3, caps=0, seed=101),
    0:       dict(y=FOG_Y[0], depth=18, spread=4, n=20, rx=(45, 140), flat=(.18, .34), op=.8, v=2.6, bob=1.5, bobT=53, wave=4, caps=0.4, seed=111),
    1:       dict(y=FOG_Y[1], depth=20, spread=4, n=22, rx=(45, 140), flat=(.18, .34), op=.8, v=3.1, bob=1.8, bobT=47, wave=5, caps=0.4, seed=121),
    2:       dict(y=FOG_Y[2], depth=22, spread=5, n=24, rx=(45, 150), flat=(.2, .36), op=.82, v=3.8, bob=2.2, bobT=59, wave=7, caps=0.4, subs=2, seed=131),
    3:       dict(y=FOG_Y[3], depth=24, spread=5, n=24, rx=(45, 150), flat=(.2, .38), op=.85, v=4.6, bob=2.6, bobT=43, wave=9, caps=0.4, subs=2, seed=141),
    4:       dict(y=FOG_Y[4], depth=26, spread=6, n=24, rx=(50, 160), flat=(.2, .4), op=.85, v=5.6, bob=3.0, bobT=61, wave=10, caps=0.4, subs=2, seed=151),
    5:       dict(y=FOG_Y[5], depth=28, spread=6, n=24, rx=(50, 165), flat=(.2, .4), op=.85, v=6.8, bob=3.2, bobT=37, wave=11, caps=0.4, subs=2, seed=161),
    6:       dict(y=FOG_Y[6], depth=30, spread=7, n=24, rx=(50, 165), flat=(.2, .4), op=.85, v=8.0, bob=3.4, bobT=67, wave=12, caps=0.4, subs=2, seed=171),
    'lane':  dict(y=236, depth=12, spread=4, n=22, rx=(70, 165), flat=(.1, .2), op=.55, v=10.0, bob=2.0, bobT=41, wave=4, caps=0.3, seed=181),
    'front': dict(y=292, depth=6, spread=5, n=22, rx=(50, 150), flat=(.06, .12), op=.4, v=13.0, bob=2.4, bobT=29, wave=3, caps=0, seed=191),
}


def puff_group(gid, spec, seed, k=1.0):
    r = np.random.default_rng(seed)
    n = spec['n']
    # slow undulation of the bank's top, periodic in FOG_P so the wrap is seamless:
    # tongues of fog creep up the walls here, dips reveal more cliff there
    k1, k2 = r.choice([2, 3]), r.choice([5, 6, 7])
    p1, p2 = r.uniform(0, 2 * np.pi, 2)
    A = spec.get('wave', 0.0)

    def top(x):
        return spec['y'] + A * (0.7 * math.sin(2 * math.pi * k1 * x / FOG_P + p1)
                                + 0.3 * math.sin(2 * math.pi * k2 * x / FOG_P + p2))
    out = []
    xs = (np.arange(n) + r.uniform(0.1, 0.9, n)) * FOG_P / n
    for x in xs:
        rx = r.uniform(*spec['rx'])
        ry = rx * r.uniform(*spec['flat'])
        # flat-bottomed bank: big puffs hang below the top line
        y = top(x) + spec['depth'] * r.uniform() ** 1.5 + 0.55 * ry + r.normal(0, spec['spread'])
        out.append(f'<ellipse cx="{f(x)}" cy="{f(y)}" rx="{f(rx)}" ry="{f(ry)}" fill-opacity="{k * r.uniform(.45, 1.0):.2f}"/>')
    # small rounder caps along the top give the edge a soft, lumpy billow
    nc = int(n * spec.get('caps', 0))
    for x in (np.arange(nc) + r.uniform(0.1, 0.9, nc)) * FOG_P / nc:
        rx = r.uniform(22, 55)
        ry = rx * r.uniform(.3, .48)
        y = top(x) + 0.35 * ry + r.normal(0, spec['spread'] * 0.8)
        out.append(f'<ellipse cx="{f(x)}" cy="{f(y)}" rx="{f(rx)}" ry="{f(ry)}" fill-opacity="{k * r.uniform(.35, .8):.2f}"/>')
    return f'<g id="{gid}">{"".join(out)}</g>'


def drift(v, phase):
    """translate animation for a layer drifting rightward at v px/s, periodic in FOG_P."""
    dur = FOG_P / v
    a, b = -FOG_D - FOG_P, -FOG_D
    o = a + phase * FOG_P
    k = 1 - phase
    return (f'<animateTransform attributeName="transform" type="translate" dur="{dur:.1f}s" '
            f'repeatCount="indefinite" calcMode="linear" values="{f(o)} 0;{f(b)} 0;{f(a)} 0;{f(o)} 0" '
            f'keyTimes="0;{k:.4f};{k:.4f};1"/>')


def bank_svg(th, uid, key, defs):
    spec = BANKS[key]
    r = np.random.default_rng(spec['seed'] + 7)
    subs = []
    layers = ((1.0, 0), (1.37, 1)) if spec.get('subs', 1) == 2 else ((1.0, 0),)
    for j, (vm, sd) in enumerate(layers):
        gid = f'fb{uid}{key}{j}'
        defs.append(puff_group(gid, spec, spec['seed'] + sd, spec['op'] * th['fog_op']))
        subs.append(f'<g>{drift(spec["v"] * vm, r.uniform())}'
                    f'<use href="#{gid}"/><use href="#{gid}" x="{f(FOG_P)}"/></g>')
    amp, T = spec['bob'], spec['bobT']
    ph = r.uniform()
    bob = sine_vals(lambda p: f'0 {amp * math.sin(2 * math.pi * (p + ph)):.2f}')
    return (f'<g fill="url(#puff{uid})">'
            f'<animateTransform attributeName="transform" type="translate" dur="{T}s" repeatCount="indefinite" values="{bob}"/>'
            f'{"".join(subs)}</g>')


# two large fog masses on the flanks; the clear patch between them is where
# the ship turns crisp. They breathe slowly (translate + opacity).
VEILS = [
    dict(cx=40, cy=262, rx=470, ry=165, op=.97, dx=40, T=53, oT=37, x0=-400, x1=760, ph=0.0),
    dict(cx=1190, cy=258, rx=420, ry=158, op=.97, dx=36, T=47, oT=43, x0=520, x1=1600, ph=0.4),
]


def veils_svg(th, uid, defs):
    out = []
    for i, v in enumerate(VEILS):
        gid = f'veil{uid}{i}'
        mo = v['op'] * th['veil_op']
        defs.append(f'<radialGradient id="{gid}" gradientUnits="userSpaceOnUse" cx="0" cy="0" r="1" '
                    f'gradientTransform="translate({v["cx"]} {v["cy"]}) scale({v["rx"]} {v["ry"]})">'
                    + ''.join(f'<stop offset="{o}" stop-color="{th["veil"]}" stop-opacity="{mo * a:.3f}"/>'
                              for o, a in ((0, 1), (.4, .97), (.6, .8), (.77, .46), (.9, .15), (1, 0)))
                    + '</radialGradient>')
        mv = sine_vals(lambda p: f'{v["dx"] * math.sin(2 * math.pi * (p + v["ph"])):.1f} '
                                 f'{3 * math.sin(4 * math.pi * (p + v["ph"])):.1f}')
        ov = sine_vals(lambda p: f'{0.92 + 0.08 * math.sin(2 * math.pi * (p + v["ph"])):.3f}', 12)
        out.append(f'<rect x="{v["x0"]}" y="-20" width="{v["x1"] - v["x0"]}" height="{H + 40}" fill="url(#{gid})">'
                   f'<animateTransform attributeName="transform" type="translate" dur="{v["T"]}s" '
                   f'repeatCount="indefinite" values="{mv}"/>'
                   f'<animate attributeName="fill-opacity" dur="{v["oT"]}s" repeatCount="indefinite" values="{ov}"/></rect>')
    return ''.join(out)


def drizzle_svg(th, uid, defs):
    """Fine slanted streaks in a skewed pattern; the fill rect slides by one lattice vector."""
    out = []
    slope = 0.2
    sk = math.degrees(math.atan(slope))
    for k, (TW, TH, n, ln, w, dur, op, seed) in enumerate(((240, 400, 12, (6, 10), .6, 6.5, 1.0, 201),
                                                             (180, 300, 16, (3, 6), .45, 8.5, .6, 202))):
        r = np.random.default_rng(seed)
        lines = []
        for _ in range(n):
            L = r.uniform(*ln)
            x, y = r.uniform(2, TW - 2), r.uniform(1, TH - L - 1)
            lines.append(f'<path d="M{f(x)} {f(y)}v{f(L)}" stroke-opacity="{r.uniform(.4, 1):.2f}"/>')
        pid = f'dz{uid}{k}'
        defs.append(f'<pattern id="{pid}" width="{TW}" height="{TH}" patternUnits="userSpaceOnUse" '
                    f'patternTransform="skewX({sk:.3f})">'
                    f'<g stroke="{th["drizzle"]}" stroke-width="{w}" stroke-linecap="round">{"".join(lines)}</g></pattern>')
        dx, dy = slope * TH, TH
        out.append(f'<rect x="-{f(dx + 20)}" y="-{f(dy + 20)}" width="{f(W + dx + 40)}" height="{f(H + dy + 40)}" '
                   f'fill="url(#{pid})" fill-opacity="{op * th["drizzle_op"]:.3f}">'
                   f'<animateTransform attributeName="transform" type="translate" dur="{dur}s" '
                   f'repeatCount="indefinite" values="0 0;{f(dx)} {f(dy)}"/></rect>')
    return ''.join(out)


def rain_rings_svg(th):
    """Tiny drizzle rings on the water; each slot hops to a new spot every cycle."""
    r = np.random.default_rng(303)
    out = [f'<g fill="none" stroke="{th["ring"]}" stroke-width=".6">']
    for k in range(16):
        D = r.uniform(2.6, 4.2)
        K = 5
        pos = []
        for _ in range(K):
            y = YH + 10 + (H - YH - 12) * r.uniform() ** 0.7
            pos.append((r.uniform(10, W - 10), y))
        s = (pos[0][1] - YH) / (H - YH)
        rr = 2.2 + 5.5 * s
        pv = ';'.join(f'{f(x)} {f(y)}' for x, y in pos)
        kt = ';'.join(f'{i / K:.3f}' for i in range(K))
        b = r.uniform(0, D)
        out.append(f'<g><animateTransform attributeName="transform" type="translate" calcMode="discrete" '
                   f'dur="{K * D:.2f}s" begin="-{b:.2f}s" repeatCount="indefinite" values="{pv}" keyTimes="{kt}"/>'
                   f'<ellipse rx="1" ry=".25" stroke-opacity="0">'
                   f'<animate attributeName="rx" values=".5;{rr:.1f};{rr:.1f}" keyTimes="0;.7;1" dur="{D:.2f}s" begin="-{b:.2f}s" repeatCount="indefinite"/>'
                   f'<animate attributeName="ry" values=".12;{rr * .24:.2f};{rr * .24:.2f}" keyTimes="0;.7;1" dur="{D:.2f}s" begin="-{b:.2f}s" repeatCount="indefinite"/>'
                   f'<animate attributeName="stroke-opacity" values="0;{th["ring_op"]};0;0" keyTimes="0;.06;.7;1" dur="{D:.2f}s" begin="-{b:.2f}s" repeatCount="indefinite"/>'
                   f'</ellipse></g>')
    out.append('</g>')
    return ''.join(out)


def ripples_svg(th):
    r = np.random.default_rng(23)
    out = [f'<g stroke="{th["ripple"]}" stroke-width=".6" fill="none" stroke-linecap="round">']
    y = YH + 4
    gap = 3.0
    while y < H:
        depth = (y - YH) / (H - YH)
        dash = r.uniform(40, 160) * (0.5 + depth)
        space = r.uniform(30, 220) * (0.6 + depth)
        per = dash + space
        op = th['ripple_op'] * (0.3 + 0.7 * depth)
        dur = r.uniform(30, 60)
        d = 1 if r.uniform() < .5 else -1
        out.append(f'<path d="M-10 {f(y)}H{W + 10}" stroke-dasharray="{f(dash)} {f(space)}" stroke-opacity="{op:.2f}">'
                   f'<animate attributeName="stroke-dashoffset" values="0;{f(d * per)}" dur="{dur:.1f}s" '
                   f'repeatCount="indefinite"/></path>')
        y += gap
        gap *= 1.12
    out.append('</g>')
    return ''.join(out)


def glade_svg(th, uid, defs):
    """Diffuse light on the water under the veiled sun: soft column, sparse glints."""
    r = np.random.default_rng(17)
    groups = [[] for _ in range(4)]
    y = YH + 3
    gap = 2.6
    while y < H:
        depth = (y - YH) / (H - YH)
        halfw = 20 + 90 * depth ** 0.9
        for _ in range(int(1 + 3 * depth ** 0.5)):
            x = LIGHT_X + r.normal(0, 0.45) * halfw
            ln = r.uniform(3, 8) * (0.6 + 1.2 * depth)
            cen = math.exp(-((x - LIGHT_X) / (halfw * 0.75)) ** 2)
            op = (0.3 + 0.7 * cen) * (1 - 0.5 * depth)
            groups[r.integers(0, 4)].append(f'<path d="M{f(x - ln / 2)} {f(y)}h{f(ln)}" stroke-opacity="{op:.2f}"/>')
        y += gap
        gap *= 1.07
    out = [f'<g stroke="{th["glade"]}" stroke-width="1" stroke-linecap="round" opacity="{th["glade_op"]}">']
    for i, g in enumerate(groups):
        dur = 3.7 + 1.1 * i
        vals = ';'.join(f'{0.3 + 0.7 * (0.5 + 0.5 * math.sin(2 * math.pi * (j / 6 + i / 4))):.2f}' for j in range(7))
        out.append(f'<g>{"".join(g)}<animate attributeName="opacity" values="{vals}" dur="{dur:.2f}s" repeatCount="indefinite"/></g>')
    out.append('</g>')
    defs.append(f'<radialGradient id="col{uid}" cx="{LIGHT_X}" cy="{YH}" r="120" gradientUnits="userSpaceOnUse" '
                f'gradientTransform="translate({LIGHT_X} {YH}) scale(.6 1) translate({-LIGHT_X} {-YH})">'
                f'<stop offset="0" stop-color="{th["glade"]}" stop-opacity="{th["col_op"]}"/>'
                f'<stop offset="1" stop-color="{th["glade"]}" stop-opacity="0"/></radialGradient>')
    return f'<rect x="{LIGHT_X - 80}" y="{YH}" width="160" height="{H - YH}" fill="url(#col{uid})"/>' + ''.join(out)


WATERFALLS = [(3, 896.0)]


def waterfall_svg(th, x0, prof, xs):
    """A thin thread falling from the fjell edge, vanishing into the fog."""
    ytop = float(np.interp(x0, xs, prof)) + 1.0
    ybot = min(ytop + 0.62 * (YH - ytop), FOG_Y[3] + 6)
    n = 30
    ys = np.linspace(ytop, ybot, n)
    xx = x0 + 1.4 * (1 - np.exp(-(ys - ytop) / 5)) + 0.35 * np.sin(ys / 7)
    d = poly(list(zip(xx, ys)), False)
    gid = f'fall{int(x0)}{th["fall"][1:]}'
    return (f'<defs><linearGradient id="{gid}" x1="0" y1="{f(ytop)}" x2="0" y2="{f(ybot)}" gradientUnits="userSpaceOnUse">'
            f'<stop offset="0" stop-color="{th["fall"]}" stop-opacity="{th["fall_op"]}"/>'
            f'<stop offset=".5" stop-color="{th["fall"]}" stop-opacity="{th["fall_op"] * .6:.2f}"/>'
            f'<stop offset="1" stop-color="{th["fall"]}" stop-opacity="0"/></linearGradient></defs>'
            f'<g fill="none" stroke="url(#{gid})" stroke-linecap="round">'
            f'<path d="{d}" stroke-width=".8" opacity=".55"/>'
            f'<path d="{d}" stroke-width=".9" stroke-dasharray="5 4 2 5" opacity=".9">'
            f'<animate attributeName="stroke-dashoffset" values="0;-16" dur="1.1s" repeatCount="indefinite"/></path>'
            f'</g>')


# ------------------------------------------------------------------ scene
def theme_svg(th, uid):
    xs = np.arange(-4, W + 5, 2.5)
    parts = []
    defs = []
    fog = th['fog']

    defs.append(f'<linearGradient id="sky{uid}" x1="0" y1="0" x2="0" y2="{YH}" gradientUnits="userSpaceOnUse">'
                + ''.join(f'<stop offset="{o}" stop-color="{c}"/>' for o, c in th['sky']) + '</linearGradient>')
    defs.append(f'<radialGradient id="glow{uid}" cx="{LIGHT_X}" cy="{LIGHT_Y}" r="300" gradientUnits="userSpaceOnUse" '
                f'gradientTransform="translate({LIGHT_X} {LIGHT_Y}) scale(1.3 .7) translate({-LIGHT_X} {-LIGHT_Y})">'
                f'<stop offset="0" stop-color="{th["glow"]}" stop-opacity="{th["glow_op"]}"/>'
                f'<stop offset=".3" stop-color="{th["glow"]}" stop-opacity="{th["glow_op"] * .4:.3f}"/>'
                f'<stop offset="1" stop-color="{th["glow"]}" stop-opacity="0"/></radialGradient>')
    defs.append(f'<radialGradient id="halo{uid}" cx="{LIGHT_X}" cy="{LIGHT_Y}" r="40" gradientUnits="userSpaceOnUse">'
                f'<stop offset=".35" stop-color="{th["halo"]}" stop-opacity="{th["halo_op"]}"/>'
                f'<stop offset="1" stop-color="{th["halo"]}" stop-opacity="0"/></radialGradient>')
    defs.append(f'<radialGradient id="disc{uid}">'
                f'<stop offset=".55" stop-color="{th["disc"]}" stop-opacity="{th["disc_op"]}"/>'
                f'<stop offset="1" stop-color="{th["disc"]}" stop-opacity="0"/></radialGradient>')
    defs.append(f'<radialGradient id="puff{uid}">'
                + ''.join(f'<stop offset="{o}" stop-color="{th['fog_hi']}" stop-opacity="{a}"/>'
                          for o, a in ((0, 1), (.25, .92), (.5, .6), (.72, .27), (.88, .08), (1, 0)))
                + '</radialGradient>')
    defs.append(f'<linearGradient id="water{uid}" x1="0" y1="{YH}" x2="0" y2="{H}" gradientUnits="userSpaceOnUse">'
                + ''.join(f'<stop offset="{o}" stop-color="{c}"/>' for o, c in th['water']) + '</linearGradient>')
    defs.append(f'<linearGradient id="sailg{uid}" x1="0" y1="{YARD_Y}" x2="0" y2="{SAIL_BOT}" gradientUnits="userSpaceOnUse">'
                f'<stop offset="0" stop-color="{th["sail_top"]}"/><stop offset="1" stop-color="{th["sail_bot"]}"/></linearGradient>')
    defs.append(f'<linearGradient id="wakefade" x1="77" y1="0" x2="{77 - WAKE_LEN}" y2="0" gradientUnits="userSpaceOnUse">'
                f'<stop offset="0" stop-color="{th["wake"]}" stop-opacity=".6"/>'
                f'<stop offset=".55" stop-color="{th["wake"]}" stop-opacity="0"/></linearGradient>')
    defs.append(f'<linearGradient id="wakenear" x1="73" y1="0" x2="{73 - WAKE_LEN}" y2="0" gradientUnits="userSpaceOnUse">'
                f'<stop offset=".3" stop-color="{th["wake"]}" stop-opacity="0"/>'
                f'<stop offset=".55" stop-color="{th["wake"]}" stop-opacity=".45"/>'
                f'<stop offset="1" stop-color="{th["wake"]}" stop-opacity="0"/></linearGradient>')
    defs.append(f'<linearGradient id="trailfade" x1="-74" y1="0" x2="-304" y2="0" gradientUnits="userSpaceOnUse">'
                f'<stop offset="0" stop-color="{th["wake"]}" stop-opacity="1"/>'
                f'<stop offset="1" stop-color="{th["wake"]}" stop-opacity="0"/></linearGradient>')

    # sky, veiled light, high wisps
    parts.append(f'<rect width="{W}" height="{YH + 2}" fill="url(#sky{uid})"/>')
    parts.append(f'<rect width="{W}" height="{YH + 2}" fill="url(#glow{uid})"/>')
    parts.append(f'<circle cx="{LIGHT_X}" cy="{LIGHT_Y}" r="40" fill="url(#halo{uid})"/>')
    parts.append(f'<circle cx="{LIGHT_X}" cy="{LIGHT_Y}" r="{LIGHT_R * 1.15:.1f}" fill="url(#disc{uid})"/>')
    parts.append(bank_svg(th, uid, 'sky', defs))

    # fjord walls, each sinking into its own rolling fog bank
    for i, L in enumerate(LAYERS):
        y = layer_profile(L, xs)
        col = th['mtn'][i]
        topy = float(y.min())
        fy = FOG_Y[i]
        gid = f'm{uid}{i}'
        defs.append(f'<linearGradient id="{gid}" x1="0" y1="{f(topy)}" x2="0" y2="{YH}" gradientUnits="userSpaceOnUse">'
                    + stops_at([(fy - 6, col, 1), (fy + 10, mix(col, fog, .35), 1), (fy + 30, mix(col, fog, .62), 1),
                                (fy + 56, mix(col, fog, .84), 1), (YH, mix(col, fog, .94), 1)], topy, YH)
                    + '</linearGradient>')
        parts.append(f'<path d="{layer_path(y, xs)}" fill="url(#{gid})"/>')
        # faint rim on the ridge, fading out before the fog line
        m = y < fy + 14
        seg, run = [], []
        for xx, yy, mm in zip(xs, y, m):
            if mm:
                run.append((xx, yy))
            elif run:
                seg.append(run)
                run = []
        if run:
            seg.append(run)
        dd = ''.join(poly(r_, False) for r_ in seg if len(r_) > 2)
        if dd and th['ridge_op'][i] > 0:
            rg = f'rg{uid}{i}'
            defs.append(f'<linearGradient id="{rg}" x1="0" y1="{f(topy)}" x2="0" y2="{fy + 14}" gradientUnits="userSpaceOnUse">'
                        f'<stop offset="0" stop-color="{th["ridge"]}" stop-opacity="{th["ridge_op"][i]}"/>'
                        f'<stop offset=".6" stop-color="{th["ridge"]}" stop-opacity="{th["ridge_op"][i] * .6:.3f}"/>'
                        f'<stop offset="1" stop-color="{th["ridge"]}" stop-opacity="0"/></linearGradient>')
            parts.append(f'<path d="{dd}" fill="none" stroke="url(#{rg})" stroke-width=".7"/>')
        for wl, wx in WATERFALLS:
            if wl == i:
                parts.append(waterfall_svg(th, wx, y, xs))
        parts.append(bank_svg(th, uid, i, defs))

    # water
    parts.append(f'<rect y="{YH}" width="{W}" height="{H - YH}" fill="url(#water{uid})"/>')
    # (no mirrored walls: the fog lying on the water swallows their reflection)
    defs.append(f'<linearGradient id="wtint{uid}" x1="0" y1="{YH}" x2="0" y2="{H}" gradientUnits="userSpaceOnUse">'
                f'<stop offset="0" stop-color="{fog}" stop-opacity=".0"/>'
                f'<stop offset="1" stop-color="{th["water"][-1][1]}" stop-opacity="{th["wtint_op"]}"/></linearGradient>')
    parts.append(f'<rect y="{YH}" width="{W}" height="{H - YH}" fill="url(#wtint{uid})"/>')
    defs.append(f'<linearGradient id="shore{uid}" x1="0" y1="{YH - 14}" x2="0" y2="{YH + 20}" gradientUnits="userSpaceOnUse">'
                f'<stop offset="0" stop-color="{fog}" stop-opacity="0"/>'
                f'<stop offset=".45" stop-color="{fog}" stop-opacity="{th["shore_op"]}"/>'
                f'<stop offset="1" stop-color="{fog}" stop-opacity="0"/></linearGradient>')
    parts.append(f'<rect y="{YH - 14}" width="{W}" height="34" fill="url(#shore{uid})"/>')
    parts.append(glade_svg(th, uid, defs))
    parts.append(ripples_svg(th))
    parts.append(rain_rings_svg(th))
    parts.append(bank_svg(th, uid, 'lane', defs))

    # ship reflection in swaying strips
    pud_defs, pud_uses = puddles_svg(th, uid)
    defs.append(oar_defs(th, uid))
    defs.append(ship_refl_svg(th, uid))
    defs.append(pud_defs)
    strips = []
    y0 = YW - 1.5
    hgt = 3.2
    k = 0
    while y0 < H:
        h = hgt
        cid = f'rs{uid}{k}'
        defs.append(f'<clipPath id="{cid}"><rect x="0" y="{f(y0)}" width="{W}" height="{f(h - th["gap"] * min(1, k / 3))}"/></clipPath>')
        amp = 0.6 + 0.16 * k
        per = 3.4 + 0.9 * math.sin(k * 1.7) + 0.05 * k
        ph = (k * 0.37) % 1
        vals = ';'.join(f'{amp * math.sin(2 * math.pi * (i / 8 + ph)):.2f} 0' for i in range(9))
        op = min(1.0, max(0.0, 1 - (y0 - YW) / 120))
        strips.append(
            f'<g clip-path="url(#{cid})" opacity="{op:.2f}"><g>'
            f'<animateTransform attributeName="transform" type="translate" values="{vals}" dur="{per:.2f}s" repeatCount="indefinite"/>'
            f'<g transform="translate(0 {f(2 * YW)}) scale(1 -1)"><g>{mover_anim()}'
            f'<use href="#shipRefl{uid}"/></g></g></g></g>')
        y0 += h
        hgt = min(hgt * 1.13 + 0.25, 11)
        k += 1
    parts.append(f'<g opacity="{th["ship_refl_op"]}">{"".join(strips)}</g>')

    # ship, wake, puddles (all in the ship frame)
    parts.append(f'<g>{mover_anim()}'
                 f'<g>{wake_svg(th)}</g>'
                 f'<g>{pud_uses}</g>'
                 f'<g>{surge_anim()}{ship_svg(th, uid)}</g>'
                 f'</g>')
    # low wisps sliding past the hull, then the flanking fog masses, then drizzle
    parts.append(bank_svg(th, uid, 'front', defs))
    parts.append(veils_svg(th, uid, defs))
    parts.append(drizzle_svg(th, uid, defs))
    return ''.join(defs), ''.join(parts)


def ramp(a, b, n, g):
    return [mix(a, b, (i / (n - 1)) ** g) for i in range(n)]


THEMES = {
    'dark': dict(
        bg='#0d1117', border='#30363d',
        sky=[(0, '#080b10'), (.45, '#10161e'), (1, '#1f2833')],
        glow='#9fb0c4', glow_op=.16, halo='#c9d2dc', halo_op=.22, disc='#c3ccd6', disc_op=.42,
        fog='#2f3a47', fog_hi='#3d4959', veil='#37434f', fog_op=1.0, veil_op=1.0,
        fall='#7d8da2', fall_op=.45,
        mtn=ramp('#34404e', '#18202a', 7, 1.0),
        ridge='#8796aa', ridge_op=[.2, .22, .25, .28, .3, .33, .36],
        water=[(0, '#242e3a'), (.4, '#18202a'), (1, '#0e131a')],
        wtint_op=.5, shore_op=.7,
        glade='#c9d2dc', glade_op=.35, col_op=.10,
        ripple='#4d5b6d', ripple_op=.35, ring='#6b7a8e', ring_op=.5,
        drizzle='#8a99ac', drizzle_op=.16,
        ship='#06090d', rig='#3a4656', rig_w=.55, strake='#1f2833', rim='#4e5d72',
        shield_a='#151c26', shield_b='#26303d', shield_rim='#4a586b', boss='#5d6b7e',
        oar='#0a0e14',
        sail_top='#64372d', sail_bot='#4d2b25', stripe='#a0705c', stripe_op=.55, seam='#000000',
        sail_edge='#7a4a3c',
        wake='#7f90a6', refl_ship='#070a0e', refl_sail='#3a2522', ship_refl_op=.8, gap=1.0,
    ),
    'light': dict(
        bg='#ffffff', border='#d0d7de',
        sky=[(0, '#d2d8de'), (.5, '#e2e6e9'), (1, '#edeff0')],
        glow='#ffffff', glow_op=.5, halo='#ffffff', halo_op=.6, disc='#fcfcfb', disc_op=.95,
        fog='#eceef0', fog_hi='#f7f8f8', veil='#eef0f1', fog_op=1.0, veil_op=1.0,
        fall='#ffffff', fall_op=.8,
        mtn=ramp('#bcc4cb', '#4c5763', 7, 0.9),
        ridge='#2f3943', ridge_op=[.12, .14, .16, .18, .2, .22, .24],
        water=[(0, '#e9ecee'), (.4, '#dfe4e7'), (1, '#d2d9de')],
        wtint_op=.45, shore_op=.75,
        glade='#ffffff', glade_op=.8, col_op=.5,
        ripple='#7d8894', ripple_op=.28, ring='#7b8794', ring_op=.45,
        drizzle='#6f7a86', drizzle_op=.16,
        ship='#1b222a', rig='#4a5561', rig_w=.6, strake='#38424d', rim='#4c5865',
        shield_a='#2a333d', shield_b='#5a6570', shield_rim='#1b222a', boss='#c9cfd5',
        oar='#1f262e',
        sail_top='#9a5644', sail_bot='#874a3a', stripe='#efe6da', stripe_op=.62, seam='#3a2a25',
        sail_edge='#5a3b32',
        wake='#4f5b68', refl_ship='#3a434e', refl_sail='#9a6c5f', ship_refl_op=.55, gap=1.0,
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
