#!/usr/bin/env python3
"""Midnight-sun variant of the fjord + Viking longship banner.

The same fjord and the same Gokstad/Oseberg-proportioned longship as
designs/longship, re-lit by a low Arctic sun skimming the far ridge.

Light model
-----------
* sky: deep dusk overhead grading through rose to gold at the horizon, with a
  wide, flattened horizon glow and a few thin, sun-lit haze strands;
* the sun sits just above the distant range in the fjord mouth, slightly
  flattened by refraction, and drifts almost imperceptibly along the horizon
  (a slow there-and-back over two crossings);
* aerial perspective: the nearest fjord walls are warm, near-black
  silhouettes, each layer further away is mixed toward a pale blue haze; rose
  mist pools between the walls; ridge lines carry rim light whose strength
  falls off with distance from the sun; a faint veiling glare lies over all;
* water reflects the sky (gold near the horizon, dusk toward the viewer); a
  long, narrow sun path of glints runs from the horizon to the foreground;
* the ship is a crisp silhouette. Rim light is made the physical way: a
  gold copy of each silhouette sits behind it, offset toward the sun, so only
  edges that face the sun catch light. The offset swings with the ship's
  position relative to the sun (bow-side rims while approaching, stern-side
  after passing). When the ship crosses in front of the sun, the sun shows
  through the woollen sail, which glows like a lantern.

Motion (all SMIL, seamless): the ship rows across once per LOOP seconds, oars
in unison with puddle rings and a Kelvin wake; the sun path glitters; haze
and mist drift; the sun drifts.

Usage: python generate.py  -> banner-dark.svg, banner-light.svg
"""
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

W, H = 1200, 340
YH = 226.0          # horizon (water line of the distant shore)
YW = 276.0          # ship's water line
LIGHT_X, LIGHT_R = 640.0, 13.0   # LIGHT_Y is set from the far ridge below
SUN_DRIFT = 7.0     # px either side, over two crossings

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
    dict(kind='range', top=184, seed=3),
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


# the sun's lower limb skims the far ridge
LIGHT_Y = float(layer_profile(LAYERS[0], np.array([LIGHT_X - 8, LIGHT_X, LIGHT_X + 8.0])).min()) - LIGHT_R - 2.0


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


def ship_x_frac(p):
    """Ship centre (world x) at loop fraction p, matching mover_anim()."""
    f0 = (X_START_VIS - X_A) / (X_B - X_A)
    k = 1 - f0
    if p <= k:
        return X_START_VIS + (X_B - X_START_VIS) * p / k
    return X_A + (X_START_VIS - X_A) * (p - k) / (1 - k)


def rim_anim(th, lift, gain=1.0):
    """Offset of the gold rim copy toward the sun, and its strength, over the loop."""
    f0 = (X_START_VIS - X_A) / (X_B - X_A)
    k = 1 - f0
    ps = list(np.linspace(0, k, 36)) + list(np.linspace(k, 1, 18))
    kt, tv, ov = [], [], []
    for i, p in enumerate(ps):
        dx = LIGHT_X - ship_x_frac(p if not (i == 36) else min(1, p + 1e-9))
        if i == 36:
            dx = LIGHT_X - X_A
        s = math.tanh(dx / 150)
        d = th['rim_d']
        kt.append(f'{p:.4f}')
        tv.append(f'{d * s:.2f} {-d * (lift + 0.25 * (1 - abs(s))):.2f}')
        ov.append(f'{min(1, gain * (0.5 + 0.5 * math.exp(-(dx / 380) ** 2))):.2f}')
    kt = ';'.join(kt)
    return (f'<animateTransform attributeName="transform" type="translate" dur="{LOOP}s" repeatCount="indefinite" '
            f'keyTimes="{kt}" values="{";".join(tv)}"/>'
            f'<animate attributeName="opacity" dur="{LOOP}s" repeatCount="indefinite" keyTimes="{kt}" '
            f'values="{";".join(ov)}"/>')


def sail_path():
    sx0, sx1 = MAST_X - SAIL_HALF, MAST_X + SAIL_HALF
    top_l, top_r = (sx0 - 1, YARD_Y + 1.2), (sx1 + 1, YARD_Y + 1.2)
    bot_l, bot_r = (sx0 + 3, SAIL_BOT), (sx1 + 1, SAIL_BOT + 1)
    return (f'M{f(top_l[0])} {f(top_l[1])}L{f(top_r[0])} {f(top_r[1])}'
            f'C{f(sx1 + 6)} {f(-86)} {f(sx1 + 6)} {f(-52)} {f(bot_r[0])} {f(bot_r[1])}'
            f'Q{f(MAST_X + 2)} {f(SAIL_BOT - 9)} {f(bot_l[0])} {f(bot_l[1])}'
            f'C{f(sx0 + 1)} {f(-56)} {f(sx0 - 3)} {f(-86)} {f(top_l[0])} {f(top_l[1])}Z'), bot_l, bot_r


def ship_svg(th, uid):
    S = SHIP
    p = []
    ink = th['ship']
    sx0, sx1 = MAST_X - SAIL_HALF, MAST_X + SAIL_HALF
    sail, bot_l, bot_r = sail_path()
    mast = (f'M{f(MAST_X - 1.4)} {f(S["sheer_at"](MAST_X))}L{f(MAST_X - 0.9)} {f(MAST_TOP)}'
            f'L{f(MAST_X + 0.9)} {f(MAST_TOP)}L{f(MAST_X + 1.4)} {f(S["sheer_at"](MAST_X))}Z')
    yard = f'M{f(sx0 - 6)} {f(YARD_Y + .6)}Q{f(MAST_X)} {f(YARD_Y - 1.5)} {f(sx1 + 6)} {f(YARD_Y + .6)}'
    hull = poly(S['hull']) + poly(S['bow_post']) + poly(S['stern_post'])
    rim = th['rim_light']

    # rim light, rig level: gold copies of the sail, mast and yard, offset toward the sun
    p.append(f'<g fill="{rim}">{rim_anim(th, 0.35, th["rim_gain"])}'
             f'<path d="{sail}{mast}"/>'
             f'<path d="{yard}" fill="none" stroke="{rim}" stroke-width="2.2" stroke-linecap="round"/></g>')
    # rigging behind the sail
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
    p.append(f'<path d="{mast}" fill="{ink}"/>')
    # sail, vertical cloth stripes
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
    # the sun seen through the wool: a world-fixed glow, clipped to the sail
    f0 = (X_START_VIS - X_A) / (X_B - X_A)
    k = 1 - f0
    inv = (f'<animateTransform attributeName="transform" type="translate" dur="{LOOP}s" '
           f'repeatCount="indefinite" calcMode="linear" '
           f'values="{f(-X_START_VIS)} {f(-YW)};{f(-X_B)} {f(-YW)};{f(-X_A)} {f(-YW)};{f(-X_START_VIS)} {f(-YW)}" '
           f'keyTimes="0;{k:.5f};{k:.5f};1"/>')
    p.append(f'<g clip-path="url(#sailclip{uid})"><g>{inv}<g>{sun_drift_anim()}'
             f'<ellipse cx="{f(LIGHT_X)}" cy="{f(LIGHT_Y)}" rx="64" ry="58" fill="url(#sailglow{uid})"/></g></g></g>')
    seams = ''.join(f'M{f(sx0 - 4)} {f(y)}L{f(sx1 + 6)} {f(y + 0.6)}' for y in (-90, -72, -54))
    p.append(f'<path clip-path="url(#sailclip{uid})" d="{seams}" stroke="{th["seam"]}" '
             f'stroke-width=".5" fill="none" opacity=".45"/>')
    p.append(f'<path d="{sail}" fill="none" stroke="{th["sail_edge"]}" stroke-width=".7"/>')
    p.append(f'<path d="{yard}" stroke="{ink}" stroke-width="2.2" stroke-linecap="round" fill="none"/>')
    # sheets from the clews to the hull
    p.append(f'<path d="M{f(bot_l[0])} {f(bot_l[1])}L{f(sx0 - 22)} {f(S["sheer_at"](sx0 - 22))}'
             f'M{f(bot_r[0])} {f(bot_r[1])}L{f(sx1 + 20)} {f(S["sheer_at"](sx1 + 20))}" '
             f'stroke="{th["rig"]}" stroke-width="{stay_w}" fill="none"/>')
    # rim light, hull level (sun is higher than the hull: rims sit on top edges)
    p.append(f'<g fill="{rim}">{rim_anim(th, 0.75, th["rim_gain"])}<path d="{hull}"/></g>')
    # hull
    p.append(f'<path d="{hull}" fill="{ink}"/>')
    # clinker strakes
    strakes = []
    xs, sy = S['xs'], S['sheer'][:, 1]
    for fr in (0.36, 0.62, 0.84):
        y = sy * (1 - fr) + 1.5 * fr
        m = (y < 0.5)
        pts = np.stack([xs[m], y[m]], 1)
        strakes.append(poly(pts, False))
    p.append(f'<clipPath id="hullclip{uid}"><path d="{poly(S["hull"])}"/></clipPath>')
    p.append(f'<path clip-path="url(#hullclip{uid})" d="{"".join(strakes)}" fill="none" stroke="{th["strake"]}" stroke-width=".55"/>')
    # shields along the rail, each with its own sun-side rim
    sh, shr = [], []
    for i, x in enumerate(SHIELD_X):
        y = S['sheer_at'](x) + 0.6
        fill = th['shield_a'] if i % 2 == 0 else th['shield_b']
        sh.append(f'<circle cx="{f(x)}" cy="{f(y)}" r="3.4" fill="{fill}"/>')
        shr.append(f'<circle cx="{f(x)}" cy="{f(y)}" r="3.4"/>')
    p.append(f'<g fill="{rim}">{rim_anim(th, 0.75, th["rim_gain"] * .9)}{"".join(shr)}</g>')
    p.append(f'<g stroke="{th["shield_rim"]}" stroke-width=".45">{"".join(sh)}</g>')
    p.append('<g fill="' + th['boss'] + '">' + ''.join(
        f'<circle cx="{f(x)}" cy="{f(S["sheer_at"](x) + .6)}" r=".9"/>' for x in SHIELD_X) + '</g>')
    # oars
    p.append(f'<g>{oar_uses(S, uid)}</g>')
    return ''.join(p)


def sun_drift_anim():
    """The midnight sun creeping along the horizon (there and back over two crossings)."""
    return (f'<animateTransform attributeName="transform" type="translate" dur="{2 * LOOP:.1f}s" '
            f'repeatCount="indefinite" calcMode="spline" keyTimes="0;.5;1" '
            f'keySplines=".45 0 .55 1;.45 0 .55 1" '
            f'values="{-SUN_DRIFT} 0;{SUN_DRIFT} 0;{-SUN_DRIFT} 0"/>')


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


# ------------------------------------------------------------------ scene
def theme_svg(th, uid):
    xs = np.arange(-4, W + 5, 2.5)
    parts = []
    defs = []

    def stops(lst):
        return ''.join(f'<stop offset="{o}" stop-color="{c}"/>' for o, c in lst)

    # gradients
    defs.append(f'<linearGradient id="sky{uid}" x1="0" y1="0" x2="0" y2="{YH}" gradientUnits="userSpaceOnUse">'
                + stops(th['sky']) + '</linearGradient>')
    # wide, flattened horizon glow + a tighter warm core
    defs.append(f'<radialGradient id="glow{uid}" cx="{LIGHT_X}" cy="{f(YH - 18)}" r="520" gradientUnits="userSpaceOnUse" '
                f'gradientTransform="translate({LIGHT_X} {f(YH - 18)}) scale(1.35 .36) translate({-LIGHT_X} {f(-(YH - 18))})">'
                f'<stop offset="0" stop-color="{th["glow"]}" stop-opacity="{th["glow_op"]}"/>'
                f'<stop offset=".4" stop-color="{th["glow"]}" stop-opacity="{th["glow_op"] * .4:.3f}"/>'
                f'<stop offset="1" stop-color="{th["glow"]}" stop-opacity="0"/></radialGradient>')
    defs.append(f'<radialGradient id="core{uid}" cx="{LIGHT_X}" cy="{f(LIGHT_Y)}" r="190" gradientUnits="userSpaceOnUse" '
                f'gradientTransform="translate({LIGHT_X} {f(LIGHT_Y)}) scale(1.5 .62) translate({-LIGHT_X} {f(-LIGHT_Y)})">'
                f'<stop offset="0" stop-color="{th["core"]}" stop-opacity="{th["core_op"]}"/>'
                f'<stop offset=".3" stop-color="{th["core"]}" stop-opacity="{th["core_op"] * .35:.3f}"/>'
                f'<stop offset="1" stop-color="{th["core"]}" stop-opacity="0"/></radialGradient>')
    defs.append(f'<radialGradient id="halo{uid}" cx="{LIGHT_X}" cy="{f(LIGHT_Y)}" r="42" gradientUnits="userSpaceOnUse">'
                f'<stop offset=".3" stop-color="{th["halo"]}" stop-opacity="{th["halo_op"]}"/>'
                f'<stop offset="1" stop-color="{th["halo"]}" stop-opacity="0"/></radialGradient>')
    defs.append(f'<radialGradient id="disc{uid}" cx="{LIGHT_X}" cy="{f(LIGHT_Y)}" r="{LIGHT_R}" gradientUnits="userSpaceOnUse">'
                + stops(th['disc']) + '</radialGradient>')
    defs.append(f'<linearGradient id="water{uid}" x1="0" y1="{YH}" x2="0" y2="{H}" gradientUnits="userSpaceOnUse">'
                + stops(th['water']) + '</linearGradient>')
    defs.append(f'<linearGradient id="sailg{uid}" x1="0" y1="{YARD_Y}" x2="0" y2="{SAIL_BOT}" gradientUnits="userSpaceOnUse">'
                f'<stop offset="0" stop-color="{th["sail_top"]}"/><stop offset="1" stop-color="{th["sail_bot"]}"/></linearGradient>')
    defs.append(f'<radialGradient id="sailglow{uid}" cx="{LIGHT_X}" cy="{f(LIGHT_Y)}" r="64" gradientUnits="userSpaceOnUse">'
                + ''.join(f'<stop offset="{o}" stop-color="{c}" stop-opacity="{a}"/>' for o, c, a in th['sailglow'])
                + '</radialGradient>')
    # ridge rim light: strongest near the sun
    defs.append(f'<radialGradient id="rimg{uid}" cx="{LIGHT_X}" cy="{f(LIGHT_Y)}" r="620" gradientUnits="userSpaceOnUse" '
                f'gradientTransform="translate({LIGHT_X} {f(LIGHT_Y)}) scale(1 .8) translate({-LIGHT_X} {f(-LIGHT_Y)})">'
                f'<stop offset="0" stop-color="{th["ridge"]}"/>'
                f'<stop offset=".25" stop-color="{th["ridge"]}" stop-opacity=".7"/>'
                f'<stop offset="1" stop-color="{th["ridge"]}" stop-opacity=".18"/></radialGradient>')
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

    # sky, glow, haze strands, sun
    parts.append(f'<rect width="{W}" height="{YH + 2}" fill="url(#sky{uid})"/>')
    parts.append(f'<rect width="{W}" height="{YH + 2}" fill="url(#glow{uid})"/>')
    parts.append(f'<g>{sun_drift_anim()}<rect width="{W}" height="{YH + 2}" fill="url(#core{uid})"/></g>')
    parts.append(sky_haze(th, uid))
    parts.append(f'<g>{sun_drift_anim()}'
                 f'<circle cx="{LIGHT_X}" cy="{f(LIGHT_Y)}" r="42" fill="url(#halo{uid})"/>'
                 f'<ellipse cx="{LIGHT_X}" cy="{f(LIGHT_Y + 0.4)}" rx="{LIGHT_R}" ry="{LIGHT_R * 0.94:.2f}" fill="url(#disc{uid})"/>'
                 f'</g>')

    # mountains: aerial perspective, mist at their feet, rim light on ridges
    profiles = []
    for i, L in enumerate(LAYERS):
        y = layer_profile(L, xs)
        profiles.append(y)
        col = th['mtn'][i]
        mist = th['mist']
        gid = f'm{uid}{i}'
        topy = float(y.min())
        defs.append(f'<linearGradient id="{gid}" x1="0" y1="{f(topy)}" x2="0" y2="{YH}" gradientUnits="userSpaceOnUse">'
                    f'<stop offset="0" stop-color="{col}"/>'
                    f'<stop offset=".5" stop-color="{col}"/>'
                    f'<stop offset="1" stop-color="{mix(col, mist, th["mist_mix"][i])}"/></linearGradient>')
        parts.append(f'<path d="{layer_path(y, xs)}" fill="url(#{gid})"/>')
        if th['ridge_op'][i] > 0:
            m = y < YH - 3
            seg, run = [], []
            for xx, yy, mm in zip(xs, y, m):
                if mm:
                    run.append((xx, yy))
                elif run:
                    seg.append(run)
                    run = []
            if run:
                seg.append(run)
            dd = ''.join(poly(r, False) for r in seg if len(r) > 2)
            parts.append(f'<path d="{dd}" fill="none" stroke="url(#rimg{uid})" stroke-width="{th["ridge_w"][i]}" '
                         f'stroke-linejoin="round" opacity="{th["ridge_op"][i]}"/>')
        for wl, wx in WATERFALLS:
            if wl == i:
                parts.append(waterfall_svg(th, wx, y, xs))
        if i in th['mist_after']:
            parts.append(mist_band(th, uid, i))
    # veiling glare: scattered light in front of every wall
    defs.append(f'<radialGradient id="veil{uid}" cx="{LIGHT_X}" cy="{f(YH - 20)}" r="460" gradientUnits="userSpaceOnUse" '
                f'gradientTransform="translate({LIGHT_X} {f(YH - 20)}) scale(1.3 .5) translate({-LIGHT_X} {f(-(YH - 20))})">'
                f'<stop offset="0" stop-color="{th["glow"]}" stop-opacity="{th["veil_op"]}"/>'
                f'<stop offset="1" stop-color="{th["glow"]}" stop-opacity="0"/></radialGradient>')
    parts.append(f'<rect width="{W}" height="{YH}" fill="url(#veil{uid})"/>')

    # water
    parts.append(f'<rect y="{YH}" width="{W}" height="{H - YH}" fill="url(#water{uid})"/>')
    # mirrored mountains
    rg = []
    for i, (L, y) in enumerate(zip(LAYERS, profiles)):
        yr = 2 * YH - y
        col = mix(th['mtn'][i], th['water_mid'], th['refl_mix'])
        gid = f'r{uid}{i}'
        defs.append(f'<linearGradient id="{gid}" x1="0" y1="{YH}" x2="0" y2="{f(min(float(yr.max()), H + 120))}" gradientUnits="userSpaceOnUse">'
                    f'<stop offset="0" stop-color="{mix(col, th["mist"], th["mist_mix"][i] * .8)}"/>'
                    f'<stop offset=".5" stop-color="{col}"/>'
                    f'<stop offset="1" stop-color="{col}"/></linearGradient>')
        pts = [(xs[0], YH - 1)] + list(zip(xs, yr)) + [(xs[-1], YH - 1)]
        rg.append(f'<path d="{poly(pts)}" fill="url(#{gid})"/>')
    parts.append(f'<g opacity="{th["refl_op"]}">{"".join(rg)}</g>')
    # mist hugging the water line
    defs.append(f'<linearGradient id="shoremist{uid}" x1="0" y1="{YH - 10}" x2="0" y2="{YH + 12}" gradientUnits="userSpaceOnUse">'
                f'<stop offset="0" stop-color="{th["mist"]}" stop-opacity="0"/>'
                f'<stop offset=".5" stop-color="{th["mist"]}" stop-opacity="{th["shore_op"]}"/>'
                f'<stop offset="1" stop-color="{th["mist"]}" stop-opacity="0"/></linearGradient>')
    parts.append(f'<rect y="{YH - 10}" width="{W}" height="22" fill="url(#shoremist{uid})"/>')
    parts.append(mist_band(th, uid, 'fg'))
    # sun path
    parts.append(glade_svg(th, uid, defs))
    parts.append(ripples_svg(th))

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
    return ''.join(defs), ''.join(parts)


def sky_haze(th, uid):
    """Thin strands of high haze drifting across the glow.

    The strands are moving clip shapes over a static, sun-centred gradient, so
    a strand brightens as it drifts toward the sun and fades away from it."""
    shapes = []
    for j, (yc, thick, per, dur, seed) in enumerate(th['haze']):
        xs = np.arange(0, W + per + 1, 8.0)
        r = np.random.default_rng(seed)
        ph = r.uniform(0, 2 * np.pi, 4)
        u = 2 * np.pi * xs / per
        mid = yc + 3.0 * np.sin(u + ph[0]) + 1.2 * np.sin(3 * u + ph[1])
        w = thick * np.clip(0.5 + 0.5 * np.sin(2 * u + ph[2]), 0, 1) ** 1.5 * (0.75 + 0.25 * np.sin(5 * u + ph[3]))
        pts = list(zip(xs, mid - w / 2)) + list(zip(xs[::-1], (mid + w / 2)[::-1]))
        shapes.append(f'<path d="{poly(pts)}">'
                      f'<animateTransform attributeName="transform" type="translate" values="0 0;{-per} 0" '
                      f'dur="{dur}s" repeatCount="indefinite"/></path>')
    cy = float(np.mean([h[0] for h in th['haze']]))
    return (f'<defs><clipPath id="hzc{uid}">{"".join(shapes)}</clipPath>'
            f'<radialGradient id="hzg{uid}" cx="{LIGHT_X}" cy="{f(cy)}" r="560" gradientUnits="userSpaceOnUse" '
            f'gradientTransform="translate({LIGHT_X} {f(cy)}) scale(1 .25) translate({-LIGHT_X} {f(-cy)})">'
            f'<stop offset="0" stop-color="{th["haze_c"]}" stop-opacity="{th["haze_op"]}"/>'
            f'<stop offset=".45" stop-color="{th["haze_c"]}" stop-opacity="{th["haze_op"] * .45:.3f}"/>'
            f'<stop offset="1" stop-color="{th["haze_c"]}" stop-opacity="{th["haze_op"] * .12:.3f}"/></radialGradient></defs>'
            f'<rect width="{W}" height="{YH}" fill="url(#hzg{uid})" clip-path="url(#hzc{uid})"/>')


WATERFALLS = [(3, 896.0)]


def waterfall_svg(th, x0, prof, xs):
    """A thin thread falling from the fjell edge, fading into the mist."""
    ytop = float(np.interp(x0, xs, prof)) + 1.0
    ybot = ytop + 0.62 * (YH - ytop)
    n = 30
    ys = np.linspace(ytop, ybot, n)
    xx = x0 + 1.4 * (1 - np.exp(-(ys - ytop) / 5)) + 0.35 * np.sin(ys / 7)
    d = poly(list(zip(xx, ys)), False)
    gid = f'fall{int(x0)}{th["fall"][1:]}'
    return (f'<defs><linearGradient id="{gid}" x1="0" y1="{f(ytop)}" x2="0" y2="{f(ybot)}" gradientUnits="userSpaceOnUse">'
            f'<stop offset="0" stop-color="{th["fall"]}" stop-opacity="{th["fall_op"]}"/>'
            f'<stop offset=".55" stop-color="{th["fall"]}" stop-opacity="{th["fall_op"] * .6:.2f}"/>'
            f'<stop offset="1" stop-color="{th["fall"]}" stop-opacity="0"/></linearGradient></defs>'
            f'<g fill="none" stroke="url(#{gid})" stroke-linecap="round">'
            f'<path d="{d}" stroke-width=".8" opacity=".55"/>'
            f'<path d="{d}" stroke-width=".9" stroke-dasharray="5 4 2 5" opacity=".9">'
            f'<animate attributeName="stroke-dashoffset" values="0;-16" dur="1.1s" repeatCount="indefinite"/></path>'
            f'</g>')


MIST = {
    0: dict(y=206, h=30, per=520, amp=7, dur=260, op=.55, seed=2),
    1: dict(y=210, h=26, per=640, amp=8, dur=330, op=.45, seed=4),
    2: dict(y=214, h=24, per=760, amp=6, dur=300, op=.40, seed=6),
    3: dict(y=216, h=22, per=600, amp=6, dur=240, op=.35, seed=9),
    4: dict(y=219, h=20, per=700, amp=5, dur=280, op=.30, seed=10),
    'fg': dict(y=226, h=16, per=900, amp=4, dur=420, op=.25, seed=12),
}


def mist_band(th, uid, key):
    m = MIST[key]
    per = m['per']
    xs = np.arange(0, W + per + 1, 6.0)
    r = np.random.default_rng(m['seed'])
    ph = r.uniform(0, 2 * np.pi, 3)
    top = m['y'] - m['amp'] * (0.55 * np.sin(2 * np.pi * xs / per + ph[0])
                               + 0.3 * np.sin(4 * np.pi * xs / per + ph[1])
                               + 0.15 * np.sin(6 * np.pi * xs / per + ph[2]))
    bot = m['y'] + m['h'] + 0.4 * m['amp'] * np.sin(2 * np.pi * xs / per + ph[1])
    pts = list(zip(xs, top)) + list(zip(xs[::-1], bot[::-1]))
    gid = f'mist{uid}{key}'
    g = (f'<linearGradient id="{gid}" x1="0" y1="0" x2="0" y2="1">'
         f'<stop offset="0" stop-color="{th["mist"]}" stop-opacity="0"/>'
         f'<stop offset=".45" stop-color="{th["mist"]}" stop-opacity="{m["op"] * th["mist_op"]:.3f}"/>'
         f'<stop offset="1" stop-color="{th["mist"]}" stop-opacity="0"/></linearGradient>')
    return (f'<defs>{g}</defs><path d="{poly(pts)}" fill="url(#{gid})">'
            f'<animateTransform attributeName="transform" type="translate" values="0 0;{-per} 0" '
            f'dur="{m["dur"]}s" repeatCount="indefinite"/></path>')


def glade_svg(th, uid, defs):
    """The sun path: a long, narrow road of glints from the horizon to the
    foreground. Its width grows with nearness (constant angular spread of wave
    slopes); glints near the horizon are short and dense, near ones long."""
    r = np.random.default_rng(17)
    NG = 7
    groups = [[] for _ in range(NG)]
    y = YH + 1.2
    gap = 1.5
    while y < H:
        depth = (y - YH) / (H - YH)
        halfw = 5 + 82 * depth ** 1.05
        n = int(3 + 6 * depth ** 0.6)
        for _ in range(n):
            x = LIGHT_X + r.normal(0, 0.42) * halfw
            ln = r.uniform(2.5, 8) * (0.5 + 1.6 * depth)
            cen = math.exp(-((x - LIGHT_X) / (halfw * 0.7)) ** 2)
            op = (0.3 + 0.7 * cen) * (1 - 0.5 * depth)
            groups[r.integers(0, NG)].append(
                f'<path d="M{f(x - ln / 2)} {f(y)}h{f(ln)}" stroke-opacity="{op:.2f}"/>')
        y += gap
        gap *= 1.07
    out = [f'<g stroke="{th["glade"]}" stroke-width="{th["glade_w"]}" stroke-linecap="round" '
           f'opacity="{th["glade_op"]}">']
    for i, g in enumerate(groups):
        dur = 2.3 + 0.61 * i
        vals = ';'.join(f'{0.2 + 0.8 * (0.5 + 0.5 * math.sin(2 * math.pi * (j / 8 + i / NG))) ** 1.6:.2f}' for j in range(9))
        out.append(f'<g>{"".join(g)}<animate attributeName="opacity" values="{vals}" dur="{dur:.2f}s" '
                   f'repeatCount="indefinite"/></g>')
    out.append('</g>')
    # soft column of reflected light, brightest at the horizon
    defs.append(f'<radialGradient id="col{uid}" cx="{LIGHT_X}" cy="{YH}" r="{H - YH + 10}" gradientUnits="userSpaceOnUse" '
                f'gradientTransform="translate({LIGHT_X} {YH}) scale(.5 1) translate({-LIGHT_X} {-YH})">'
                f'<stop offset="0" stop-color="{th["col"]}" stop-opacity="{th["col_op"]}"/>'
                f'<stop offset=".35" stop-color="{th["col"]}" stop-opacity="{th["col_op"] * .4:.3f}"/>'
                f'<stop offset="1" stop-color="{th["col"]}" stop-opacity="0"/></radialGradient>')
    defs.append(f'<radialGradient id="hstreak{uid}" cx="{LIGHT_X}" cy="{YH + 1}" r="90" gradientUnits="userSpaceOnUse" '
                f'gradientTransform="translate({LIGHT_X} {YH + 1}) scale(1 .05) translate({-LIGHT_X} {-YH - 1})">'
                f'<stop offset="0" stop-color="{th["glade"]}" stop-opacity="{th["streak_op"]}"/>'
                f'<stop offset="1" stop-color="{th["glade"]}" stop-opacity="0"/></radialGradient>')
    return (f'<g>{sun_drift_anim()}'
            f'<rect x="{LIGHT_X - 80}" y="{YH}" width="160" height="{H - YH}" fill="url(#col{uid})"/>'
            f'<rect x="{LIGHT_X - 90}" y="{YH - 4}" width="180" height="10" fill="url(#hstreak{uid})"/>'
            + ''.join(out) + '</g>')


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
        op = th['ripple_op'] * (0.5 + 0.5 * depth)
        dur = r.uniform(30, 60)
        d = 1 if r.uniform() < .5 else -1
        out.append(f'<path d="M-10 {f(y)}H{W + 10}" stroke-dasharray="{f(dash)} {f(space)}" stroke-opacity="{op:.2f}">'
                   f'<animate attributeName="stroke-dashoffset" values="0;{f(d * per)}" dur="{dur:.1f}s" '
                   f'repeatCount="indefinite"/></path>')
        y += gap
        gap *= 1.12
    out.append('</g>')
    return ''.join(out)


def ramp(a, b, n, g):
    return [mix(a, b, (i / (n - 1)) ** g) for i in range(n)]


THEMES = {
    'dark': dict(
        bg='#0d1117', border='#30363d',
        sky=[(0, '#12151f'), (.35, '#1f2133'), (.62, '#3d2d40'), (.82, '#7a4b52'), (.94, '#b07258'), (1, '#c88d63')],
        glow='#e8a66c', glow_op=.45, core='#ffd29a', core_op=.55,
        halo='#ffe1b0', halo_op=.55, disc=[(0, '#fffaf0'), (.6, '#fff0cf'), (1, '#ffd99a')],
        haze=[(128, 3.2, 700, 380, 1), (146, 2.4, 560, 300, 2), (161, 1.8, 880, 460, 3)],
        haze_c='#f0b58a', haze_op=.38,
        mtn=['#9099b6', '#4f5168', '#433f55', '#30293a', '#251d27', '#1c151b', '#151013'],
        mist='#a8706a', mist_mix=[.35, .45, .45, .45, .45, .45, .45], mist_op=.6, mist_after=[0, 1, 2, 3, 4],
        ridge='#ffc98e', ridge_op=[.9, .8, .7, .6, .55, .5, .45], ridge_w=[.6, .7, .7, .75, .8, .8, .85],
        veil_op=.08,
        water=[(0, '#b77d5f'), (.12, '#7a4f55'), (.45, '#33263a'), (1, '#15141f')], water_mid='#3a2a3c',
        refl_mix=.4, refl_op=.8, shore_op=.22,
        glade='#ffd9a0', col='#ffc98a', glade_w=1.0, glade_op=.95, col_op=.35, streak_op=.8,
        ripple='#c48a6c', ripple_op=.22,
        ship='#0e0a0c', rig='#3b2e2e', rig_w=.55, strake='#241a1c',
        rim_light='#ffc47e', rim_d=1.0, rim_gain=.95,
        shield_a='#1a1214', shield_b='#2c2022', shield_rim='#3e2e2c', boss='#8a6a52',
        oar='#0e0a0c',
        sail_top='#2e191b', sail_bot='#22141a', stripe='#5e2c22', stripe_op=.8, seam='#000000',
        sail_edge='#5a3428',
        sailglow=[(0, '#fff1c8', .95), (.18, '#ffc777', .75), (.45, '#e0703f', .35), (1, '#b04030', 0)],
        wake='#e0a57a', refl_ship='#0c090b', refl_sail='#26161a', ship_refl_op=.85, gap=1.0,
        fall='#d8a58a', fall_op=.45,
    ),
    'light': dict(
        bg='#ffffff', border='#d0d7de',
        sky=[(0, '#e3e9f0'), (.45, '#ece9e8'), (.75, '#f6e6d6'), (.92, '#f9dcbc'), (1, '#f7d1a8')],
        glow='#f7c58c', glow_op=.5, core='#ffd49c', core_op=.75,
        halo='#ffe8c4', halo_op=.9, disc=[(0, '#fffefa'), (.75, '#fff6e0'), (1, '#ffe2b0')],
        haze=[(128, 3.2, 700, 380, 1), (146, 2.4, 560, 300, 2), (161, 1.8, 880, 460, 3)],
        haze_c='#fff3e2', haze_op=.6,
        mtn=['#bdc7d8', '#a6aec0', '#9496a6', '#85818c', '#77707a', '#6a6066', '#5c504f'],
        mist='#f7e1cb', mist_mix=[.45, .5, .5, .5, .5, .5, .5], mist_op=.7, mist_after=[0, 1, 2, 3, 4],
        ridge='#fff0d0', ridge_op=[.8, .75, .7, .65, .6, .55, .5], ridge_w=[.7, .8, .8, .85, .9, .9, .95],
        veil_op=.2,
        water=[(0, '#f3dcc4'), (.15, '#e5d9d2'), (.5, '#d9d8de'), (1, '#cfd4de')], water_mid='#dcd8dc',
        refl_mix=.42, refl_op=.75, shore_op=.4,
        glade='#f4b46a', col='#f8cf9c', glade_w=1.15, glade_op=1.0, col_op=.6, streak_op=1.0,
        ripple='#a08a80', ripple_op=.25,
        ship='#2a1f1f', rig='#5a4a48', rig_w=.6, strake='#3e3232',
        rim_light='#ffb866', rim_d=1.0, rim_gain=.9,
        shield_a='#3a2c2c', shield_b='#64544f', shield_rim='#2a1f1f', boss='#d8c2a8',
        oar='#2a1f1f',
        sail_top='#9a5c48', sail_bot='#8a5040', stripe='#efe2d0', stripe_op=.6, seam='#3a2a25',
        sail_edge='#5e3a2e',
        sailglow=[(0, '#fff8e0', .95), (.2, '#ffd79a', .7), (.5, '#f0a060', .3), (1, '#e08050', 0)],
        wake='#8a6a5c', refl_ship='#4a3c3c', refl_sail='#a06e5e', ship_refl_op=.5, gap=1.0,
        fall='#ffffff', fall_op=.85,
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
