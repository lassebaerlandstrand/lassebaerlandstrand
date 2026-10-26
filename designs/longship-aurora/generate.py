#!/usr/bin/env python3
"""Animated fjord + Viking longship banner, under the northern lights.

A variant of designs/longship: the same fjord and ship, at night, lit by a calm
aurora instead of the moon.

Scene (back to front)
---------------------
night sky with stars; an auroral curtain hanging low over the fjord's mouth (a
broad east-west arc with folds, a sharp bright lower rim, a diffuse sheet with
fine field-aligned rays, green fading to violet with height); seven procedurally
generated fjord-wall silhouettes (glacial U-valley profile: steep cliffs, fjell
plateaus, fractal roughness) receding into mist, their ridges rim-lit by the
aurora; the water, which mirrors the aurora about the far shore line (softened,
Fresnel-veiled with depth, broken by faint swell) and the mountains in front of
it; slow ripple lines; the longship's reflection (horizontal strips that sway
independently), the wake, and the ship.

The ship is drawn from hand-designed parametric curves at Gokstad/Oseberg
proportions (23 m hull, low freeboard amidships, stems rising to ~5 m and
ending in an Oseberg-style spiral, striped square sail, 21 shields on the rail,
14 oars a side; clinker strakes, stays, shrouds and sheets as hairlines).

Motion (all SMIL, seamless; the aurora only moves bitmaps by transform/opacity):
* the ship crosses the fjord once per LOOP seconds and re-enters off-screen;
  it starts mid-frame so a fresh page load already shows it;
* the oars row in unison (catch, drive, release, feathered recovery); the
  stroke speed matches the ship speed so blades "grip" the water;
* each catch leaves a puddle ring that stays put in the world while the ship
  moves on; a Kelvin wake (featherlets on both arms) and a stern trail follow;
* the aurora sways, its rays lengthen and shorten (stretch anchored on the
  lower rim), its light breathes (and with it the rim light on the ridges and
  its reflection), and two ray sets drift along the curtain while cross-fading
  half a cycle apart, so rays come and go without popping;
* the reflection's strips sway, mist drifts, a waterfall runs down one cliff.

All soft shapes (curtain, rays, the softened reflection) are pre-rendered with
numpy into small PNGs embedded as data URIs; no filters at all.

Usage: python generate.py  -> banner-dark.svg, banner-light.svg
"""
import base64
import math
import os
import struct
import zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

W, H = 1200, 340
YH = 226.0          # horizon (water line of the distant shore)
YW = 276.0          # ship's water line

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


def rel_poly(pts):
    """Closed polyline with relative coordinates (rounded absolutes first, so no drift)."""
    q = [(round(x * 10), round(y * 10)) for x, y in pts]

    def n(v):
        s = f'{v / 10:.1f}'.rstrip('0').rstrip('.')
        s = s.replace('0.', '.', 1) if s.startswith('0.') else s.replace('-0.', '-.', 1)
        return s or '0'
    out = [f'M{n(q[0][0])} {n(q[0][1])}l']
    prev = q[0]
    segs = []
    for x, y in q[1:]:
        a, b = n(x - prev[0]), n(y - prev[1])
        segs.append(a + ('' if b.startswith('-') else ' ') + b)
        prev = (x, y)
    return out[0] + ' '.join(segs) + 'Z'


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
    return rel_poly(keep)


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


OAR_DIP = 1.9          # how far the blade sinks below the surface during the drive


def oar_frames(n=28):
    """(t, tip_dx, tip_dy, u_wet) keyframes for one stroke, relative to the oar port.

    u_wet is the fraction along the oar where it pierces the water surface (>= 1
    when the oar is clear of the water). The surface under a point sits lower on
    screen the further it is toward the viewer, so the crossing is found against
    that perspective water plane, not the hull's water line."""
    drive = 0.45
    a0, a1 = 30.0, -25.0
    out = []
    for i in range(n + 1):
        p = i / n
        if p <= drive:
            s = p / drive
            dip = OAR_DIP * math.sin(math.pi * s) ** 0.5
            s = 0.5 - 0.5 * math.cos(math.pi * s)
            th = math.radians(a0 + (a1 - a0) * s)
            lift = -dip
        else:
            s = (p - drive) / (1 - drive)
            e = 0.5 - 0.5 * math.cos(math.pi * s)
            th = math.radians(a1 + (a0 - a1) * e)
            lift = 8.0 * math.sin(math.pi * s) ** 0.6
        dx = OAR_R * math.sin(th)
        depth = OAR_R * math.cos(th)
        dy = 5.0 + K_PERSP * depth - lift      # port sits 5px above water
        # height above the local surface goes linearly from 5 (port) to lift (tip)
        u_wet = 5.0 / (5.0 - lift) if lift < 0 else 2.0
        out.append((p, dx, dy, u_wet))
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


BLADE_U = 0.72         # the blade runs from here to the tip


def oar_defs(th, uid):
    """Each oar is split where it enters the water: the dry part is solid, the
    submerged part faint, as if seen through the water. When a part has nothing
    to draw it collapses onto a point inside the dry part, where it can't show."""
    fr = oar_frames()
    kt = ';'.join(f'{p:.4f}' for p, *_ in fr)

    def seg(dx, dy, u0, u1):
        return f'M{dx * u0:.2f} {dy * u0:.2f}L{dx * u1:.2f} {dy * u1:.2f}'

    shaft_dry, shaft_wet, blade_dry, blade_wet = [], [], [], []
    for _, dx, dy, uw in fr:
        u = min(uw, 1.0)
        shaft_dry.append(seg(dx, dy, 0, u))
        shaft_wet.append(seg(dx, dy, u, 1) if uw < 1 else seg(dx, dy, .5, .5))
        ub = max(u, BLADE_U)
        blade_dry.append(seg(dx, dy, BLADE_U, ub) if ub > BLADE_U else seg(dx, dy, .5, .5))
        blade_wet.append(seg(dx, dy, ub, 1) if uw < 1 else seg(dx, dy, .86, .86))

    def path(vals, width, op=None):
        o = f' opacity="{op}"' if op is not None else ''
        return (f'<path stroke-width="{width}" d="{vals[0]}"{o}>'
                f'<animate attributeName="d" dur="{STROKE}s" repeatCount="indefinite" keyTimes="{kt}" '
                f'values="{";".join(vals)}"/></path>')

    wet = th['oar_wet_op']
    return (f'<g id="oar{uid}" stroke="{th["oar"]}" stroke-linecap="round" fill="none">'
            f'{path(shaft_wet, .85, wet)}{path(blade_wet, 1.9, wet)}'
            f'{path(shaft_dry, .85)}{path(blade_dry, 1.9)}'
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



# ------------------------------------------------------------------ aurora rasters
# Every soft shape of the aurora is rendered here with numpy into a small PNG that is
# embedded as a data URI. The SVG then only moves / fades those images (SMIL
# transform + opacity), so no filter is ever evaluated while it animates.
def png_bytes(rgba):
    """Encode uint8 HxWx4 as PNG, picking the best of the five filters per row."""
    h, w, c = rgba.shape
    a = rgba.astype(np.int16)
    rows = []
    prev = np.zeros((w * c,), np.int16)
    for y in range(h):
        cur = a[y].reshape(-1)
        left = np.concatenate([np.zeros(c, np.int16), cur[:-c]])
        upleft = np.concatenate([np.zeros(c, np.int16), prev[:-c]])
        p = left + prev - upleft
        pa, pb, pc = np.abs(p - left), np.abs(p - prev), np.abs(p - upleft)
        paeth = np.where((pa <= pb) & (pa <= pc), left, np.where(pb <= pc, prev, upleft))
        cands = [cur, cur - left, cur - prev, cur - ((left + prev) >> 1), cur - paeth]
        cands = [(x & 0xFF).astype(np.uint8) for x in cands]
        scores = [np.abs(x.astype(np.int8).astype(np.int32)).sum() for x in cands]
        k = int(np.argmin(scores))
        rows.append(bytes([k]) + cands[k].tobytes())
        prev = cur
    raw = b''.join(rows)

    def chunk(t, d):
        return struct.pack('>I', len(d)) + t + d + struct.pack('>I', zlib.crc32(t + d) & 0xFFFFFFFF)
    ihdr = struct.pack('>IIBBBBB', w, h, 8, 6, 0, 0, 0)
    return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', ihdr)
            + chunk(b'IDAT', zlib.compress(raw, 9)) + chunk(b'IEND', b''))


def data_uri(rgba):
    return 'data:image/png;base64,' + base64.b64encode(png_bytes(rgba)).decode()


def smoothstep(e0, e1, v):
    t = np.clip((v - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def hex_rgb(h):
    h = h.lstrip('#')
    return np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)])


def cramp(stops, t):
    ts = np.array([s[0] for s in stops])
    cs = np.array([hex_rgb(s[1]) for s in stops])
    out = np.empty(t.shape + (3,))
    for ch in range(3):
        out[..., ch] = np.interp(t, ts, cs[:, ch])
    return out


def sines(x, rng, kmin, kmax, span, falloff=1.0):
    """Smooth band-limited noise in [-1, 1] (sum of random sinusoids)."""
    out = np.zeros_like(x, dtype=np.float64)
    for k in range(kmin, kmax + 1):
        out += rng.normal() / k ** falloff * np.sin(2 * np.pi * k * x / span + rng.uniform(0, 2 * np.pi))
    return out / (np.abs(out).max() + 1e-9)


def to_rgba(col, inten, dith=True):
    a = np.clip(inten, 0, 1)
    rgba = np.concatenate([col, a[..., None]], -1)
    ny, nx = inten.shape
    d = ((np.indices((ny, nx)).sum(0) % 2) - 0.5) * 0.6 if dith else 0
    out = np.clip(rgba * 255 + (d[..., None] if dith else 0), 0, 255).astype(np.uint8)
    # colour varies slowly and alpha carries the detail: quantise RGB, zero it where
    # fully transparent (much smaller PNG, invisible difference)
    out[..., :3] = np.minimum(out[..., :3] // 4 * 4 + 2, 255)
    out[out[..., 3] == 0, :3] = 0
    return out


class Curtain:
    """One auroral curtain: a lower edge with slow folds, a soft diffuse sheet above
    it, a hot lower rim, and fine vertical (field-aligned) rays in clusters.
    glow() / rays(seed) / reflection() return (RGBA raster, placement box)."""

    def __init__(self, seed, x0, x1, y0, y1, edge, height, colors, col_h, gain=1.0,
                 sigma=4.5, core=0.4, halo=0.12, diffuse=0.55, haze=0.0, env=None):
        self.seed, self.box = seed, (x0, y0, x1 - x0, y1 - y0)
        self.edge, self.height, self.colors, self.col_h = edge, height, colors, col_h
        self.gain, self.sigma, self.core, self.halo, self.diffuse = gain, sigma, core, halo, diffuse
        self.env, self.haze = env, haze

    @staticmethod
    def _grid(box, sx, sy):
        x0, y0, w, h = box
        nx, ny = int(round(w * sx)), int(round(h * sy))
        return x0 + (np.arange(nx) + 0.5) / sx, y0 + (np.arange(ny) + 0.5) / sy

    def _common(self, x):
        rng = np.random.default_rng(self.seed)
        yb = self.edge(x)
        L = self.height * (1 + 0.3 * sines(x, rng, 1, 6, 1400))
        bright = 0.55 + 0.45 * smoothstep(-0.6, 0.8, sines(x, rng, 1, 5, 1400))
        # where the sheet folds toward us we look along it: brighter
        slope = np.abs(np.gradient(yb, x))
        bright = bright * (1 + 1.6 * np.clip(slope, 0, 0.35))
        if self.env is not None:
            bright = bright * self.env(x)
        return yb, L, bright

    def _top_fade(self, y):
        return smoothstep(self.box[1], self.box[1] + 26, y)[:, None]

    def _glow_i(self, x, y):
        yb, L, bright = self._common(x)
        d = yb[None, :] - y[:, None]                       # > 0 above the edge
        s = self.sigma
        below = np.exp(-(np.minimum(d, 0) / s) ** 2)
        above = np.exp(-np.maximum(d, 0) / L[None, :])
        rim = np.exp(-(d / (s * 1.3)) ** 2) * self.core
        halo = self.halo * np.exp(-np.maximum(-d, 0) / 26) * (d < 0)
        inten = (below * above * self.diffuse + rim * below + halo) * bright[None, :] * self.gain
        # broad scattered light: a wide, faint skirt that turns violet with height
        wide = np.exp(-((x - 640) / 560) ** 2)[None, :]
        inten = inten + self.haze * wide * below * np.exp(-np.maximum(d, 0) / (2.4 * L[None, :]))
        return inten * self._top_fade(y), d

    def _rays_i(self, x, y, rseed, n, strength):
        yb, L, bright = self._common(x)
        r = np.random.default_rng(rseed)
        d = yb[None, :] - y[:, None]
        # clusters: a slow envelope decides where rays are dense
        clus = smoothstep(-0.1, 0.9, sines(x, r, 3, 16, 1400, 0.8)) ** 1.5
        acc = np.zeros_like(d)
        xs = r.uniform(self.box[0], self.box[0] + self.box[2], n * 3)
        keep = r.uniform(0, 1, n * 3) < np.interp(xs, x, 0.06 + 0.94 * clus)
        for xi in xs[keep][:n]:
            wi = r.uniform(0.6, 2.6) * (1 if r.uniform() < 0.8 else 1.8)
            ai = r.uniform(0.25, 1.0) ** 1.5
            li = float(np.interp(xi, x, L)) * r.uniform(0.5, 1.5)
            lo = r.uniform(-4, 6)                          # foot slightly above / below rim
            col = np.exp(-((x - xi) / wi) ** 2)
            m = col > 0.01
            dd = d[:, m] - lo
            prof = smoothstep(-5, 7, dd) * np.exp(-np.maximum(dd, 0) / li)
            acc[:, m] += ai * col[m][None, :] * prof
        return acc * strength * bright[None, :] * self.gain * self._top_fade(y), d

    def ray_list(self, rseed, n, strength):
        """The same rays as _rays_i (identical random draws), as parameters for
        drawing them as vector shapes: (x, half-width, foot y, length, peak alpha)."""
        x = np.arange(self.box[0], self.box[0] + self.box[2]) + 0.5
        yb, L, bright = self._common(x)
        r = np.random.default_rng(rseed)
        clus = smoothstep(-0.1, 0.9, sines(x, r, 3, 16, 1400, 0.8)) ** 1.5
        xs = r.uniform(self.box[0], self.box[0] + self.box[2], n * 3)
        keep = r.uniform(0, 1, n * 3) < np.interp(xs, x, 0.06 + 0.94 * clus)
        out = []
        for xi in xs[keep][:n]:
            wi = r.uniform(0.6, 2.6) * (1 if r.uniform() < 0.8 else 1.8)
            ai = r.uniform(0.25, 1.0) ** 1.5
            li = float(np.interp(xi, x, L)) * r.uniform(0.5, 1.5)
            lo = r.uniform(-4, 6)
            peak = ai * strength * float(np.interp(xi, x, bright)) * self.gain
            out.append((float(xi), wi, float(np.interp(xi, x, yb)) - lo, li, lo, peak))
        return out

    def glow(self, sx=0.3, sy=0.5):
        x, y = self._grid(self.box, sx, sy)
        inten, d = self._glow_i(x, y)
        return to_rgba(cramp(self.colors, np.clip(d / self.col_h, 0, 1)), inten), self.box

    def rays(self, rseed, n=150, strength=0.55, sx=1.0, sy=0.25, ytop=0, ybot=None):
        """Fine rays only (layered over glow()); different seeds give different ray
        sets that are cross-faded in the SVG."""
        box = (self.box[0], ytop, self.box[2], (ybot or self.box[1] + self.box[3]) - ytop)
        x, y = self._grid(box, sx, sy)
        inten, d = self._rays_i(x, y, rseed, n, strength)
        return to_rgba(cramp(self.colors, np.clip(d / self.col_h, 0, 1)), inten), box

    def reflection(self, rseeds, n, strength, ytop, sx=0.25, sy=0.35, blur=(2.0, 3.5)):
        """Glow + the average of the ray sets, softened (mostly vertically, as light is
        smeared by a faint swell) for the mirrored copy in the water."""
        box = (self.box[0], ytop, self.box[2], self.box[1] + self.box[3] - ytop)
        x, y = self._grid(box, sx, sy)
        inten, d = self._glow_i(x, y)
        for rs in rseeds:
            inten = inten + self._rays_i(x, y, rs, n, strength)[0] / len(rseeds)
        for axis, sig in ((1, blur[0] * sx), (0, blur[1] * sy)):
            k = np.exp(-0.5 * (np.arange(-int(3 * sig) - 1, int(3 * sig) + 2) / sig) ** 2)
            k /= k.sum()
            inten = np.apply_along_axis(lambda v: np.convolve(v, k, mode='same'), axis, inten)
        return to_rgba(cramp(self.colors, np.clip(d / self.col_h, 0, 1)), inten), box


def edge_arc(base, sag, cx, half, fold, seed, tilt=0.0):
    """Lower edge of a curtain: a broad arc (sag < 0: highest in the middle, as an
    east-west arc seen from the south) with slow folds and smaller kinks along it."""
    def fn(x):
        rng = np.random.default_rng(seed)
        u = (x - cx) / half
        return (base - sag * u ** 2 + tilt * u + fold * sines(x, rng, 2, 9, 1400, 1.1)
                + 0.45 * fold * sines(x, rng, 10, 18, 1400, 1.0))
    return fn


WATERFALLS = [(3, 1000.0)]   # (wall layer, x): the shallow saddle in that wall's skyline


def waterfall_svg(th, x0, prof, xs):
    """Horsetail veil: a narrow pour off the skyline that very gradually frays into a
    faint, soft-edged veil of spray as it falls (like the Seven Sisters in Geiranger),
    and melts into the mist at the foot. The flanks are smooth and slightly uneven, the
    brightness sits in a core that fades as the water spreads. Motion: soft brightness
    swells sliding slowly down through the veil."""
    col, op = th['fall'], th['fall_op']
    ytop = float(np.interp(x0, xs, prof))
    ybot = YH
    L = ybot - ytop
    ys = np.linspace(ytop, ybot, 90)
    s = (ys - ytop) / L
    rng = np.random.default_rng(int(x0))
    ph = rng.uniform(0, 2 * np.pi, 4)
    cx = (x0 + 1.1 * (np.sin(3.2 * s + ph[0]) - np.sin(ph[0]))
          + .25 * (np.sin(10 * s + ph[1]) - np.sin(ph[1])))
    # half-widths of the veil: a thread at the lip, spreading slowly (faster low down),
    # each flank with its own gentle unevenness
    spread = th['fall_spread'] * s ** 1.7
    hl = .6 + spread * (1 + .12 * np.sin(7 * s + ph[2]))
    hr = .6 + spread * (1 + .12 * np.sin(6 * s + ph[3]))

    def ribbon(kl, kr, add=0.0):
        lx, rx = cx - hl * kl - add, cx + hr * kr + add
        top = [(lx[0], float(np.interp(lx[0], xs, prof))), (rx[0], float(np.interp(rx[0], xs, prof)))]
        return poly([top[0]] + list(zip(lx[1:], ys[1:])) + list(zip(rx[:0:-1], ys[:0:-1])) + [top[1]])

    def grad(gid, stops, c=col, k=op):
        return (f'<linearGradient id="{gid}" x1="0" y1="{f(ytop)}" x2="0" y2="{f(ybot)}" gradientUnits="userSpaceOnUse">'
                + ''.join(f'<stop offset="{o}" stop-color="{c}" stop-opacity="{k * v:.3f}"/>' for o, v in stops)
                + '</linearGradient>')

    u = f'fall{int(x0)}'
    so = np.linspace(0, 1, 21)
    tex = 1 + .22 * np.random.default_rng(int(x0) + 1).uniform(-1, 1, so.size)
    tex[:2] = 1

    def tx(pts):
        return [(f'{o:.3f}', v * t) for o, v, t in
                zip(so, np.interp(so, [p for p, _ in pts], [v for _, v in pts]), tex)]
    # the core thread fades as the water spreads; the veil picks up and then thins out
    core = tx([(0, .8), (.03, 1), (.2, .8), (.5, .5), (.75, .28), (.9, .1), (1, 0)])
    veil = [(0, 0), (.1, .1), (.45, .2), (.75, .22), (.9, .14), (1, 0)]
    per, speed = 40.0, 11.0
    bands = ''.join(f'<rect x="{f(x0 - 20)}" y="{f(ytop - per + k * per)}" width="40" height="{f(per)}"/>'
                    for k in range(int(L / per) + 2))
    out = ['<defs>',
           grad(f'{u}c', core),
           grad(f'{u}v', veil),
           grad(f'{u}m', [(0, .8), (.1, .7), (.5, .6), (.8, .5), (.95, .1), (1, 0)], '#fff', 1),
           f'<linearGradient id="{u}w" x1="0" y1="0" x2="0" y2="1">'
           f'<stop offset="0" stop-color="{col}" stop-opacity="0"/>'
           f'<stop offset=".5" stop-color="{col}" stop-opacity="{op * .4:.3f}"/>'
           f'<stop offset="1" stop-color="{col}" stop-opacity="0"/></linearGradient>',
           f'<radialGradient id="{u}s"><stop offset="0" stop-color="{col}" stop-opacity="{op * .26:.3f}"/>'
           f'<stop offset=".6" stop-color="{col}" stop-opacity="{op * .08:.3f}"/>'
           f'<stop offset="1" stop-color="{col}" stop-opacity="0"/></radialGradient>',
           f'<mask id="{u}k" maskUnits="userSpaceOnUse" x="{f(x0 - 20)}" y="{f(ytop - 1)}" width="40" height="{f(L + 2)}">'
           f'<path d="{ribbon(.75, .75, .4)}" fill="url(#{u}m)"/></mask>',
           '</defs>',
           f'<ellipse cx="{f(cx[-1])}" cy="{f(ybot - 4)}" rx="{f(hl[-1] + hr[-1] + 6)}" ry="5" fill="url(#{u}s)">'
           f'<animate attributeName="opacity" values="1;.6;1" dur="9s" repeatCount="indefinite"/></ellipse>',
           # soft flanks: the veil drawn twice, wide and faint over narrower and denser
           f'<path d="{ribbon(1, 1, .8)}" fill="url(#{u}v)" opacity=".55"/>',
           f'<path d="{ribbon(.6, .6, .3)}" fill="url(#{u}v)"/>',
           f'<path d="{ribbon(.15, .15, .51)}" fill="url(#{u}c)"/>',
           f'<g mask="url(#{u}k)"><g fill="url(#{u}w)">{bands}'
           f'<animateTransform attributeName="transform" type="translate" values="0 0;0 {f(per)}" '
           f'dur="{per / speed:.2f}s" repeatCount="indefinite"/></g></g>']
    return ''.join(out)


def stars_svg(th):
    r = np.random.default_rng(91)
    out = [f'<g fill="{th["star"]}">']
    tw = []
    for i in range(80):
        x, y = r.uniform(20, W - 20), r.uniform(8, 170) ** 1.0
        rad = r.uniform(0.45, 0.95)
        op = r.uniform(0.15, 0.55) * (1 - y / 220)
        if i % 9 == 0:
            dur = r.uniform(5, 9)
            tw.append(f'<circle cx="{f(x)}" cy="{f(y)}" r="{rad:.2f}" opacity="{op:.2f}">'
                      f'<animate attributeName="opacity" values="{op:.2f};{op * .25:.2f};{op:.2f}" dur="{dur:.1f}s" '
                      f'repeatCount="indefinite"/></circle>')
        else:
            out.append(f'<circle cx="{f(x)}" cy="{f(y)}" r="{rad:.2f}" opacity="{op:.2f}"/>')
    return ''.join(out + tw) + '</g>'


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



# ------------------------------------------------------------------ aurora (SVG side)
PULSE = 23.0      # s, slow brightness breathing shared by aurora, rim light and water


def ease(attr, values, dur, typ=None, begin=0.0, kt=None):
    n = len(values.split(';')) - 1
    ks = ';'.join(['0.45 0 0.55 1'] * n)
    kt = kt or ';'.join(f'{i / n:.4g}' for i in range(n + 1))
    t = f' type="{typ}"' if typ else ''
    tag = 'animateTransform' if typ else 'animate'
    b = f' begin="{begin:.2f}s"' if begin else ''
    return (f'<{tag} attributeName="{attr}"{t} values="{values}" dur="{dur}s"{b} '
            f'calcMode="spline" keyTimes="{kt}" keySplines="{ks}" repeatCount="indefinite"/>')


def pulse_vals(lo, hi):
    return f'{lo:.3g};{hi:.3g};{lo:.3g}'


def image(id_, arr_box):
    arr, (x, y, w, h) = arr_box
    return (f'<image id="{id_}" x="{f(x)}" y="{f(y)}" width="{f(w)}" height="{f(h)}" '
            f'preserveAspectRatio="none" href="{data_uri(arr)}"/>')


RAY_LEN_BINS = np.geomspace(30, 160, 9)   # rays share the gradient of the nearest length
RAY_SPAN = 3.5                             # rays end this many lengths above the foot (3% left)


def ray_profile(dd, li):
    """Brightness of a ray at height dd above its foot (as in Curtain._rays_i)."""
    return smoothstep(-5, 7, dd) * np.exp(-np.maximum(dd, 0) / li)


def ray_gradients(th, uid):
    """One vertical gradient per length bin, in bounding-box units: the rect runs from
    5 px below the foot (dark) to RAY_SPAN lengths above it; colour follows height above
    the curtain's edge, alpha the ray profile."""
    out = []
    for i, li in enumerate(RAY_LEN_BINS):
        top, bot = RAY_SPAN * li, -5.0
        stops = []
        for u in np.linspace(0, 1, 9):
            dd = bot + (top - bot) * (1 - u) ** 2.2      # stops packed near the foot
            col = cramp(th['a_cols'], np.clip(np.array([dd + 1.0]) / th['a_colh'], 0, 1))[0]
            hexc = '#%02x%02x%02x' % tuple(int(round(c * 255)) for c in col)
            stops.append((1 - (dd - bot) / (top - bot), hexc, float(ray_profile(np.array(dd), li))))
        stops.sort()
        out.append(f'<defs><linearGradient id="ray{uid}{i}" x1="0" y1="0" x2="0" y2="1">'
                   + ''.join(f'<stop offset="{o:.3f}" stop-color="{c}" stop-opacity="{a:.3f}"/>'
                             for o, c, a in stops)
                   + '</linearGradient></defs>')
    return out


def ray_shapes(rays, th, uid, used):
    """Each ray as one rect as wide as its gaussian cross-section carries light
    (~1.6 sigma), filled with the gradient of its length bin."""
    out = []
    for xi, wi, yf, li, lo, peak in rays:
        if peak < 0.02:
            continue
        i = int(np.argmin(np.abs(np.log(RAY_LEN_BINS / li))))
        used.add(i)
        y0, h = yf - RAY_SPAN * li, RAY_SPAN * li + 5.0
        w = 1.6 * wi
        out.append(f'<rect x="{f(xi - w / 2)}" y="{f(y0)}" width="{f(w)}" height="{f(h)}" '
                   f'fill="url(#ray{uid}{i})" fill-opacity="{min(1, peak):.3f}"/>')
    return ''.join(out)


def band_opacity(ph, k, ns, n=48):
    """Opacity keyframes for ray set k of ns in one band of the curtain: the sets
    take turns (raised-cosine weights 360/ns degrees apart, whose sum is constant),
    under a slow, irregular swell of the whole band."""
    t = np.arange(n + 1) / n
    swell = 0.72 + 0.2 * np.sin(2 * np.pi * t + ph[0]) + 0.08 * np.sin(4 * np.pi * t + ph[1])
    turn = (0.5 + 0.5 * np.cos(2 * np.pi * (2 * t - k / ns) + ph[2])) ** 2
    norm = ns * 3 / 8                                   # constant sum of the ns weights
    return ';'.join(f'{v:.3f}' for v in np.clip(swell * turn / norm * 1.25, 0, 1))


def aurora_svg(th, uid):
    """Returns (sky, reflection) markup. The aurora stays in place: it lives through
    brightness alone. The glow breathes; along the curtain, bands swell and fade on
    their own slow cycles while each band cross-fades between ray sets, so rays come
    and go where they stand. Only a slow vertical stretch moves anything."""
    A = Curtain(th['a_seed'], 0, W, 0, YH, edge_arc(**th['a_edge']), th['a_h'], th['a_cols'], th['a_colh'],
                gain=th['a_gain'], core=th['a_core'], halo=th['a_halo'], diffuse=th['a_diff'],
                haze=th['a_haze'], env=lambda x: 0.15 + 0.85 * np.exp(-((x - 640) / th['a_env']) ** 4))
    NS, NB = 3, 7                      # ray sets, bands along the curtain
    base = th['a_edge']['base']

    def frame(inner):
        # stretches upward and back, anchored on its lower edge
        return (f'<g transform="translate(0 {f(base)})"><g>'
                f'{ease("transform", "1 1;1 1.07;1 1", 31, "scale", begin=-6)}'
                f'<g transform="translate(0 {f(-base)})">{inner}</g></g></g>')

    def img(arr_box, anim, op=None):
        arr, (x, y, w, h) = arr_box
        o = f' opacity="{op}"' if op is not None else ''
        return (f'<image x="{f(x)}" y="{f(y)}" width="{f(w)}" height="{f(h)}"{o} '
                f'preserveAspectRatio="none" href="{data_uri(arr)}">{anim}</image>')

    inner = img(A.glow(), ease('opacity', pulse_vals(.8, 1), PULSE))
    grads, used = ray_gradients(th, uid), set()
    sets = [A.ray_list(100 + k, th['a_nrays'], th['a_rays']) for k in range(NS)]
    rng = np.random.default_rng(th['a_seed'] + 7)
    for b in range(NB):
        x0, x1 = W * b / NB, W * (b + 1) / NB
        dur = rng.uniform(20, 32)
        ph = rng.uniform(0, 2 * np.pi, 3)          # shared by the band's ray sets
        for k in range(NS):
            rays = [r for r in sets[k] if x0 <= r[0] < x1]
            if not rays:
                continue
            vals = band_opacity(ph, k, NS)
            inner += (f'<g opacity="{vals.split(";")[0]}">'
                      f'<animate attributeName="opacity" values="{vals}" dur="{dur:.1f}s" repeatCount="indefinite"/>'
                      f'{ray_shapes(rays, th, uid, used)}</g>')
    inner = ''.join(g for i, g in enumerate(grads) if i in used) + inner
    sky = frame(inner)
    # water: one softened raster (glow + the ray sets, smeared), breathing with the sky
    r = th['aur_refl']
    refl = frame(img(A.reflection([100 + k for k in range(NS)], th['a_nrays'], th['a_rays'],
                                  ytop=th['refl_top']),
                     ease('opacity', pulse_vals(.8 * r, r), PULSE), op=r))
    return sky, refl


def water_streaks(th):
    """Static streaks in the water colour over the aurora's reflection, so it reads as
    calm water with faint swell rather than a perfect mirror."""
    r = np.random.default_rng(29)
    out = [f'<g stroke="{th["streak"]}" fill="none" opacity="{th["streak_k"]}">']
    y = YH + 3.0
    gap = 2.2
    while y < H:
        depth = (y - YH) / (H - YH)
        dash = r.uniform(30, 140) * (0.6 + depth)
        space = r.uniform(10, 60) * (0.6 + depth)
        off = r.uniform(0, dash + space)
        out.append(f'<path d="M0 {f(y)}H{W}" stroke-width="{0.5 + 1.0 * depth:.2f}" '
                   f'stroke-dasharray="{f(dash)} {f(space)}" stroke-dashoffset="{f(off)}"/>')
        y += gap
        gap *= 1.13
    out.append('</g>')
    return ''.join(out)


# ------------------------------------------------------------------ scene
def theme_svg(th, uid):
    xs = np.arange(-4, W + 5, 2.5)
    parts = []
    defs = []

    defs.append(f'<linearGradient id="sky{uid}" x1="0" y1="0" x2="0" y2="{YH}" gradientUnits="userSpaceOnUse">'
                + ''.join(f'<stop offset="{o}" stop-color="{c}"/>' for o, c in th['sky']) + '</linearGradient>')
    defs.append(f'<linearGradient id="water{uid}" x1="0" y1="{YH}" x2="0" y2="{H}" gradientUnits="userSpaceOnUse">'
                + ''.join(f'<stop offset="{o}" stop-color="{c}"/>' for o, c in th['water']) + '</linearGradient>')
    # Fresnel: the still water mirrors the sky strongly at grazing angles near the far
    # shore and less toward the viewer
    defs.append(f'<linearGradient id="veil{uid}" x1="0" y1="{YH}" x2="0" y2="{H}" gradientUnits="userSpaceOnUse">'
                f'<stop offset="0" stop-color="{th["water_mid"]}" stop-opacity="{th["veil"][0]}"/>'
                f'<stop offset="1" stop-color="{th["water_mid"]}" stop-opacity="{th["veil"][1]}"/></linearGradient>')
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
    aur_sky, aur_refl = aurora_svg(th, uid)

    defs.append(f'<clipPath id="sky{uid}c"><rect width="{W}" height="{YH - 3}"/></clipPath>')
    # sky, stars, aurora
    parts.append(f'<rect width="{W}" height="{YH + 2}" fill="url(#sky{uid})"/>')
    if th.get('stars'):
        parts.append(stars_svg(th))
    parts.append(aur_sky)

    # mountains with mist at their feet; rim light from the aurora breathes with it
    profiles = []
    rims = []
    for i, L in enumerate(LAYERS):
        y = layer_profile(L, xs)
        profiles.append(y)
        col = th['mtn'][i]
        mist = th['mist']
        gid = f'm{uid}{i}'
        topy = float(y.min())
        defs.append(f'<linearGradient id="{gid}" x1="0" y1="{f(topy)}" x2="0" y2="{YH}" gradientUnits="userSpaceOnUse">'
                    f'<stop offset="0" stop-color="{col}"/>'
                    f'<stop offset=".55" stop-color="{col}"/>'
                    f'<stop offset="1" stop-color="{mix(col, mist, th["mist_mix"][i])}"/></linearGradient>')
        defs.append(f'<path id="mt{uid}{i}" d="{layer_path(y, xs)}"/>')
        parts.append(f'<use href="#mt{uid}{i}" fill="url(#{gid})"/>')
        if th['ridge_op'][i] > 0:
            # rim light: the same outline, stroked, clipped off the horizon
            op = th['ridge_op'][i]
            parts.append(f'<use href="#mt{uid}{i}" clip-path="url(#sky{uid}c)" fill="none" stroke="{th["ridge"]}" '
                         f'stroke-width=".7" stroke-opacity="{op}">'
                         f'{ease("stroke-opacity", pulse_vals(op * .7, op), PULSE)}</use>')
        for wl, wx in WATERFALLS:
            if wl == i:
                parts.append(waterfall_svg(th, wx, y, xs))
        if i in th['mist_after']:
            parts.append(mist_band(th, uid, i))

    # water: the aurora mirrored about the far shore line, veiled with depth
    parts.append(f'<rect y="{YH}" width="{W}" height="{H - YH}" fill="url(#water{uid})"/>')
    parts.append(f'<g transform="translate(0 {2 * YH}) scale(1 -1)">{aur_refl}</g>')
    parts.append(f'<rect y="{YH}" width="{W}" height="{H - YH}" fill="url(#veil{uid})"/>')
    parts.append(water_streaks(th))
    # mirrored mountains (opaque: they hide the aurora's reflection behind them)
    rg = []
    for i, (L, y) in enumerate(zip(LAYERS, profiles)):
        yr = 2 * YH - y
        col = mix(th['water_mid'], mix(th['mtn'][i], th['water_mid'], th['refl_mix']), th['refl_op'])
        gid = f'r{uid}{i}'
        # gradient in the pre-mirror space of the <use> below
        defs.append(f'<linearGradient id="{gid}" x1="0" y1="{YH}" x2="0" y2="{f(2 * YH - min(float(yr.max()), H + 120))}" gradientUnits="userSpaceOnUse">'
                    f'<stop offset="0" stop-color="{mix(col, th["mist"], th["mist_mix"][i] * .8)}"/>'
                    f'<stop offset=".5" stop-color="{col}"/>'
                    f'<stop offset="1" stop-color="{col}"/></linearGradient>')
        rg.append(f'<use href="#mt{uid}{i}" fill="url(#{gid})"/>')
    parts.append(f'<g transform="matrix(1 0 0 -1 0 {2 * YH})">{"".join(rg)}</g>')
    # mist hugging the water line, lit faintly by the aurora
    defs.append(f'<linearGradient id="shoremist{uid}" x1="0" y1="{YH - 10}" x2="0" y2="{YH + 12}" gradientUnits="userSpaceOnUse">'
                f'<stop offset="0" stop-color="{th["mist"]}" stop-opacity="0"/>'
                f'<stop offset=".5" stop-color="{th["mist"]}" stop-opacity="{th["shore_op"]}"/>'
                f'<stop offset="1" stop-color="{th["mist"]}" stop-opacity="0"/></linearGradient>')
    parts.append(f'<rect y="{YH - 10}" width="{W}" height="22" fill="url(#shoremist{uid})"/>')
    parts.append(mist_band(th, uid, 'fg'))
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

    parts.append(f'<g>{mover_anim()}'
                 f'<g>{wake_svg(th)}</g>'
                 f'<g>{pud_uses}</g>'
                 f'<g>{surge_anim()}{ship_svg(th, uid)}</g>'
                 f'</g>')
    return ''.join(defs), ''.join(parts)


def ramp(a, b, n, g):
    return [mix(a, b, (i / (n - 1)) ** g) for i in range(n)]


THEMES = {
    'dark': dict(
        bg='#0d1117', border='#30363d',
        sky=[(0, '#05080d'), (.55, '#0a111a'), (1, '#121d26')],
        stars=True, star='#c9d6df', fall='#b4d3cb', fall_op=.38, fall_spread=4.5,
        # aurora
        a_edge=dict(base=150, sag=-45, cx=640, half=620, fold=9, seed=12, tilt=34),
        a_h=68, a_colh=105, a_gain=1.0, a_core=.3, a_halo=.10, a_diff=.7, a_rays=.5, a_nrays=150,
        a_seed=17, a_env=560, ray_top=0, ray_bot=190, refl_top=90,
        a_cols=[(0, '#b4f7d8'), (.06, '#72eab5'), (.25, '#43d39c'), (.55, '#2ea88b'),
                (.8, '#4a72b0'), (1, '#7c5cc2')],
        a_haze=.10, aur_refl=.85, veil=(.05, .75),
        mtn=ramp('#1d2b33', '#070b0f', 7, 1.1),
        mist='#34484e', mist_mix=[.5, .55, .55, .6, .6, .62, .65], mist_op=1.0, mist_after=[0, 1, 2, 3, 4],
        ridge='#9fdcc2', ridge_op=[.36, .32, .28, .25, .22, .20, .18],
        water=[(0, '#15212a'), (.4, '#0f1820'), (1, '#0a0f16')], water_mid='#0f1820',
        refl_mix=.35, refl_op=.75, shore_op=.5, streak='#0b131a', streak_k=.55,
        ripple='#547078', ripple_op=.32,
        ship='#05080b', rig='#34474d', rig_w=.55, strake='#1c272c', rim='#4b6863',
        shield_a='#131b21', shield_b='#24303a', shield_rim='#476060', boss='#5d746f',
        oar='#090d12', oar_wet_op=.3,
        sail_top='#33292a', sail_bot='#241d20', stripe='#61392f', stripe_op=.8, seam='#000000',
        sail_edge='#4e3a34',
        wake='#86a7a0', refl_ship='#04070a', refl_sail='#22191b', ship_refl_op=.85, gap=1.0,
    ),
    'light': dict(
        # blue hour: the sky still pale, the aurora only just showing
        bg='#ffffff', border='#d0d7de',
        sky=[(0, '#93a0bd'), (.55, '#c6ccd8'), (1, '#e9e5de')],
        stars=False, star='#ffffff', fall='#ffffff', fall_op=.85,
        a_edge=dict(base=150, sag=-45, cx=640, half=620, fold=9, seed=12, tilt=34),
        a_h=68, a_colh=105, a_gain=.95, a_core=.3, a_halo=.06, a_diff=.75, a_rays=.4, a_nrays=150,
        a_seed=17, a_env=560, ray_top=0, ray_bot=190, refl_top=90,
        a_cols=[(0, '#4fdba3'), (.06, '#2fcc90'), (.25, '#25b886'), (.55, '#2f9f8c'),
                (.8, '#6f86c8'), (1, '#8d72d2')],
        a_haze=.10, aur_refl=.7, veil=(.1, .8),
        mtn=ramp('#b3bcc8', '#3c4653', 7, 0.9),
        mist='#e3e5e9', mist_mix=[.5, .55, .55, .6, .6, .62, .62], mist_op=1.0, mist_after=[0, 1, 2, 3, 4],
        ridge='#2f4f55', ridge_op=[.2, .2, .22, .22, .24, .24, .25],
        water=[(0, '#dfe1e4'), (.4, '#d6dadf'), (1, '#ced3d9')], water_mid='#d6dadf',
        refl_mix=.42, refl_op=.8, shore_op=.6, streak='#e4e7ea', streak_k=.6,
        ripple='#7d8894', ripple_op=.28,
        ship='#1b222a', rig='#4a5561', rig_w=.6, strake='#38424d', rim='#4c5865',
        shield_a='#2a333d', shield_b='#5a6570', shield_rim='#1b222a', boss='#c9cfd5',
        oar='#1f262e', oar_wet_op=.3,
        sail_top='#93604f', sail_bot='#835546', stripe='#ece4d8', stripe_op=.62, seam='#3a2a25',
        sail_edge='#5a3b32',
        wake='#4f5b68', refl_ship='#3a434e', refl_sail='#8f6a5f', ship_refl_op=.55, gap=1.0,
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
    # the night scene is the banner on both GitHub themes; other themes stay buildable by name
    path = os.path.join(HERE, 'banner.svg' if name == 'dark' else f'banner-{name}.svg')
    with open(path, 'w') as fh:
        fh.write(svg)
    print(path, len(svg.encode()) // 1024, 'KB')


if __name__ == '__main__':
    import sys
    for n in (sys.argv[1:] or ['dark']):
        build(n)
