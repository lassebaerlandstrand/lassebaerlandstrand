#!/usr/bin/env python3
"""Animated fjord banner: a small fleet of Viking longships rowing down the fjord.

Variant of ../longship (same fjord, same ship drawing), but with four ships at
different depths instead of one.

Scene (back to front)
---------------------
sky gradient (faint stars at night), low moon / pale sun with a soft glow, seven
procedurally generated fjord-wall silhouettes (glacial U-valley profile) receding
into mist, drifting mist bands, the water (mirrored mountains, a glade of glitter
under the light, slow ripple lines), the fleet's reflections (horizontal strips
that sway more the nearer they are), all the wakes, and the ships far to near,
with a low mist band drifting between the two farthest and the two nearest.

Depth
-----
Every ship lives on a flat sea seen from a fixed eye height, so a ship whose
water line sits d px below the horizon is drawn at scale d / DEPTH_REF and moves
across the screen at that fraction of its true speed: parallax for free.  The
farther ships are pushed toward the haze colour (aerial perspective) and the
mist drifts in front of the farthest ones.

Geometry is shared: hull, stems, sail, rigging, shields and stripe patterns are
paint-free <path>/<g> definitions that every ship <use>s with its own colours.
Two hull types exist (a 23 m Gokstad-like ship and a shorter karve with fewer
oars and shields).  Each ship has its own oar definition so it can row at its own
rhythm; stroke period and speed stay within ~5 % of the single ship's ratio
(23 px/s at 3.2 s a stroke) so the blades still "grip" the water.

Motion (all SMIL, each ship's loop seamless; ships wrap independently):
* each ship crosses once per its own loop and re-enters off-screen;
* oars row (catch, drive, release, feathered recovery), the hull surges a hair;
* catch rings stay put in the world, Kelvin wakes and stern trails follow;
* reflections sway in strips, glitter twinkles, mist drifts, a waterfall runs.

Usage: python generate.py  -> banner-dark.svg, banner-light.svg
"""
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

W, H = 1200, 340
YH = 226.0          # horizon (water line of the distant shore)
LIGHT_X, LIGHT_Y, LIGHT_R = 668.0, 150.0, 15.0
K_PERSP = 0.46      # screen-y per unit of depth toward the viewer (ship frame)
DEPTH_REF = 50.0    # a ship whose water line is this far below YH is drawn at scale 1
REF_V, REF_STROKE = 23.0, 3.2
SHIP_HALF = 108.0
WAKE_LEN = 240.0

# far -> near.  x0: screen x of the ship centre at t = 0; loop: seconds per
# crossing (each ship wraps on its own, seamlessly).  Loops and x0 came from a
# small search that scores ~40 minutes of the scene for an empty frame, ships
# hiding behind each other and lopsided balance, with extra weight on the first
# 90 s after page load (ships may still pass in front of each other: parallax).
FLEET = [
    dict(k='d', geo='long', s=.20, stroke=3.1, loop=268.0, x0=174, haze=.66, sail='v5', shields='alt',
         puddles=False, detail=False),
    dict(k='c', geo='short', s=.33, stroke=3.3, loop=190.0, x0=828, haze=.50, sail='lat', shields='third',
         puddles=False, detail=False),
    dict(k='b', geo='long', s=.56, stroke=2.9, loop=95.0, x0=997, haze=.30, sail='v5', shields='pair',
         puddles=True, detail=True),
    dict(k='a', geo='long', s=1.0, stroke=3.2, loop=70.0, x0=338, haze=0.0, sail='v7', shields='alt',
         puddles=True, detail=True),
]


def span(sp):
    """Screen x range of the ship centre over one crossing (enter left, leave right)."""
    half = SHIP_HALF * (0.84 if sp['geo'] == 'short' else 1.0)
    return -(half + 12) * sp['s'], W + (half + 175) * sp['s']


for sp in FLEET:
    sp['yw'] = YH + DEPTH_REF * sp['s']
    xa, xb = span(sp)
    sp['v'] = (xb - xa) / sp['loop'] / sp['s']        # local (world) px/s


def f(v):
    s = f'{v:.1f}'
    if s.endswith('.0'):
        s = s[:-2]
    if s == '-0':
        s = '0'
    return s


def f2(v):
    s = f'{v:.2f}'.rstrip('0').rstrip('.')
    return '0' if s in ('-0', '') else s


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



# ------------------------------------------------------------------ ship geometry
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


def build_geo(kx=1.0, hy=1.0, n_oars=14, n_shields=21, mast_top=-112.0, sail_half=42.0):
    """kx: hull length factor, hy: stem height factor."""
    L = 70 * kx
    bow = stem_centerline(+1, (L, 1.5), (18, 3), (1, 15 * hy), (L + 21, -41 * hy), 8.6 * hy ** .5, 1.2)
    stern = stem_centerline(-1, (-L, 1.5), (17, 3), (1, 14 * hy), (-L - 19, -38 * hy), 7.6 * hy ** .5, 1.15)
    HJ = -26.0 * hy

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
    hull = np.vstack([[[-L, 1.5], [L, 1.5]], bow[1:ib + 1], sheer[::-1][1:-1], stern[:ia + 1][::-1]])
    return dict(bow=bow, stern=stern, hull=hull, L=L,
                bow_post=tapered(bow, 4.6, 1.4), stern_post=tapered(stern, 4.4, 1.3),
                sheer=sheer, xs=xs, sheer_at=lambda x: np.interp(x, xs, sheer_y),
                mast_x=4.0 * kx, mast_top=mast_top, yard_y=mast_top + 11, sail_half=sail_half,
                sail_bot=-30.0, oar_x=np.linspace(-54, 56, n_oars) * kx,
                shield_x=np.linspace(-63, 65, n_shields) * kx)


GEOS = {
    'long': build_geo(),
    'short': build_geo(kx=.84, hy=.92, n_oars=12, n_shields=17, mast_top=-100.0, sail_half=37.0),
}
OAR_R = 44.0


def sail_path(G, refl=False):
    mx, hs, yy, sb = G['mast_x'], G['sail_half'], G['yard_y'], G['sail_bot']
    sx0, sx1 = mx - hs, mx + hs
    sc = hs / 42.0
    top_l, top_r = (sx0 - 1, yy + 1.2), (sx1 + 1, yy + 1.2)
    bot_l, bot_r = (sx0 + 3, sb), (sx1 + 1, sb + 1)
    ya = yy + 15 * sc * (yy / -101)
    yb = sb - 22
    return (f'M{f(top_l[0])} {f(top_l[1])}L{f(top_r[0])} {f(top_r[1])}'
            f'C{f(sx1 + 6)} {f(ya)} {f(sx1 + 6)} {f(yb)} {f(bot_r[0])} {f(bot_r[1])}'
            f'Q{f(mx + 2)} {f(sb - 9)} {f(bot_l[0])} {f(bot_l[1])}'
            f'C{f(sx0 + 1)} {f(sb - 26)} {f(sx0 - 3)} {f(ya)} {f(top_l[0])} {f(top_l[1])}Z')


def dots(pts):
    """Zero-length subpaths; with round caps each one renders as a disc."""
    return ''.join(f'M{f2(x)} {f2(y)}h.01' for x, y in pts)


SHIELD_PATTERNS = {
    'alt': lambda i: i % 2,
    'pair': lambda i: (i // 2) % 2,
    'third': lambda i: 1 if i % 3 == 1 else 0,
}


def geo_defs(G, g, needed_sails, needed_shields):
    """Paint-free shared definitions for hull type G (id prefix g)."""
    S = G
    out = []
    mx, mt, yy, hs, sb = G['mast_x'], G['mast_top'], G['yard_y'], G['sail_half'], G['sail_bot']
    sx0, sx1 = mx - hs, mx + hs
    sat = S['sheer_at']
    # hull + stems
    out.append(f'<g id="{g}hull"><path d="{poly(S["hull"])}"/><path d="{poly(S["bow_post"])}"/>'
               f'<path d="{poly(S["stern_post"])}"/></g>')
    out.append(f'<clipPath id="{g}hc"><path d="{poly(S["hull"])}"/></clipPath>')
    strakes = []
    xs, sy = S['xs'], S['sheer'][:, 1]
    for fr in (0.36, 0.62, 0.84):
        y = sy * (1 - fr) + 1.5 * fr
        m = (y < 0.5)
        strakes.append(poly(np.stack([xs[m], y[m]], 1), False))
    out.append(f'<path id="{g}strk" clip-path="url(#{g}hc)" d="{"".join(strakes)}" fill="none" stroke-width=".55"/>')
    bowc = S['bow'][22:NRISE + 55]
    sterc = S['stern'][22:NRISE + 55]
    out.append(f'<g id="{g}rims" fill="none"><path d="{poly(S["sheer"][3:-3], False)}" stroke-width=".8"/>'
               f'<path d="{poly(bowc + np.array([0.2, -0.4]), False)}" stroke-width=".6" opacity=".8"/>'
               f'<path d="{poly(sterc + np.array([-0.2, -0.4]), False)}" stroke-width=".6" opacity=".8"/></g>')
    # rigging: stays, shrouds, sheets
    bow_top = S['bow'][NRISE + 12]
    stern_top = S['stern'][NRISE + 12]
    bot_l, bot_r = (sx0 + 3, sb), (sx1 + 1, sb + 1)
    out.append(f'<path id="{g}rig" fill="none" stroke-linecap="round" d="'
               f'M{f(mx)} {f(mt + 2)}L{f(bow_top[0] - 1)} {f(bow_top[1] + 4)}'
               f'M{f(mx)} {f(mt + 2)}L{f(stern_top[0] + 2)} {f(stern_top[1] + 5)}'
               f'M{f(mx)} {f(mt + 6)}L{f(mx - 30)} {f(sat(mx - 30))}'
               f'M{f(mx)} {f(mt + 6)}L{f(mx - 44)} {f(sat(mx - 44))}'
               f'M{f(mx)} {f(mt + 6)}L{f(mx + 34)} {f(sat(mx + 34))}'
               f'M{f(bot_l[0])} {f(bot_l[1])}L{f(sx0 - 22)} {f(sat(sx0 - 22))}'
               f'M{f(bot_r[0])} {f(bot_r[1])}L{f(sx1 + 20)} {f(sat(sx1 + 20))}"/>')
    out.append(f'<path id="{g}mast" d="M{f(mx - 1.4)} {f(sat(mx))}L{f(mx - 0.9)} {f(mt)}'
               f'L{f(mx + 0.9)} {f(mt)}L{f(mx + 1.4)} {f(sat(mx))}Z"/>')
    out.append(f'<path id="{g}yard" d="M{f(sx0 - 6)} {f(yy + .6)}Q{f(mx)} {f(yy - 1.5)} {f(sx1 + 6)} {f(yy + .6)}" '
               f'stroke-width="2.2" stroke-linecap="round" fill="none"/>')
    sail = sail_path(G)
    out.append(f'<path id="{g}sail" d="{sail}"/>')
    out.append(f'<clipPath id="{g}sc"><path d="{sail}"/></clipPath>')
    seams = ''.join(f'M{f(sx0 - 4)} {f(y)}L{f(sx1 + 6)} {f(y + 0.6)}'
                    for y in np.linspace(yy, sb, 5)[1:-1])
    out.append(f'<path id="{g}seam" clip-path="url(#{g}sc)" d="{seams}" stroke-width=".5" fill="none" opacity=".45"/>')
    # sail patterns
    for pat in needed_sails:
        if pat.startswith('v'):
            nst = int(pat[1:])
            st = []
            for i in range(1, nst, 2):
                a = sx0 - 6 + (sx1 - sx0 + 12) * i / nst
                b = sx0 - 6 + (sx1 - sx0 + 12) * (i + 1) / nst
                st.append(f'M{f(a)} {f(mt)}L{f(b)} {f(mt)}L{f(b + 2)} -20L{f(a + 2)} -20Z')
            out.append(f'<path id="{g}p{pat}" clip-path="url(#{g}sc)" d="{"".join(st)}"/>')
        elif pat == 'lat':
            # lozenge lattice (as on the Gotland picture stones)
            step = (sx1 - sx0) / 4.0
            hgt = sb - yy
            ln = []
            for i in range(-6, 11):
                x = sx0 + i * step
                ln.append(f'M{f(x)} {f(yy)}l{f(hgt * .62)} {f(hgt)}')
                ln.append(f'M{f(x)} {f(yy)}l{f(-hgt * .62)} {f(hgt)}')
            out.append(f'<path id="{g}plat" clip-path="url(#{g}sc)" d="{"".join(ln)}" fill="none" '
                       f'stroke-width="{f2(step * .24)}"/>')
    # shields: one disc per subpath
    pts = [(x, float(sat(x)) + 0.6) for x in S['shield_x']]
    out.append(f'<path id="{g}shAll" d="{dots(pts)}" fill="none" stroke-linecap="round"/>')
    for pat in needed_shields:
        sel = SHIELD_PATTERNS[pat]
        out.append(f'<path id="{g}sh{pat}" d="{dots([p for i, p in enumerate(pts) if sel(i)])}" '
                   f'fill="none" stroke-linecap="round"/>')
    # simplified mast for the reflection
    out.append(f'<path id="{g}rmast" d="M{f(mx - 1.2)} -12L{f(mx - 1)} {f(mt)}L{f(mx + 1)} {f(mt)}L{f(mx + 1.2)} -12Z"/>')
    return ''.join(out)


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


def oar_defs(sp, col):
    """Per-ship oar (own stroke period) and the group of all its oars."""
    k, G = sp['k'], GEOS[sp['geo']]
    fr = oar_frames()
    kt = ';'.join(f'{p:.4f}' for p, _, _ in fr)
    shaft = ';'.join(f'M0 0L{dx:.2f} {dy:.2f}' for _, dx, dy in fr)
    blade = ';'.join(f'M{dx * .72:.2f} {dy * .72:.2f}L{dx:.2f} {dy:.2f}' for _, dx, dy in fr)
    wsh = .85 / max(sp['s'], .5) ** .35
    oar = (f'<g id="oar{k}" stroke="{col}" stroke-linecap="round" fill="none">'
           f'<path stroke-width="{f2(wsh)}" d="M0 0L{fr[0][1]:.2f} {fr[0][2]:.2f}">'
           f'<animate attributeName="d" dur="{sp["stroke"]}s" repeatCount="indefinite" keyTimes="{kt}" values="{shaft}"/></path>'
           f'<path stroke-width="1.9" d="M0 0L1 1">'
           f'<animate attributeName="d" dur="{sp["stroke"]}s" repeatCount="indefinite" keyTimes="{kt}" values="{blade}"/></path>'
           f'</g>')
    uses = ''.join(f'<use href="#oar{k}" x="{f(x)}" y="{f(G["sheer_at"](x) + 5.4)}"/>' for x in G['oar_x'])
    return oar + f'<g id="oars{k}">{uses}</g>'


def ship_uses(sp, c):
    """The ship, as <use>s of the shared geometry painted with its colours c."""
    g, k = sp['geo'], sp['k']
    rw = f2(c['rig_w'] / sp['s'] ** .45)
    p = [f'<use href="#{g}rig" stroke="{c["rig"]}" stroke-width="{rw}"/>',
         f'<use href="#{g}mast" fill="{c["ship"]}"/>',
         f'<use href="#{g}sail" fill="url(#sailg{k})"/>']
    pat = sp['sail']
    if pat == 'lat':
        p.append(f'<use href="#{g}plat" stroke="{c["stripe"]}" opacity="{c["stripe_op"]}"/>')
    else:
        p.append(f'<use href="#{g}p{pat}" fill="{c["stripe"]}" opacity="{c["stripe_op"]}"/>')
    if sp['detail']:
        p.append(f'<use href="#{g}seam" stroke="{c["seam"]}"/>')
    p.append(f'<use href="#{g}sail" fill="none" stroke="{c["sail_edge"]}" stroke-width=".7"/>')
    p.append(f'<use href="#{g}yard" stroke="{c["ship"]}"/>')
    p.append(f'<use href="#{g}hull" fill="{c["ship"]}"/>')
    if sp['detail']:
        p.append(f'<use href="#{g}strk" stroke="{c["strake"]}"/>')
    p.append(f'<use href="#{g}rims" stroke="{c["rim"]}"/>')
    p.append(f'<use href="#{g}shAll" stroke="{c["shield_rim"]}" stroke-width="7.35"/>')
    p.append(f'<use href="#{g}shAll" stroke="{c["shield_a"]}" stroke-width="6.25"/>')
    p.append(f'<use href="#{g}sh{sp["shields"]}" stroke="{c["shield_b"]}" stroke-width="6.25"/>')
    p.append(f'<use href="#{g}shAll" stroke="{c["boss"]}" stroke-width="1.8"/>')
    p.append(f'<use href="#oars{k}"/>')
    return ''.join(p)


def mover_anim(sp):
    xa, xb = span(sp)
    x0, yw = sp['x0'], sp['yw']
    k = 1 - (x0 - xa) / (xb - xa)
    return (f'<animateTransform attributeName="transform" type="translate" dur="{f(sp["loop"])}s" '
            f'repeatCount="indefinite" calcMode="linear" '
            f'values="{f(x0)} {f(yw)};{f(xb)} {f(yw)};{f(xa)} {f(yw)};{f(x0)} {f(yw)}" '
            f'keyTimes="0;{k:.5f};{k:.5f};1"/>')


def surge_anim(sp):
    vals = []
    n = 12
    for i in range(n + 1):
        ph = i / n
        vals.append(f'{1.1 * math.sin(2 * math.pi * (ph - 0.12)):.2f} 0')
    return (f'<animateTransform attributeName="transform" type="translate" dur="{sp["stroke"]}s" '
            f'repeatCount="indefinite" values="{";".join(vals)}"/>')


def wake_svg(sp, c):
    k, G = sp['k'], GEOS[sp['geo']]
    L0 = G['L']
    out = []
    col = c['wake']
    ws = 1 / sp['s'] ** .5          # thicker local lines far away (they still shrink on screen)
    slope = K_PERSP * math.tan(math.radians(19.5))
    B = (L0 + 3.0, 1.4)
    for side in (+1, -1):
        Lw = WAKE_LEN if side > 0 else WAKE_LEN * 0.95
        ex, ey = B[0] - Lw, B[1] + side * slope * Lw
        out.append(f'<path d="M{f(B[0])} {f(B[1])}L{f(ex)} {f(ey)}" stroke="url(#{"wn" if side > 0 else "wf"}{k})" '
                   f'stroke-width="{f2((0.9 if side > 0 else 0.7) * ws)}" fill="none"/>')
        nfe = 13 if sp['detail'] else 8
        for i in range(1, nfe + 1):
            s = Lw * (i / (nfe + 1)) ** 1.15
            if side < 0 and B[0] - s > -L0 - 14:
                continue      # hidden behind the hull
            if side > 0 and B[0] - s > -40 * L0 / 70:
                continue      # still tucked under the oars
            px, py = B[0] - s, B[1] + side * slope * s
            ln = 7 + 10 * (s / Lw)
            dx, dy = -0.92 * ln, side * 0.10 * ln * 0.9
            op = 0.7 * (1 - s / Lw) ** 1.3
            out.append(f'<path d="M{f(px)} {f(py)}l{f(dx)} {f(dy)}" stroke="{col}" '
                       f'stroke-width="{f2(.8 * ws)}" opacity="{op:.2f}" fill="none"/>')
    out.append(f'<path d="M{f(L0 + 2)} 2.4q-6 2.2-15 2.4M{f(L0)} 3.8q-8 2-19 1.6" stroke="{col}" '
               f'stroke-width="{f2(.8 * ws)}" fill="none" opacity=".55"/>')
    per = 26.0
    dur = per / sp['v']
    for j, (dy, op, w) in enumerate(((2.6, .55, .8), (5.2, .35, .7), (-0.6, .3, .6))):
        if not sp['detail'] and j == 2:
            continue
        out.append(f'<path d="M{f(-L0)} {f(dy)}L{f(-L0 - 230)} {f(dy + j * 1.5)}" stroke="url(#tf{k})" '
                   f'stroke-width="{f2(w * ws)}" stroke-dasharray="{12 - 3 * j} {14 + 3 * j}" opacity="{op}" fill="none">'
                   f'<animate attributeName="stroke-dashoffset" values="{j * 7};{j * 7 - per}" dur="{dur:.3f}s" '
                   f'repeatCount="indefinite"/></path>')
    return ''.join(out)


def wake_grads(sp, c):
    k, L0 = sp['k'], GEOS[sp['geo']]['L']
    col = c['wake']
    a = L0 + 7
    return (f'<linearGradient id="wf{k}" x1="{f(a)}" y1="0" x2="{f(a - WAKE_LEN)}" y2="0" gradientUnits="userSpaceOnUse">'
            f'<stop offset="0" stop-color="{col}" stop-opacity=".6"/>'
            f'<stop offset=".55" stop-color="{col}" stop-opacity="0"/></linearGradient>'
            f'<linearGradient id="wn{k}" x1="{f(L0 + 3)}" y1="0" x2="{f(L0 + 3 - WAKE_LEN)}" y2="0" gradientUnits="userSpaceOnUse">'
            f'<stop offset=".3" stop-color="{col}" stop-opacity="0"/>'
            f'<stop offset=".55" stop-color="{col}" stop-opacity=".45"/>'
            f'<stop offset="1" stop-color="{col}" stop-opacity="0"/></linearGradient>'
            f'<linearGradient id="tf{k}" x1="{f(-L0 - 4)}" y1="0" x2="{f(-L0 - 234)}" y2="0" gradientUnits="userSpaceOnUse">'
            f'<stop offset="0" stop-color="{col}" stop-opacity="1"/>'
            f'<stop offset="1" stop-color="{col}" stop-opacity="0"/></linearGradient>')


def puddles_svg(sp, c):
    """Ring left by each blade at the catch; stays fixed in the world."""
    k, G = sp['k'], GEOS[sp['geo']]
    fr = oar_frames()
    cdx, cdy = fr[0][1], fr[0][2]
    life = 4.2
    stroke = sp['stroke']
    dur = 2 * stroke
    kt_end = min(life / dur, 0.98)
    back = sp['v'] * life
    ws = f2(.7 / sp['s'] ** .5)
    defs = []
    for j in range(2):
        b = f'begin="{j * stroke}s" dur="{dur}s" repeatCount="indefinite"'
        defs.append(
            f'<ellipse id="pud{k}{j}" rx="1" ry=".3" fill="none" stroke="{c["wake"]}" stroke-width="{ws}" opacity="0">'
            f'<animate attributeName="rx" values="2;14;14" keyTimes="0;{kt_end:.3f};1" {b}/>'
            f'<animate attributeName="ry" values=".6;3.6;3.6" keyTimes="0;{kt_end:.3f};1" {b}/>'
            f'<animate attributeName="opacity" values="0;.38;0;0" keyTimes="0;.05;{kt_end:.3f};1" {b}/>'
            f'<animateTransform attributeName="transform" type="translate" values="0 0;{-back:.1f} 0;{-back:.1f} 0" '
            f'keyTimes="0;{kt_end:.3f};1" {b}/>'
            f'</ellipse>')
    uses = []
    for x in G['oar_x']:
        for j in range(2):
            uses.append(f'<use href="#pud{k}{j}" x="{f(x + cdx)}" y="{f(cdy + G["sheer_at"](x) + 5.8)}"/>')
    return ''.join(defs), ''.join(uses)


def ship_colours(th, sp):
    """Theme colours with the ship's own variation, pushed into the haze by depth."""
    c = dict(th['shipcol'])
    c.update(th['variants'].get(sp['k'], {}))
    hz, a = th['haze'], sp['haze']
    out = {}
    for key, v in c.items():
        if isinstance(v, str) and v.startswith('#'):
            amt = a * th['haze_k'].get(key, 1.0)
            out[key] = mix(v, hz, min(amt, 1.0))
        else:
            out[key] = v
    return out


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


def stars_svg(th):
    r = np.random.default_rng(91)
    out = [f'<g fill="{th["star"]}">']
    tw = []
    for i in range(46):
        x, y = r.uniform(20, W - 20), r.uniform(8, 150) ** 1.0
        if math.hypot(x - LIGHT_X, (y - LIGHT_Y) * 1.6) < 150:
            continue
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


def glade_svg(th, uid, defs):
    r = np.random.default_rng(17)
    groups = [[] for _ in range(5)]
    y = YH + 2.5
    gap = 2.2
    while y < H:
        depth = (y - YH) / (H - YH)
        halfw = 16 + 120 * depth ** 0.9
        n = int(2 + 4 * depth ** 0.5)
        for _ in range(n):
            x = LIGHT_X + r.normal(0, 0.45) * halfw
            ln = r.uniform(3, 9) * (0.6 + 1.4 * depth)
            cen = math.exp(-((x - LIGHT_X) / (halfw * 0.75)) ** 2)
            op = (0.35 + 0.65 * cen) * (1 - 0.55 * depth)
            groups[r.integers(0, 5)].append(
                f'<path d="M{f(x - ln / 2)} {f(y)}h{f(ln)}" stroke-opacity="{op:.2f}"/>')
        y += gap
        gap *= 1.06
    out = [f'<g stroke="{th["glade"]}" stroke-width="{th["glade_w"]}" stroke-linecap="round" '
           f'opacity="{th["glade_op"]}">']
    for i, g in enumerate(groups):
        dur = 3.1 + 0.83 * i
        vals = ';'.join(f'{0.35 + 0.65 * (0.5 + 0.5 * math.sin(2 * math.pi * (j / 6 + i / 5))):.2f}' for j in range(7))
        out.append(f'<g>{"".join(g)}<animate attributeName="opacity" values="{vals}" dur="{dur:.2f}s" '
                   f'repeatCount="indefinite"/></g>')
    out.append('</g>')
    # soft vertical column of reflected light
    defs.append(f'<radialGradient id="col{uid}" cx="{LIGHT_X}" cy="{YH}" r="120" gradientUnits="userSpaceOnUse" '
                f'gradientTransform="translate({LIGHT_X} {YH}) scale(.55 1) translate({-LIGHT_X} {-YH})">'
                f'<stop offset="0" stop-color="{th["glade"]}" stop-opacity="{th["col_op"]}"/>'
                f'<stop offset="1" stop-color="{th["glade"]}" stop-opacity="0"/></radialGradient>')
    return f'<rect x="{LIGHT_X - 70}" y="{YH}" width="140" height="{H - YH}" fill="url(#col{uid})"/>' + ''.join(out)


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



# ------------------------------------------------------------------ scene
def fleet_svg(th, defs):
    """Reflections, wakes and ships for the whole fleet."""
    sails = {g: set() for g in GEOS}
    shields = {g: set() for g in GEOS}
    for sp in FLEET:
        sails[sp['geo']].add(sp['sail'])
        shields[sp['geo']].add(sp['shields'])
    for g, G in GEOS.items():
        defs.append(geo_defs(G, g, sorted(sails[g]), sorted(shields[g])))

    cols = {}
    wakes, ships, pud_layer = [], [], []
    for sp in FLEET:
        k, g, s = sp['k'], sp['geo'], sp['s']
        c = ship_colours(th, sp)
        cols[k] = c
        G = GEOS[g]
        defs.append(f'<linearGradient id="sailg{k}" x1="0" y1="{f(G["yard_y"])}" x2="0" y2="{f(G["sail_bot"])}" '
                    f'gradientUnits="userSpaceOnUse"><stop offset="0" stop-color="{c["sail_top"]}"/>'
                    f'<stop offset="1" stop-color="{c["sail_bot"]}"/></linearGradient>')
        defs.append(wake_grads(sp, c))
        defs.append(oar_defs(sp, c['oar']))
        pud = ''
        if sp['puddles']:
            pd, pud = puddles_svg(sp, c)
            defs.append(pd)
        mv = mover_anim(sp)
        sc = f'scale({f2(s)})'
        wakes.append(f'<g>{mv}<g transform="{sc}">{wake_svg(sp, c)}{pud}</g></g>')
        ships.append(f'<g>{mv}<g transform="{sc}"><g>{surge_anim(sp)}{ship_uses(sp, c)}</g></g></g>')
        # reflection pieces, mirrored about this ship's own water line
        mirror = f'translate(0 {f(2 * sp["yw"])}) scale(1 -1)'
        defs.append(f'<g id="rb{k}" transform="{mirror}"><g>{mv}<g transform="{sc}">'
                    f'<use href="#{g}sail" fill="{c["refl_sail"]}"/>'
                    f'<use href="#{g}rmast" fill="{c["refl_ship"]}"/>'
                    f'<use href="#{g}hull" fill="{c["refl_ship"]}"/></g></g></g>')
        defs.append(f'<g id="ro{k}" transform="{mirror}"><g>{mv}<g transform="{sc}" opacity=".9">'
                    f'<use href="#oars{k}"/></g></g></g>')

    # reflection strips, per ship.  Strips sway more the nearer they are; the
    # two far ships sway by well under a pixel, so their strips are static and
    # merged into three fading bands (one clone each) to save work.
    strips = []
    n = 0
    for sp in FLEET:
        k, s, yw = sp['k'], sp['s'], sp['yw']
        y0 = yw - 1.5 * s
        hl = 3.2      # strip height; on screen it is hl * s ** 0.3 (coarser far away)
        j = 0
        rows = []
        while y0 < min(H, yw + 116 * s):
            h = hl * s ** 0.3 * (1.0 if s > .9 else 1.25)
            gap = min(th['gap'], 0.25 * h) * min(1, j / 3)
            op = min(1.0, max(0.0, 1 - (y0 - yw) / (120 * s)))
            rows.append((y0, h, gap, op))
            y0 += h
            hl = min(hl * 1.13 + 0.25, 11)
            j += 1
        if s < 0.4:
            nb = 3
            for b in range(nb):
                sel = rows[b * len(rows) // nb:(b + 1) * len(rows) // nb]
                cid = f'rs{n}'
                n += 1
                rects = ''.join(f'<rect y="{f2(y)}" width="{W}" height="{f2(h - gp)}"/>' for y, h, gp, _ in sel)
                defs.append(f'<clipPath id="{cid}">{rects}</clipPath>')
                op = sum(r[3] for r in sel) / len(sel)
                inner = f'<use href="#rb{k}"/>' + (f'<use href="#ro{k}"/>' if b == 0 else '')
                strips.append(f'<g clip-path="url(#{cid})" opacity="{op:.2f}">{inner}</g>')
            continue
        for j, (y0, h, gap, op) in enumerate(rows):
            d = y0 - YH
            cid = f'rs{n}'
            defs.append(f'<clipPath id="{cid}"><rect y="{f2(y0)}" width="{W}" height="{f2(h - gap)}"/></clipPath>')
            amp = max(0.15, min(3.4, 0.6 * (d / DEPTH_REF) ** 1.6))
            per = 3.4 + 0.9 * math.sin(n * 1.7) + 0.05 * j
            ph = (n * 0.37) % 1
            vals = ';'.join(f'{amp * math.sin(2 * math.pi * (i / 8 + ph)):.2f} 0' for i in range(9))
            o = '' if op > .995 else f' opacity="{op:.2f}"'
            inner = f'<use href="#rb{k}"/>' + (f'<use href="#ro{k}"/>' if y0 < yw + 16 * s else '')
            strips.append(f'<g clip-path="url(#{cid})"{o}><g>'
                          f'<animateTransform attributeName="transform" type="translate" values="{vals}" '
                          f'dur="{per:.2f}s" repeatCount="indefinite"/>{inner}</g></g>')
            n += 1
    refl = f'<g opacity="{th["ship_refl_op"]}">{"".join(strips)}</g>'
    return refl, ''.join(wakes), ships


def theme_svg(th, uid):
    xs = np.arange(-4, W + 5, 2.5)
    parts = []
    defs = []

    defs.append(f'<linearGradient id="sky" x1="0" y1="0" x2="0" y2="{YH}" gradientUnits="userSpaceOnUse">'
                + ''.join(f'<stop offset="{o}" stop-color="{c}"/>' for o, c in th['sky']) + '</linearGradient>')
    defs.append(f'<radialGradient id="glow" cx="{LIGHT_X}" cy="{LIGHT_Y}" r="330" gradientUnits="userSpaceOnUse" '
                f'gradientTransform="translate({LIGHT_X} {LIGHT_Y}) scale(1.25 .6) translate({-LIGHT_X} {-LIGHT_Y})">'
                f'<stop offset="0" stop-color="{th["glow"]}" stop-opacity="{th["glow_op"]}"/>'
                f'<stop offset=".35" stop-color="{th["glow"]}" stop-opacity="{th["glow_op"] * .35:.3f}"/>'
                f'<stop offset="1" stop-color="{th["glow"]}" stop-opacity="0"/></radialGradient>')
    defs.append(f'<radialGradient id="halo" cx="{LIGHT_X}" cy="{LIGHT_Y}" r="46" gradientUnits="userSpaceOnUse">'
                f'<stop offset=".3" stop-color="{th["halo"]}" stop-opacity="{th["halo_op"]}"/>'
                f'<stop offset="1" stop-color="{th["halo"]}" stop-opacity="0"/></radialGradient>')
    defs.append(f'<linearGradient id="water" x1="0" y1="{YH}" x2="0" y2="{H}" gradientUnits="userSpaceOnUse">'
                + ''.join(f'<stop offset="{o}" stop-color="{c}"/>' for o, c in th['water']) + '</linearGradient>')

    parts.append(f'<rect width="{W}" height="{YH + 2}" fill="url(#sky)"/>')
    parts.append(f'<rect width="{W}" height="{H}" fill="url(#glow)"/>')
    parts.append(f'<circle cx="{LIGHT_X}" cy="{LIGHT_Y}" r="46" fill="url(#halo)"/>')
    parts.append(f'<circle cx="{LIGHT_X}" cy="{LIGHT_Y}" r="{LIGHT_R}" fill="{th["disc"]}"/>')
    if th.get('stars'):
        parts.append(stars_svg(th))
    profiles = []
    for i, L in enumerate(LAYERS):
        y = layer_profile(L, xs)
        profiles.append(y)
        col = th['mtn'][i]
        gid = f'm{i}'
        topy = float(y.min())
        defs.append(f'<linearGradient id="{gid}" x1="0" y1="{f(topy)}" x2="0" y2="{YH}" gradientUnits="userSpaceOnUse">'
                    f'<stop offset="0" stop-color="{col}"/>'
                    f'<stop offset=".55" stop-color="{col}"/>'
                    f'<stop offset="1" stop-color="{mix(col, th["mist"], th["mist_mix"][i])}"/></linearGradient>')
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
            parts.append(f'<path d="{dd}" fill="none" stroke="{th["ridge"]}" stroke-width=".7" '
                         f'opacity="{th["ridge_op"][i]}"/>')
        for wl, wx in WATERFALLS:
            if wl == i:
                parts.append(waterfall_svg(th, wx, y, xs))
        if i in th['mist_after']:
            parts.append(mist_band(th, uid, i))

    # water
    parts.append(f'<rect y="{YH}" width="{W}" height="{H - YH}" fill="url(#water)"/>')
    rg = []
    for i, (L, y) in enumerate(zip(LAYERS, profiles)):
        yr = 2 * YH - y
        col = mix(th['mtn'][i], th['water_mid'], th['refl_mix'])
        gid = f'r{i}'
        defs.append(f'<linearGradient id="{gid}" x1="0" y1="{YH}" x2="0" y2="{f(min(float(yr.max()), H + 120))}" gradientUnits="userSpaceOnUse">'
                    f'<stop offset="0" stop-color="{mix(col, th["mist"], th["mist_mix"][i] * .8)}"/>'
                    f'<stop offset=".5" stop-color="{col}"/>'
                    f'<stop offset="1" stop-color="{col}"/></linearGradient>')
        pts = [(xs[0], YH - 1)] + list(zip(xs, yr)) + [(xs[-1], YH - 1)]
        rg.append(f'<path d="{poly(pts)}" fill="url(#{gid})"/>')
    parts.append(f'<g opacity="{th["refl_op"]}">{"".join(rg)}</g>')
    parts.append(f'<rect y="{YH - 10}" width="{W}" height="22" fill="url(#shoremist)"/>')
    defs.append(f'<linearGradient id="shoremist" x1="0" y1="{YH - 10}" x2="0" y2="{YH + 12}" gradientUnits="userSpaceOnUse">'
                f'<stop offset="0" stop-color="{th["mist"]}" stop-opacity="0"/>'
                f'<stop offset=".5" stop-color="{th["mist"]}" stop-opacity="{th["shore_op"]}"/>'
                f'<stop offset="1" stop-color="{th["mist"]}" stop-opacity="0"/></linearGradient>')
    parts.append(glade_svg(th, uid, defs))
    parts.append(ripples_svg(th))

    refl, wakes, ships = fleet_svg(th, defs)
    parts.append(refl)
    parts.append(wakes)
    for i, (sp, sh) in enumerate(zip(FLEET, ships)):
        parts.append(sh)
        if i == 1:
            # low mist drifting in front of the two farthest ships
            parts.append(mist_band(th, uid, 'fg'))
    return ''.join(defs), ''.join(parts)




THEMES = {
    'dark': dict(
        bg='#0d1117', border='#30363d',
        sky=[(0, '#0a0f17'), (.55, '#111a27'), (1, '#1d2838')],
        glow='#b9c6d8', glow_op=.20, halo='#e8e6dc', halo_op=.30, disc='#e9e5d8',
        stars=True, star='#c9d3e0', fall='#8696ab', fall_op=.55,
        mtn=ramp('#303d50', '#0c1219', 7, 1.25),
        mist='#3c4a5e', mist_mix=[.5, .55, .55, .6, .6, .62, .65], mist_op=1.0, mist_after=[0, 1, 2, 3, 4],
        ridge='#9fb0c6', ridge_op=[.38, .33, .29, .26, .23, .21, .19],
        water=[(0, '#1a2433'), (.4, '#121a26'), (1, '#0b1018')], water_mid='#121a26',
        refl_op=.75, refl_mix=.35, shore_op=.55,
        glade='#e6e1d2', glade_w=1.0, glade_op=.75, col_op=.18,
        ripple='#5b6b82', ripple_op=.35,
        ship_refl_op=.85, gap=1.0,
        haze='#2b384a',
        haze_k=dict(wake=.55, stripe=1.05, sail_top=1.05, sail_bot=1.05),
        shipcol=dict(
            ship='#06090d', rig='#3a4656', rig_w=.55, strake='#1f2833', rim='#4e5d72',
            shield_a='#151c26', shield_b='#26303d', shield_rim='#4a586b', boss='#5d6b7e',
            oar='#0a0e14',
            sail_top='#33292a', sail_bot='#241d20', stripe='#61392f', stripe_op=.8, seam='#000000',
            sail_edge='#4e3a34',
            wake='#8394ab', refl_ship='#05080c', refl_sail='#24191b'),
        variants=dict(
            b=dict(sail_top='#3a3631', sail_bot='#2b2825', stripe='#6a3a2e', stripe_op=.85,
                   sail_edge='#4d4740', shield_a='#3d3424', shield_b='#141a22', refl_sail='#24211d'),
            c=dict(sail_top='#3a2a28', sail_bot='#2a1f1f', stripe='#5b524a', stripe_op=.75,
                   shield_a='#2a3442', shield_b='#5a3229'),
            d=dict(sail_top='#3a3a3a', sail_bot='#2c2c2e', stripe='#5c3a30'),
        ),
    ),
    'light': dict(
        bg='#ffffff', border='#d0d7de',
        sky=[(0, '#e4e8ec'), (.6, '#eef0f0'), (1, '#f6f3ec')],
        glow='#f3dcc0', glow_op=.55, halo='#fbefe0', halo_op=.8, disc='#fbf6ec',
        fall='#ffffff', fall_op=.9,
        mtn=ramp('#d3d8dd', '#56616e', 7, 0.9),
        mist='#f1f1ee', mist_mix=[.5, .55, .55, .6, .6, .62, .62], mist_op=1.0, mist_after=[0, 1, 2, 3, 4],
        ridge='#3d4752', ridge_op=[.2, .2, .22, .22, .24, .24, .25],
        water=[(0, '#eceae4'), (.4, '#e5e6e5'), (1, '#dfe2e5')], water_mid='#e3e5e6',
        refl_op=.8, refl_mix=.42, shore_op=.6,
        glade='#ffffff', glade_w=1.1, glade_op=.95, col_op=.55,
        ripple='#7d8894', ripple_op=.30,
        ship_refl_op=.55, gap=1.0,
        haze='#c9d0d8',
        haze_k=dict(wake=.6),
        shipcol=dict(
            ship='#1b222a', rig='#4a5561', rig_w=.6, strake='#38424d', rim='#4c5865',
            shield_a='#2a333d', shield_b='#5a6570', shield_rim='#1b222a', boss='#c9cfd5',
            oar='#1f262e',
            sail_top='#93604f', sail_bot='#835546', stripe='#ece4d8', stripe_op=.62, seam='#3a2a25',
            sail_edge='#5a3b32',
            wake='#4f5b68', refl_ship='#3a434e', refl_sail='#8f6a5f'),
        variants=dict(
            b=dict(sail_top='#e4dccd', sail_bot='#d2c8b7', stripe='#9c4434', stripe_op=.8,
                   sail_edge='#7a6a58', shield_a='#c2a25c', shield_b='#2a333d', refl_sail='#b9ae9e'),
            c=dict(sail_top='#8a5a48', sail_bot='#7a4f40', stripe='#ece4d8', stripe_op=.55,
                   shield_a='#5a6a7a', shield_b='#9a4a3a'),
            d=dict(sail_top='#a07a5e', sail_bot='#8f6c53', stripe='#4a3a34', stripe_op=.5),
        ),
    ),
}


def build(name):
    th = THEMES[name]
    uid = name[0]
    defs, body = theme_svg(th, uid)
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">'
           f'<defs><clipPath id="card"><rect width="{W}" height="{H}" rx="16"/></clipPath>{defs}</defs>'
           f'<g clip-path="url(#card)"><rect width="{W}" height="{H}" fill="{th["bg"]}"/>{body}</g>'
           f'<rect x=".5" y=".5" width="{W - 1}" height="{H - 1}" rx="15.5" fill="none" stroke="{th["border"]}"/>'
           f'</svg>')
    path = os.path.join(HERE, f'banner-{name}.svg')
    with open(path, 'w') as fh:
        fh.write(svg)
    print(path, len(svg.encode()) // 1024, 'KB', svg.count('<'), 'elements')


if __name__ == '__main__':
    for n in ('dark', 'light'):
        build(n)
