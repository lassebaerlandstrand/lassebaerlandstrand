#!/usr/bin/env python3
"""Animated fjord + Viking longship banner.

Scene (back to front)
---------------------
sky gradient (faint stars at night), low moon / pale sun with a soft glow, seven
procedurally generated
fjord-wall silhouettes (glacial U-valley profile: steep cliffs, fjell plateaus,
fractal roughness) receding into mist, drifting mist bands between them, the
water (mirrored mountains, a glade of glitter under the light, slow ripple
lines), the longship's reflection (cut into horizontal strips that sway
independently), the wake, and the ship.

The ship is drawn from hand-designed parametric curves at Gokstad/Oseberg
proportions (23 m hull, low freeboard amidships, stems rising to ~5 m and
ending in an Oseberg-style spiral, striped square sail, 21 shields on the rail,
14 oars a side; clinker strakes, stays, shrouds and sheets as hairlines).

Motion (all SMIL, seamless):
* the ship crosses the fjord once per LOOP seconds and re-enters off-screen;
  it starts mid-frame so a fresh page load already shows it;
* the oars row in unison (catch, drive, release, feathered recovery); the
  stroke speed matches the ship speed so blades "grip" the water;
* each catch leaves a puddle ring that stays put in the world while the ship
  moves on; a Kelvin wake (featherlets on both arms) and a stern trail follow;
* the reflection's strips sway, glitter twinkles, mist drifts, a waterfall
  runs down one cliff.

Sail variants (this copy weaves the one named by SAIL below):
* a: the Game of Life glider, as dyed roundels in a corded 3x3 grid;
* b: a striped sail whose stripe cells are the bits of an ASCII word;
* c: a complete binary tree, root at the top, couched in wool cord.

Usage: python generate.py  -> banner-dark.svg, banner-light.svg
"""
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# which sail to weave: 'a' glider, 'b' bits, 'c' tree (overridable via $SAIL)
SAIL = os.environ.get('SAIL', 'a')

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


# ------------------------------------------------------------------ sail
# The sail's outline is unchanged from the original banner.  Its cloth is
# described in a unit square (u across, v down) and laid onto the billowing
# sail with a Coons patch built from the four edge curves, so every stripe,
# panel and appliqué follows the belly of the cloth.
def _sail_geom():
    sx0, sx1 = MAST_X - SAIL_HALF, MAST_X + SAIL_HALF
    k = (SAIL_BOT - YARD_Y) / 71.0
    yc_hi = YARD_Y + 15 * k            # -86 in the original
    yc_r = SAIL_BOT - 22 * k           # -52
    yc_l = SAIL_BOT - 26 * k           # -56
    top_l, top_r = (sx0 - 1, YARD_Y + 1.2), (sx1 + 1, YARD_Y + 1.2)
    bot_l, bot_r = (sx0 + 3, SAIL_BOT), (sx1 + 1, SAIL_BOT + 1)
    foot_c = (MAST_X + 2, SAIL_BOT - 9)
    path = (f'M{f(top_l[0])} {f(top_l[1])}L{f(top_r[0])} {f(top_r[1])}'
            f'C{f(sx1 + 6)} {f(yc_hi)} {f(sx1 + 6)} {f(yc_r)} {f(bot_r[0])} {f(bot_r[1])}'
            f'Q{f(foot_c[0])} {f(foot_c[1])} {f(bot_l[0])} {f(bot_l[1])}'
            f'C{f(sx0 + 1)} {f(yc_l)} {f(sx0 - 3)} {f(yc_hi)} {f(top_l[0])} {f(top_l[1])}Z')
    A = np.array
    P00, P10, P01, P11 = A(top_l), A(top_r), A(bot_l), A(bot_r)
    R = (P10, A((sx1 + 6, yc_hi)), A((sx1 + 6, yc_r)), P11)
    L = (P00, A((sx0 - 3, yc_hi)), A((sx0 + 1, yc_l)), P01)
    Bq = (P01, A(foot_c), P11)

    def cub(c, t):
        t = t[:, None]
        return ((1 - t) ** 3 * c[0] + 3 * (1 - t) ** 2 * t * c[1]
                + 3 * (1 - t) * t ** 2 * c[2] + t ** 3 * c[3])

    def mp(u, v):
        u = np.asarray(u, float)
        v = np.asarray(v, float)
        uu, vv = u[:, None], v[:, None]
        top = P00 + (P10 - P00) * uu
        bot = (1 - uu) ** 2 * Bq[0] + 2 * (1 - uu) * uu * Bq[1] + uu ** 2 * Bq[2]
        lef, rig = cub(L, v), cub(R, v)
        corner = ((1 - uu) * (1 - vv) * P00 + uu * (1 - vv) * P10
                  + (1 - uu) * vv * P01 + uu * vv * P11)
        return (1 - vv) * top + vv * bot + (1 - uu) * lef + uu * rig - corner

    return dict(path=path, mp=mp, bot_l=bot_l, bot_r=bot_r, sx0=sx0, sx1=sx1)


SAIL_GEOM = _sail_geom()


def _dense(pts, closed=True, step=0.04):
    """Subdivide a (u, v) polyline so straight cloth edges bend with the sail."""
    pts = list(pts)
    if closed:
        pts = pts + [pts[0]]
    out = []
    for (a, b), (c, d) in zip(pts[:-1], pts[1:]):
        n = max(1, int(math.ceil(max(abs(c - a), abs(d - b)) / step)))
        for i in range(n):
            out.append((a + (c - a) * i / n, b + (d - b) * i / n))
    if not closed:
        out.append(pts[-1])
    return out


def sail_poly(pts, closed=True):
    """(u, v) polygon -> path data on the sail."""
    q = _dense(pts, closed)
    xy = SAIL_GEOM['mp']([p[0] for p in q], [p[1] for p in q])
    return poly(xy, closed)


def sail_rect(u0, v0, u1, v1):
    return sail_poly([(u0, v0), (u1, v0), (u1, v1), (u0, v1)])


def sail_line(pts):
    return sail_poly(pts, closed=False)


def sail_svg(th, uid):
    sail = SAIL_GEOM['path']
    S = th['sails'][SAIL]
    out = [f'<clipPath id="sailclip{uid}"><path d="{sail}"/></clipPath>',
           f'<path d="{sail}" fill="url(#sailg{uid})"/>',
           f'<g clip-path="url(#sailclip{uid})">']
    out.append(SAIL_MOTIFS[SAIL](S))
    # light across the belly of the cloth: darker at the luffs, lit in the middle
    out.append(f'<path d="{sail}" fill="url(#sailshade{uid})"/>')
    out.append('</g>')
    out.append(f'<path d="{sail}" fill="none" stroke="{th["sail_edge"]}" stroke-width=".7"/>')
    return ''.join(out)


def sail_defs(th, uid):
    G = SAIL_GEOM
    x0, x1 = G['sx0'] - 4, G['sx1'] + 6
    sh = th['sail_shade']
    return (f'<linearGradient id="sailshade{uid}" x1="{f(x0)}" y1="0" x2="{f(x1)}" y2="0" gradientUnits="userSpaceOnUse">'
            f'<stop offset="0" stop-color="{sh}" stop-opacity="{th["shade_op"]}"/>'
            f'<stop offset=".22" stop-color="{sh}" stop-opacity="0"/>'
            f'<stop offset=".62" stop-color="{sh}" stop-opacity="0"/>'
            f'<stop offset="1" stop-color="{sh}" stop-opacity="{th["shade_op"] * 1.3:.3f}"/></linearGradient>')


def _seams(S, us=(), vs=(), op=None):
    """Thin stitched seams between cloth panels."""
    d = ''.join(sail_line([(-0.05, v), (1.05, v)]) for v in vs)
    d += ''.join(sail_line([(u, -0.05), (u, 1.05)]) for u in us)
    return (f'<path d="{d}" stroke="{S["seam"]}" stroke-width="{S.get("seam_w", .5)}" fill="none" '
            f'opacity="{op if op is not None else S["seam_op"]}"/>')


def _field(S):
    if not S.get('field'):
        return ''          # keep the original cloth gradient
    return f'<path d="{sail_rect(-.1, -.1, 1.1, 1.1)}" fill="{S["field"]}"/>'


# --- a: the glider ----------------------------------------------------------
GLIDER = ((0, 1, 0),
          (0, 0, 1),
          (1, 1, 1))


def motif_glider(S):
    """Undyed sail with a madder-red edging, divided by couched cords into a
    3x3 grid; five cells carry a dyed appliqué: the Game of Life glider, laid
    out exactly as the hacker emblem."""
    out = [_field(S)]
    hem = S['hem']
    eu = [hem + (1 - 2 * hem) * i / 3 for i in range(4)]
    ev = [hem + (1 - 2 * hem) * i / 3 for i in range(4)]
    cell = (SAIL_BOT - YARD_Y) * (1 - 2 * hem) / 3
    cu = [(eu[c] + eu[c + 1]) / 2 for r in range(3) for c in range(3) if GLIDER[r][c]]
    cv = [(ev[r] + ev[r + 1]) / 2 for r in range(3) for c in range(3) if GLIDER[r][c]]
    dots = ''.join(f'<circle cx="{f(x)}" cy="{f(y)}" r="{S["dot_r"] * cell:.1f}"/>'
                   for x, y in SAIL_GEOM['mp'](cu, cv))
    out.append(f'<g fill="{S["on"]}">{dots}</g>')
    # edging: a dyed band all round the sail
    if hem > 0:
        ring = sail_rect(-.1, -.1, 1.1, 1.1) + sail_rect(hem, hem, 1 - hem, 1 - hem)
        out.append(f'<path d="{ring}" fill="{S["on"]}" fill-rule="evenodd"/>')
    # cords of the grid
    g = ''.join(sail_line([(u, ev[0]), (u, ev[-1])]) for u in eu[1:-1])
    g += ''.join(sail_line([(eu[0], v), (eu[-1], v)]) for v in ev[1:-1])
    out.append(f'<path d="{g}" stroke="{S["cord"]}" stroke-width="{S["cord_w"]}" fill="none" '
               f'opacity="{S["cord_op"]}"/>')
    return ''.join(out)


# --- b: bits ----------------------------------------------------------------
WORD = 'hei'


def motif_bits(S):
    """Striped sail whose stripes are bits: one cloth band per letter, eight
    stripe cells per band, MSB at the left luff; a dyed stripe cell is a 1."""
    out = [_field(S)]
    rows = [format(ord(ch), '08b') for ch in WORD]
    nb, nr = 8, len(rows)
    d = ''
    for r, bits in enumerate(rows):
        v0 = r / nr - (0.1 if r == 0 else 0)
        v1 = (r + 1) / nr + (0.1 if r == nr - 1 else 0)
        c = 0
        while c < nb:
            if bits[c] == '1':
                e = c
                while e + 1 < nb and bits[e + 1] == '1':
                    e += 1
                u0 = c / nb - (0.1 if c == 0 else 0)
                u1 = (e + 1) / nb + (0.1 if e == nb - 1 else 0)
                d += sail_rect(u0, v0, u1, v1)
                c = e + 1
            else:
                c += 1
    out.append(f'<path d="{d}" fill="{S["on"]}" opacity="{S["on_op"]}"/>')
    out.append(_seams(S, vs=[r / nr for r in range(1, nr)]))
    out.append(_seams(S, us=[c / nb for c in range(1, nb)], op=S['col_seam_op']))
    return ''.join(out)


# --- c: the tree ------------------------------------------------------------
TREE_DEPTH = 3


def motif_tree(S):
    """Yggdrasil drawn the way computer scientists draw trees: a complete
    binary tree, root at the top, couched in wool cord on the cloth."""
    out = [_field(S)]
    style = S.get('style', 'both')
    if style in ('stripes', 'both'):
        nst = 7
        d = ''.join(sail_rect(i / nst, -.1, (i + 1) / nst, 1.1) for i in range(1, nst, 2))
        out.append(f'<path d="{d}" fill="{S["stripe"]}" opacity="{S["stripe_op"]}"/>')
    if style in ('border', 'both'):
        hem = S['hem']
        out.append(f'<path d="{sail_rect(hem, hem * 1.25, 1 - hem, 1 - hem * 1.25)}" fill="none" '
                   f'stroke="{S["cord"]}" stroke-width="{S["border_w"]}"/>')
    D = TREE_DEPTH
    top, bot = S['tree_top'], S['tree_bot']
    side = S['tree_side']
    nodes = {}
    for lev in range(D + 1):
        n = 2 ** lev
        for i in range(n):
            # leaves evenly spaced; parents centred over their children
            u = side + (1 - 2 * side) * ((i + 0.5) * 2 ** (D - lev) - 0.5) / (2 ** D - 1)
            v = top + (bot - top) * lev / D
            nodes[(lev, i)] = (u, v)
    G = SAIL_GEOM
    edges = ''
    for lev in range(D):
        for i in range(2 ** lev):
            a = nodes[(lev, i)]
            for ch in (2 * i, 2 * i + 1):
                edges += sail_line([a, nodes[(lev + 1, ch)]])
    out.append(f'<path d="{edges}" stroke="{S["cord"]}" stroke-width="{S["cord_w"]}" '
               f'stroke-linecap="round" fill="none"/>')
    xy = G['mp']([p[0] for p in nodes.values()], [p[1] for p in nodes.values()])
    rs = S['node_r']
    circ = ''.join(f'<circle cx="{f(x)}" cy="{f(y)}" r="{rs[lev]}"/>'
                   for (lev, _), (x, y) in zip(nodes.keys(), xy))
    out.append(f'<g fill="{S["node"]}" stroke="{S["cord"]}" stroke-width="{S["node_sw"]}">{circ}</g>')
    return ''.join(out)


SAIL_MOTIFS = {'a': motif_glider, 'b': motif_bits, 'c': motif_tree}


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
    # sail: braced slightly, billowing; the cloth pattern is the variant's motif
    bot_l, bot_r = SAIL_GEOM['bot_l'], SAIL_GEOM['bot_r']
    p.append(sail_svg(th, uid))
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


# ------------------------------------------------------------------ scene
def theme_svg(th, uid):
    xs = np.arange(-4, W + 5, 2.5)
    parts = []
    defs = []

    # gradients
    defs.append(f'<linearGradient id="sky{uid}" x1="0" y1="0" x2="0" y2="{YH}" gradientUnits="userSpaceOnUse">'
                + ''.join(f'<stop offset="{o}" stop-color="{c}"/>' for o, c in th['sky']) + '</linearGradient>')
    defs.append(f'<radialGradient id="glow{uid}" cx="{LIGHT_X}" cy="{LIGHT_Y}" r="330" gradientUnits="userSpaceOnUse" '
                f'gradientTransform="translate({LIGHT_X} {LIGHT_Y}) scale(1.25 .6) translate({-LIGHT_X} {-LIGHT_Y})">'
                f'<stop offset="0" stop-color="{th["glow"]}" stop-opacity="{th["glow_op"]}"/>'
                f'<stop offset=".35" stop-color="{th["glow"]}" stop-opacity="{th["glow_op"] * .35:.3f}"/>'
                f'<stop offset="1" stop-color="{th["glow"]}" stop-opacity="0"/></radialGradient>')
    defs.append(f'<radialGradient id="halo{uid}" cx="{LIGHT_X}" cy="{LIGHT_Y}" r="46" gradientUnits="userSpaceOnUse">'
                f'<stop offset=".3" stop-color="{th["halo"]}" stop-opacity="{th["halo_op"]}"/>'
                f'<stop offset="1" stop-color="{th["halo"]}" stop-opacity="0"/></radialGradient>')
    defs.append(f'<linearGradient id="water{uid}" x1="0" y1="{YH}" x2="0" y2="{H}" gradientUnits="userSpaceOnUse">'
                + ''.join(f'<stop offset="{o}" stop-color="{c}"/>' for o, c in th['water']) + '</linearGradient>')
    defs.append(f'<linearGradient id="sailg{uid}" x1="0" y1="{YARD_Y}" x2="0" y2="{SAIL_BOT}" gradientUnits="userSpaceOnUse">'
                f'<stop offset="0" stop-color="{th["sail_top"]}"/><stop offset="1" stop-color="{th["sail_bot"]}"/></linearGradient>')
    defs.append(sail_defs(th, uid))
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
    defs.append(f'<linearGradient id="reflfade{uid}" x1="0" y1="{YH}" x2="0" y2="{H}" gradientUnits="userSpaceOnUse">'
                f'<stop offset="0" stop-color="#fff" stop-opacity="1"/>'
                f'<stop offset="1" stop-color="#fff" stop-opacity=".25"/></linearGradient>')

    # sky + light
    parts.append(f'<rect width="{W}" height="{YH + 2}" fill="url(#sky{uid})"/>')
    parts.append(f'<rect width="{W}" height="{H}" fill="url(#glow{uid})"/>')
    parts.append(f'<circle cx="{LIGHT_X}" cy="{LIGHT_Y}" r="46" fill="url(#halo{uid})"/>')
    parts.append(f'<circle cx="{LIGHT_X}" cy="{LIGHT_Y}" r="{LIGHT_R}" fill="{th["disc"]}"/>')
    if th.get('moon_shade'):
        # faint maria for the moon
        parts.append(f'<g fill="{th["moon_shade"]}" opacity=".5">'
                     f'<circle cx="{LIGHT_X - 4}" cy="{LIGHT_Y - 4}" r="4.5"/>'
                     f'<circle cx="{LIGHT_X + 5}" cy="{LIGHT_Y + 3}" r="3.2"/>'
                     f'<circle cx="{LIGHT_X - 2}" cy="{LIGHT_Y + 7}" r="2.2"/></g>')

    if th.get('stars'):
        parts.append(stars_svg(th))
    # mountains with mist at their feet
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
                    f'<stop offset=".55" stop-color="{col}"/>'
                    f'<stop offset="1" stop-color="{mix(col, mist, th["mist_mix"][i])}"/></linearGradient>')
        parts.append(f'<path d="{layer_path(y, xs)}" fill="url(#{gid})"/>')
        # rim light on the ridge
        if th['ridge_op'][i] > 0:
            m = y < YH - 3
            seg = []
            run = []
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
    parts.append(f'<rect y="{YH - 10}" width="{W}" height="22" fill="url(#shoremist{uid})"/>')
    defs.append(f'<linearGradient id="shoremist{uid}" x1="0" y1="{YH - 10}" x2="0" y2="{YH + 12}" gradientUnits="userSpaceOnUse">'
                f'<stop offset="0" stop-color="{th["mist"]}" stop-opacity="0"/>'
                f'<stop offset=".5" stop-color="{th["mist"]}" stop-opacity="{th["shore_op"]}"/>'
                f'<stop offset="1" stop-color="{th["mist"]}" stop-opacity="0"/></linearGradient>')
    parts.append(mist_band(th, uid, 'fg'))
    # moon glade
    parts.append(glade_svg(th, uid, defs))
    # ripple lines
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


THEMES = {
    'dark': dict(
        bg='#0d1117', border='#30363d',
        sky=[(0, '#0a0f17'), (.55, '#111a27'), (1, '#1d2838')],
        glow='#b9c6d8', glow_op=.20, halo='#e8e6dc', halo_op=.30, disc='#e9e5d8',
        moon_shade=None, stars=True, star='#c9d3e0', fall='#8696ab', fall_op=.55,
        mtn=ramp('#303d50', '#0c1219', 7, 1.25),
        mist='#3c4a5e', mist_mix=[.5, .55, .55, .6, .6, .62, .65], mist_op=1.0, mist_after=[0, 1, 2, 3, 4],
        ridge='#9fb0c6', ridge_op=[.38, .33, .29, .26, .23, .21, .19],
        water=[(0, '#1a2433'), (.4, '#121a26'), (1, '#0b1018')], water_mid='#121a26',
        refl_mix=.35, refl_op=.75, shore_op=.55,
        glade='#e6e1d2', glade_w=1.0, glade_op=.75, col_op=.18,
        ripple='#5b6b82', ripple_op=.35,
        ship='#06090d', rig='#3a4656', rig_w=.55, strake='#1f2833', rim='#4e5d72',
        shield_a='#151c26', shield_b='#26303d', shield_rim='#4a586b', boss='#5d6b7e',
        oar='#0a0e14',
        sail_top='#33292a', sail_bot='#241d20', stripe='#61392f', stripe_op=.8, seam='#000000',
        sail_edge='#4e3a34', sail_shade='#000000', shade_op=.28,
        sails=dict(
            a=dict(field='#2d2526', on='#6c3b30', dot_r=.34, hem=.07, cord='#6c3b30', cord_w=.8, cord_op=1),
            b=dict(field=None, on='#61392f', on_op=.8, seam='#000000', seam_op=.45, col_seam_op=.22),
            c=dict(field=None, cord='#9a8574', cord_w=1.6, node='#9a8574', node_sw=0,
                   node_r=(3.2, 2.7, 2.3, 2.0), tree_top=.17, tree_bot=.80, tree_side=.10,
                   style='both', hem=.05, border_w=.9, stripe='#61392f', stripe_op=.35),
        ),
        wake='#8394ab', refl_ship='#05080c', refl_sail='#24191b', ship_refl_op=.85, gap=1.0,
    ),
    'light': dict(
        bg='#ffffff', border='#d0d7de',
        sky=[(0, '#e4e8ec'), (.6, '#eef0f0'), (1, '#f6f3ec')],
        glow='#f3dcc0', glow_op=.55, halo='#fbefe0', halo_op=.8, disc='#fbf6ec',
        moon_shade=None, fall='#ffffff', fall_op=.9,
        mtn=ramp('#d3d8dd', '#56616e', 7, 0.9),
        mist='#f1f1ee', mist_mix=[.5, .55, .55, .6, .6, .62, .62], mist_op=1.0, mist_after=[0, 1, 2, 3, 4],
        ridge='#3d4752', ridge_op=[.2, .2, .22, .22, .24, .24, .25],
        water=[(0, '#eceae4'), (.4, '#e5e6e5'), (1, '#dfe2e5')], water_mid='#e3e5e6',
        refl_mix=.42, refl_op=.8, shore_op=.6,
        glade='#ffffff', glade_w=1.1, glade_op=.95, col_op=.55,
        ripple='#7d8894', ripple_op=.30,
        ship='#1b222a', rig='#4a5561', rig_w=.6, strake='#38424d', rim='#4c5865',
        shield_a='#2a333d', shield_b='#5a6570', shield_rim='#1b222a', boss='#c9cfd5',
        oar='#1f262e',
        sail_top='#93604f', sail_bot='#835546', stripe='#ece4d8', stripe_op=.62, seam='#3a2a25',
        sail_edge='#5a3b32', sail_shade='#3a2a25', shade_op=.13,
        sails=dict(
            a=dict(field='#e2d6c8', on='#8e5341', dot_r=.34, hem=.07, cord='#8e5341', cord_w=.8, cord_op=1),
            b=dict(field=None, on='#ece4d8', on_op=.62, seam='#3a2a25', seam_op=.45, col_seam_op=.2),
            c=dict(field=None, cord='#efe7db', cord_w=1.6, node='#efe7db', node_sw=0,
                   node_r=(3.2, 2.7, 2.3, 2.0), tree_top=.17, tree_bot=.80, tree_side=.10,
                   style='both', hem=.05, border_w=.9, stripe='#a87160', stripe_op=.45),
        ),
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
    path = os.path.join(HERE, f'banner-{name}.svg')
    with open(path, 'w') as fh:
        fh.write(svg)
    print(path, len(svg.encode()) // 1024, 'KB')


if __name__ == '__main__':
    for n in ('dark', 'light'):
        build(n)
