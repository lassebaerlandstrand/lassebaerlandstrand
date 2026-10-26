#!/usr/bin/env python3
"""Animated PCB banner: a QFP fans its pins out into a calm, routed board.

Routing
-------
All copper runs on 0/45/90-degree centrelines. Each side of the chip fans out
with 45-degree doglegs whose diagonals are parallel and evenly spaced (outer
traces bend first), so a fan can never self-intersect; the corner pins get
short dog-bone stubs to fan-out vias. Lanes to the left end on vias that step
back in a wedge, which leaves the outer corners of the board empty; the two
outermost lanes on each side drop through their via and leave on the bottom
layer. Every corner is filleted to fit its segments, and a numeric design-rule
check (sampled centrelines, spatial hash) reports the smallest copper gap
between different nets on each layer.

Look
----
Copper is drawn in Gerber "outline" mode: every trace, pad and via is stroked
wide in the line colour and then again, narrower, in the background colour, so
the union of all copper appears as thin contour lines with hollow channels.

Motion
------
Signals are comet-like dashes that travel inside the channels: stacked <use>
copies of the trace whose dash period is a little longer than the trace, so
only one dash is ever on the copper. Its stroke-dashoffset grows by exactly
one period per pulse at constant speed, so the next dash is already parked
before the start and the loop (and pulses that straddle it) wrap invisibly.
Pulses leave the chip on a slow beat that walks outward through the lanes,
the lower half echoing the upper a moment later; the chip core warms on each
beat, vias glow when a dash arrives, and the outer lanes hand their dash on to
a cooler one on the bottom layer. On load a one-off power-up floods every
channel with copper light from the chip outward before the pulses fade in.

Usage:  python generate.py      -> banner-dark.svg, banner-light.svg
"""
import math
import os
import random

from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont

HERE = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------------ constants
W, H = 1200, 340
CX, CY = 842, 170           # chip centre
BODY = 46                   # half body size
NPIN = 8                    # pins per side
PITCH = 10                  # pin pitch
PAD_L, PAD_W = 12, 5.4      # pad length / width
PAD_C = BODY + 8            # pad centre distance from chip centre
TW = float(os.environ.get('TW', 4.6))   # signal trace width
TW_B = 2.6                  # bottom-layer trace width
CLEAR = 5.0                 # min copper-to-copper gap
LANE = 20                   # lane pitch after fan-out
VIA_R, VIA_HOLE = 5.2, 2.2
R_BEND = 7.0                # default fillet radius
DIAG = 11.0                 # perpendicular spacing of parallel fan diagonals
FAN_X = 30                  # straight run before the first side-fan bend
TOP_RISE = 12               # straight rise before the first top-fan bend
DOG = 11                    # dog-bone diagonal length
WEDGE_X0, WEDGE_DX = 210, 330   # left via wedge: innermost x, spread
CAP_X, CAP_P = 560, 6.0     # AC-coupling caps on the left pair
FIDUCIALS = [(38, 38), (1162, 302)]
INTRO = int(os.environ.get('INTRO', 1))
INTRO_SPEED = 420.0
INTRO_END = 2.6             # pulses fade in from here
STYLE = os.environ.get('STYLE', 'outline')
POUR = int(os.environ.get('POUR', 0))
OL_W = 1.0                  # outline-mode line width
POUR_GAP = 4.0              # pour-to-copper clearance
POUR_IN = 12                # pour inset from the board edge
STITCH = 30                 # stitching-via grid pitch
STITCH_R = 1.7
EXIT_X = 1000               # right-hand top/bottom lanes leave the board here

BEAT, BEAT0 = 2.0, 0.35
ORDER = [0, 4, 1, 6, 0, 2, 5, None]   # lane level (0 = centre) fired per beat
LAG = 0.22                  # lower half echoes the upper half slightly late
T = 16.0                    # loop length, s
SPEED = 150.0               # px / s
SEED = int(os.environ.get('SEED', 3))


def f(v):
    s = f'{v:.1f}'
    if s.endswith('.0'):
        s = s[:-2]
    return '0' if s in ('-0', '') else s


MONO = '/usr/share/fonts/fonts-go/Go-Mono.ttf'
_font = TTFont(MONO)
_gs = _font.getGlyphSet()
_cmap = _font.getBestCmap()
_upem = _font['head'].unitsPerEm


def text_d(text, px, x, y, anchor='start'):
    """Go Mono outlines for silkscreen text; (x, y) is the baseline origin."""
    s = px / _upem
    adv = _font['hmtx'][_cmap[ord('0')]][0] * s
    if anchor == 'middle':
        x -= adv * len(text) / 2
    pen = SVGPathPen(_gs, f)
    for i, ch in enumerate(text):
        _gs[_cmap[ord(ch)]].draw(TransformPen(pen, (s, 0, 0, -s, x + i * adv, y)))
    return pen.getCommands()


# ------------------------------------------------------------------ geometry
def norm(dx, dy):
    l = math.hypot(dx, dy)
    return dx / l, dy / l


class Trace:
    """A filleted polyline on 0/45/90 directions."""

    def __init__(self, pts, layer='top', radius=R_BEND, radii=None, net=None):
        self.pts = [(float(x), float(y)) for x, y in pts]
        self.layer = layer
        self.net = net
        self.radius = radius
        self.radii = radii or {}
        self._build()

    def _build(self):
        P = self.pts
        n = len(P)
        for i in range(n - 1):
            dx, dy = P[i + 1][0] - P[i][0], P[i + 1][1] - P[i][1]
            assert abs(dx) < 1e-6 or abs(dy) < 1e-6 or abs(abs(dx) - abs(dy)) < 1e-6, \
                f'non-45 segment {P[i]} -> {P[i+1]}'
        seglen = [math.dist(P[i], P[i + 1]) for i in range(n - 1)]
        # tangent length available at each corner
        d = [f'M{f(P[0][0])} {f(P[0][1])}']
        samples = []
        length = 0.0
        cur = P[0]

        def line_to(q):
            nonlocal cur, length
            L = math.dist(cur, q)
            if L < 1e-9:
                return
            k = max(1, int(L / 1.0))
            for j in range(1, k + 1):
                t = j / k
                samples.append((cur[0] + (q[0] - cur[0]) * t, cur[1] + (q[1] - cur[1]) * t))
            length += L
            d.append(f'L{f(q[0])} {f(q[1])}')
            cur = q

        samples.append(P[0])
        for i in range(1, n - 1):
            u = norm(P[i][0] - P[i - 1][0], P[i][1] - P[i - 1][1])
            v = norm(P[i + 1][0] - P[i][0], P[i + 1][1] - P[i][1])
            cross = u[0] * v[1] - u[1] * v[0]
            dot = u[0] * v[0] + u[1] * v[1]
            th = math.atan2(abs(cross), dot)
            if th < 1e-6:
                continue
            r = self.radii.get(i, self.radius)
            avail_in = seglen[i - 1] * (1.0 if i == 1 else 0.5)
            avail_out = seglen[i] * (1.0 if i == n - 2 else 0.5)
            tl = r * math.tan(th / 2)
            tl = min(tl, avail_in, avail_out)
            r = tl / math.tan(th / 2)
            a = (P[i][0] - u[0] * tl, P[i][1] - u[1] * tl)
            b = (P[i][0] + v[0] * tl, P[i][1] + v[1] * tl)
            line_to(a)
            # arc
            sgn = 1 if cross > 0 else -1
            nx, ny = -u[1] * sgn, u[0] * sgn            # normal toward turn side
            c = (a[0] + nx * r, a[1] + ny * r)
            a0 = math.atan2(a[1] - c[1], a[0] - c[0])
            k = max(2, int(r * th / 1.0))
            for j in range(1, k + 1):
                ang = a0 + sgn * th * j / k
                samples.append((c[0] + r * math.cos(ang), c[1] + r * math.sin(ang)))
            length += r * th
            d.append(f'A{f(r)} {f(r)} 0 0 {1 if cross > 0 else 0} {f(b[0])} {f(b[1])}')
            cur = b
        line_to(P[-1])
        self.d = ''.join(d)
        self.samples = samples
        self.length = length

    def pos_at(self, s):
        """Arc-length position -> index into samples (approx)."""
        acc = 0.0
        for i in range(1, len(self.samples)):
            acc += math.dist(self.samples[i - 1], self.samples[i])
            if acc >= s:
                return self.samples[i]
        return self.samples[-1]


def mirror_pts(pts, mx, my):
    out = []
    for x, y in pts:
        out.append((2 * CX - x if mx else x, 2 * CY - y if my else y))
    return out


# ------------------------------------------------------------------ layout
def chip_pins():
    """Pad centres per side, ordered by offset -35..35."""
    offs = [(-(NPIN - 1) / 2 + i) * PITCH for i in range(NPIN)]
    return offs


def fan_quadrant():
    """Traces of the upper-left quadrant, from pad centres into their lanes.

    Returns (pts, lane_y, kind) in absolute coordinates; for lane traces pts
    end where the trace has settled in its lane heading -x.  Kinds: 'pair'
    (straight differential pair), 'side' / 'top' (fanned lanes) and 'dog'
    (short dog-bone stub to a fan-out via next to the chip)."""
    out = []
    x_pad = CX - PAD_C
    y_pad = CY - PAD_C
    s2 = DIAG * math.sqrt(2)
    # --- left-side pins, upper half (inner -> outer): -5 pair, -15, -25 lanes
    out.append(([(x_pad, CY - 5), (x_pad - 60, CY - 5)], CY - 5, 'pair'))
    c = None
    for o, ln in ((-15, -25), (-25, -45)):
        y0, y1 = CY + o, CY + ln
        c = (x_pad - FAN_X) - y0 if c is None else c + s2   # x - y = c
        out.append(([(x_pad, y0), (c + y0, y0), (c + y1, y1), (c + y1 - 10, y1)], y1, 'side'))
    # outermost left pin: dog-bone towards the chip corner
    y0 = CY - 35
    out.append(([(x_pad, y0), (x_pad - 9, y0), (x_pad - 9 - DOG, y0 - DOG)], None, 'dog'))
    # --- top-side pins, left half: -25, -15, -5 turn left into lanes above
    c = None
    for o, ln in ((-25, -85), (-15, -105), (-5, -125)):
        x0, y1 = CX + o, CY + ln
        c = x0 - (y_pad - TOP_RISE) if c is None else c + s2   # x - y = c
        yb = x0 - c
        out.append(([(x0, y_pad), (x0, yb), (c + y1, y1), (c + y1 - 10, y1)], y1, 'top'))
    return out


def build_layout():
    rng = random.Random(SEED)
    traces = []          # dicts: tr, side, kind, dist, via, lvl
    vias = []            # (x, y)
    q = fan_quadrant()
    for pts, ly, kind in q:
        for mx in (False, True):
            for my in (False, True):
                p = mirror_pts(pts, mx, my)
                side = 'R' if mx else 'L'
                if kind == 'dog':
                    traces.append(dict(tr=Trace(p, radius=4), side=side, kind=kind,
                                       dist=0, via=p[-1]))
                    vias.append(p[-1])
                    continue
                y = 2 * CY - ly if my else ly
                dist = abs(y - CY)
                X = (lambda x: 2 * CX - x) if mx else (lambda x: x)
                if kind == 'pair':
                    p = p[:-1] + [(-20 if side == 'L' else W + 20, y)]
                    t = dict(side=side, kind=kind, dist=dist)
                elif side == 'L':
                    # wedge: inner lanes run further left
                    frac = (dist - 25) / (125 - 25)
                    xe = round(WEDGE_X0 + WEDGE_DX * frac ** 0.85 + rng.uniform(-12, 12))
                    p = p[:-1] + [(xe, y)]
                    t = dict(side=side, kind=kind, dist=dist, via=(xe, y))
                    vias.append((xe, y))
                    if dist >= 105:
                        # the outer lanes drop through their via and leave
                        # the board on the bottom layer, towards the corner
                        sgn = -1 if y < CY else 1
                        xb = xe - (30 if dist > 115 else 44)
                        t['cont'] = Trace([(xe, y), (xb, y), (xb - 300, y + sgn * 300)],
                                          layer='bot', radius=10)
                elif kind == 'side':
                    p = p[:-1] + [(W + 20, y)]
                    t = dict(side=side, kind=kind, dist=dist)
                else:
                    sgn = -1 if y < CY else 1
                    xr = round(EXIT_X + (125 - dist) * 1.0)
                    p = p[:-1] + [(xr, y), (xr + 200, y + sgn * 200)]
                    t = dict(side=side, kind=kind, dist=dist)
                t['tr'] = Trace(p)
                traces.append(t)
    for t in traces:
        t['lvl'] = -1 if t['kind'] == 'dog' else (0 if t['kind'] == 'pair' else round((t['dist'] - 5) / 20))
    return traces, vias


def stitch_pts(traces, vias):
    """Stitching vias on a staggered grid, wherever the pour is wide."""
    pts = []
    for t in traces:
        pts += [p for p in t['tr'].samples[::3]]
    pts += list(vias)
    cell = 20
    grid = {}
    for p in pts:
        grid.setdefault((int(p[0] // cell), int(p[1] // cell)), []).append(p)
    out = []
    keep = STITCH * 0.55
    e = PAD_C + PAD_L / 2 + 14
    j = 0
    y = POUR_IN + 12
    while y < H - POUR_IN - 8:
        x = POUR_IN + 12 + (STITCH / 2 if j % 2 else 0)
        while x < W - POUR_IN - 8:
            ok = not (abs(x - CX) < e and abs(y - CY) < e)
            gx, gy = int(x // cell), int(y // cell)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for p in grid.get((gx + dx, gy + dy), []):
                        if ok and math.dist(p, (x, y)) < keep:
                            ok = False
            if ok:
                out.append((x, y))
            x += STITCH
        y += STITCH * math.sqrt(3) / 2
        j += 1
    return out


# ------------------------------------------------------------------ DRC
def drc(traces, vias):
    """Smallest copper gap between different nets on the same layer."""
    items = []
    all_tr = []
    for t in traces:
        all_tr.append(t['tr'])
        if 'cont' in t:
            all_tr.append(t['cont'])
    for i, tr in enumerate(all_tr):
        w = TW if tr.layer == 'top' else TW_B
        for p in tr.samples:
            if -5 < p[0] < W + 5 and -5 < p[1] < H + 5:
                items.append((p[0], p[1], w / 2, i, tr.layer))
    # a via belongs to its top trace and its bottom continuation
    owner = {}
    for i, tr in enumerate(all_tr):
        owner.setdefault((round(tr.pts[-1][0]), round(tr.pts[-1][1])), []).append(i)
        owner.setdefault((round(tr.pts[0][0]), round(tr.pts[0][1])), []).append(i)
    for (x, y) in vias:
        ids = owner[(round(x), round(y))]
        for k in range(24):
            a = 2 * math.pi * k / 24
            for lay in ('top', 'bot'):
                items.append((x + VIA_R * math.cos(a), y + VIA_R * math.sin(a), 0, tuple(ids), lay))
    cell = 12
    grid = {}
    for it in items:
        grid.setdefault((int(it[0] // cell), int(it[1] // cell)), []).append(it)
    worst = (1e9, None)

    def same(a, b):
        ia = a if isinstance(a, tuple) else (a,)
        ib = b if isinstance(b, tuple) else (b,)
        return bool(set(ia) & set(ib))

    for (gx, gy), lst in grid.items():
        neigh = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                neigh += grid.get((gx + dx, gy + dy), [])
        for a in lst:
            for b in neigh:
                if a[4] != b[4] or same(a[3], b[3]):
                    continue
                gap = math.hypot(a[0] - b[0], a[1] - b[1]) - a[2] - b[2]
                if gap < worst[0]:
                    worst = (gap, (a[:2], b[:2]))
    return worst


# ------------------------------------------------------------------ motion
def schedule(traces):
    """Pulse start times per trace.  A slow beat walks outward from the
    centre of the fan, top and bottom mirrored; the right side answers half a
    beat later."""
    beat, order = BEAT, ORDER
    for t in traces:
        t['pulses'] = []
    for b, lvl in enumerate(order):
        t0 = BEAT0 + b * beat
        for t in traces:
            if t['lvl'] != lvl:
                continue
            lag = LAG if (t['kind'] != 'pair' and t['tr'].pts[0][1] > CY) else 0.0
            t['pulses'].append(t0 + lag + (beat / 2 if t['side'] == 'R' else 0.0))


def kt(x):
    return f'{x / T:.4f}'.rstrip('0').rstrip('.') if x > 0 else '0'


DASHES = [  # (dash length, width, opacity, colour key); stacked into a comet
    (90, TW - 1.6, 0.16, 'glow'),
    (60, TW - 1.6, 0.20, 'glow'),
    (38, TW - 1.4, 0.26, 'glow'),
    (22, TW - 1.2, 0.40, 'glow'),
    (11, TW - 1.0, 0.85, 'pulse'),
    (22, TW + 7.0, 0.07, 'glow'),
]
MARGIN = 16


def pulse_anim(length, starts, speed):
    """Dash period, travel time and head-position keyframes over one loop.

    The head position s only ever grows; one pulse advances it by exactly one
    dash period, so any multiple of the period looks identical and a pulse
    may wrap around the loop boundary."""
    dmax = max(d for d, *_ in DASHES)
    period = length + dmax + 2 * MARGIN
    dur = period / speed
    starts = sorted(t % T for t in starts)
    wrap = [t for t in starts if t + dur > T]
    assert len(wrap) <= 1
    normal = [t for t in starts if t + dur <= T]
    p0 = (T - wrap[0]) * speed if wrap else 0.0      # progress made before 0
    s = -MARGIN
    if wrap:
        keys = [(0.0, s + period - (period - p0))]
        te = wrap[0] + dur - T
        s += period
        keys.append((te, s))
    else:
        keys = [(0.0, s)]
    for t0 in normal:
        assert t0 >= keys[-1][0] - 1e-6, 'overlapping pulses'
        if t0 > keys[-1][0] + 1e-6:
            keys.append((t0, s))
        s += period
        keys.append((t0 + dur, s))
    if wrap:
        assert wrap[0] >= keys[-1][0] - 1e-6, 'overlapping pulses'
        if wrap[0] > keys[-1][0] + 1e-6:
            keys.append((wrap[0], s))
        keys.append((T, s + (T - wrap[0]) * speed))
    elif keys[-1][0] < T - 1e-6:
        keys.append((T, s))
    return period, dur, keys


# ------------------------------------------------------------------ render
THEMES = {
    'dark': dict(
        bg='#0d1117', border='#30363d', pour='#121920', stitch_op=0.35, intro_op=0.5, line='#67768b',
        copper='#a8865f', copper_op=0.50, pad='#b8936a', pad_op=0.75,
        bottom='#3b4a5f', bottom_op=0.22,
        silk='#8796ab', silk_op=0.40,
        pulse='#ffe3b8', glow='#f3b37a', bglow='#9fc0ea',
    ),
    'light': dict(
        bg='#ffffff', border='#d0d7de', pour='#f5f7f9', stitch_op=0.35, intro_op=0.42, line='#8d99a8',
        copper='#5b6573', copper_op=0.55, pad='#5b6573', pad_op=0.7,
        bottom='#c3ccd8', bottom_op=0.35,
        silk='#5b6573', silk_op=0.5,
        pulse='#c8642a', glow='#e08a4c', bglow='#5f87c0',
    ),
}


def anim(attr, keys, values_fmt=f):
    kts = ';'.join(kt(t) for t, _ in keys)
    vals = ';'.join(values_fmt(v) for _, v in keys)
    return (f'<animate attributeName="{attr}" dur="{T:g}s" repeatCount="indefinite" '
            f'keyTimes="{kts}" values="{vals}"/>')


def flash_keys(times, rise=0.12, fall=1.6, peak=1.0):
    """Opacity keyframes for quick-rise / slow-fall flashes, periodic in T
    (a flash that runs past the loop end continues at its start)."""
    def shape(dt):
        dt %= T
        if dt < rise:
            return peak * dt / rise
        if dt < rise + fall:
            return peak * (1 - (dt - rise) / fall)
        return 0.0

    bps = {0.0, T}
    for ta in times:
        for d in (0.0, rise, rise + fall):
            bps.add(round((ta + d) % T, 4))
    bps = sorted(bps)
    return [(t, min(peak, sum(shape(t - ta) for ta in times)))
            for t in bps]


def build_svg(theme, traces, vias):
    th = THEMES[theme]
    defs, base, bot, pulses, vglow = [], [], [], [], []
    n = 0
    via_times = {}

    def add_path(tr):
        nonlocal n
        defs.append(f'<path id="p{n}" d="{tr.d}"/>')
        n += 1
        return f'p{n - 1}'

    def add_pulses(pid, tr, starts, bottom=False):
        period, dur, keys = pulse_anim(tr.length, starts, SPEED)
        for d, w, op, ck in DASHES:
            if bottom:
                w = w * TW_B / TW if w <= TW else w * 0.7
                op = op * 0.7
                ck = 'bglow'
            col = th[ck]
            pulses.append(
                f'<use href="#{pid}" stroke="{col}" stroke-opacity="{op:g}" stroke-width="{w:g}" '
                f'stroke-dasharray="{d} {f(period - d)}">'
                + anim('stroke-dashoffset', [(t, d - s) for t, s in keys]) + '</use>')

    intro = []
    pending_cont = {}

    def intro_fill(pid, length, t0, cls=''):
        L0 = length + 4
        dd = L0 / INTRO_SPEED
        t1 = t0 + dd + 0.25
        intro.append(
            f'<use href="#{pid}"{cls} stroke-dasharray="{f(L0)} {f(L0)}" stroke-dashoffset="{f(L0)}">'
            f'<animate attributeName="stroke-dashoffset" from="{f(L0)}" to="0" begin="{t0:.2f}s" '
            f'dur="{dd:.2f}s" fill="freeze" calcMode="spline" keyTimes="0;1" keySplines=".3 0 .4 1"/>'
            f'<animate attributeName="opacity" from="1" to="0" begin="{t1:.2f}s" dur="1.6s" fill="freeze"/>'
            '</use>')
        return t0 + dd

    for t in traces:
        tr = t['tr']
        pid = add_path(tr)
        base.append(f'<use href="#{pid}"/>')
        if INTRO:
            # one-off power-up: light floods each channel outward from the chip
            t0 = 0.25 + 0.14 * max(t['lvl'], 0) + (0.1 if t['side'] == 'R' else 0)
            t_end = intro_fill(pid, tr.length, t0)
            if 'via' in t:
                x, y = t['via']
                intro.append(
                    f'<circle cx="{f(x)}" cy="{f(y)}" r="16" fill="url(#halo)" opacity="0">'
                    f'<animate attributeName="opacity" values="0;.8;0" keyTimes="0;.1;1" '
                    f'begin="{t_end:.2f}s" dur="1.6s"/></circle>')
            if 'cont' in t:
                pending_cont[id(t)] = t_end + 0.05
        if t['pulses']:
            add_pulses(pid, tr, t['pulses'])
        if 'via' in t and t['kind'] != 'dog':
            arr = [t0 + (tr.length + MARGIN) / SPEED for t0 in t['pulses']]
            via_times[t['via']] = arr
            if 'cont' in t:
                c = t['cont']
                cid = add_path(c)
                bot.append(f'<use href="#{cid}"/>')
                if INTRO:
                    intro_fill(cid, c.length, pending_cont[id(t)], ' class="b"')
                if arr:
                    add_pulses(cid, c, [a - MARGIN / SPEED + 0.05 for a in arr], bottom=True)

    # chip pads
    pads = []
    for o in chip_pins():
        for side in range(4):
            if side == 0:
                x, y, w, h = CX - PAD_C, CY + o, PAD_L, PAD_W
            elif side == 1:
                x, y, w, h = CX + PAD_C, CY + o, PAD_L, PAD_W
            elif side == 2:
                x, y, w, h = CX + o, CY - PAD_C, PAD_W, PAD_L
            else:
                x, y, w, h = CX + o, CY + PAD_C, PAD_W, PAD_L
            pads.append(f'<rect x="{f(x - w / 2)}" y="{f(y - h / 2)}" width="{f(w)}" '
                        f'height="{f(h)}" rx="1.2"/>')
    gaps, silk_extra = [], []
    for y, xc in ((CY - 5, CAP_X - 14), (CY + 5, CAP_X + 14)):
        for dx in (-CAP_P, CAP_P):
            pads.append(f'<rect x="{f(xc + dx - 3.5)}" y="{f(y - 3.6)}" width="7" height="7.2" rx="1.2"/>')
            gaps.append(f'<rect x="{f(xc + dx - 3.5 - OL_W / 2)}" y="{f(y - 3.6 - OL_W / 2)}" '
                        f'width="{f(7 + OL_W)}" height="{f(7.2 + OL_W)}" rx="1.6" class="co"/>')
        gaps.insert(0, f'<rect x="{f(xc - CAP_P + 3.5 + OL_W)}" y="{f(y - 3)}" '
                    f'width="{f(2 * CAP_P - 7 - 2 * OL_W)}" height="6"/>')
        silk_extra.append(f'<rect x="{f(xc - CAP_P - 6.5)}" y="{f(y - 4.5 if y < CY else y - 4.5)}" '
                          f'width="{f(2 * CAP_P + 13)}" height="9" rx="2" fill="none"/>')
    silk_extra.append(f'<path d="{text_d("U1", 8.5, CX + BODY - 15, CY + BODY - 8, "middle")}"/>')
    fid = []
    for fx, fy in FIDUCIALS:
        fid.append(f'<circle cx="{fx}" cy="{fy}" r="3"/>')
        silk_extra.append(f'<circle cx="{fx}" cy="{fy}" r="8" fill="none" stroke-width="1"/>')
    via_svg = [f'<circle cx="{f(x)}" cy="{f(y)}" r="{VIA_R}"/>' for x, y in vias]
    hole_svg = [f'<circle cx="{f(x)}" cy="{f(y)}" r="{VIA_HOLE}"/>' for x, y in vias]
    for (x, y), arr in via_times.items():
        if not arr:
            continue
        keys = flash_keys(arr)
        vglow.append(f'<circle cx="{f(x)}" cy="{f(y)}" r="16" fill="url(#halo)" opacity="0">'
                     + anim('opacity', keys, lambda v: f'{v:.2f}'.rstrip('0').rstrip('.') or '0')
                     + '</circle>')
        vglow.append(f'<circle cx="{f(x)}" cy="{f(y)}" r="{VIA_R - 0.6}" fill="none" '
                     f'stroke="{th["pulse"]}" stroke-width="1.6" opacity="0">'
                     + anim('opacity', keys, lambda v: f'{v:.2f}'.rstrip('0').rstrip('.') or '0')
                     + '</circle>')

    # ground pour: a slightly lighter plane with clearance channels cut around
    # every piece of copper (drawn as thick background-coloured strokes)
    PG = POUR_GAP
    clr = [f'<use href="#p{i}"/>' for i in range(n)]
    clr_v = [f'<circle cx="{f(x)}" cy="{f(y)}" r="{f(VIA_R + PG)}"/>' for x, y in vias]
    e = PAD_C + PAD_L / 2 + PG
    clr_v.append(f'<rect x="{f(CX - e)}" y="{f(CY - e)}" width="{f(2 * e)}" height="{f(2 * e)}" rx="6"/>')
    stitch = [f'<circle cx="{f(x)}" cy="{f(y)}" r="{STITCH_R}"/>' for x, y in stitch_pts(traces, vias)]

    L = th['line']
    if STYLE == 'outline':
        copper = (
            f'<g class="olb">{"".join(bot)}</g><g class="inb">{"".join(bot)}</g>'
            f'<g class="ol">{"".join(base)}</g>'
            f'<g class="olp">{"".join(pads)}</g>'
            f'<g class="olv">{"".join(via_svg)}</g>'
            f'<g class="in">{"".join(base)}</g>'
            f'<g class="inp">{"".join(pads)}</g>'
            f'<g class="inv">{"".join(via_svg)}</g>'
            f'<g class="olh">{"".join(hole_svg)}</g>'
            f'<g fill="{th["bg"]}">{"".join(gaps)}</g>'
            f'<g fill="{L}" fill-opacity=".7">{"".join(fid)}</g>')
    else:
        copper = (
            (f'<rect x="{POUR_IN}" y="{POUR_IN}" width="{W - 2 * POUR_IN}" height="{H - 2 * POUR_IN}" '
             f'rx="8" fill="{th["pour"]}"/><g class="clr">{"".join(clr)}{"".join(clr_v)}</g>'
             f'<g class="st">{"".join(stitch)}</g>' if POUR else '')
            + f'<g class="bl">{"".join(bot)}</g><g class="cu">{"".join(base)}</g>'
            f'<g class="pad">{"".join(pads)}</g><g class="via">{"".join(via_svg)}</g>'
            f'<g class="hole">{"".join(hole_svg)}</g>')
    L = th['line']
    css_ol = (
        f'.ol use{{fill:none;stroke:{L};stroke-width:{f(TW + 2 * OL_W)};stroke-linecap:round;stroke-linejoin:round}}'
        f'.in use{{fill:none;stroke:{th["bg"]};stroke-width:{f(TW)};stroke-linecap:round;stroke-linejoin:round}}'
        f'.olp rect{{fill:{L};stroke:{L};stroke-width:{f(2 * OL_W)}}}'
        f'.inp rect{{fill:{th["bg"]}}}'
        f'.olv circle{{fill:{L};stroke:{L};stroke-width:{f(2 * OL_W)}}}'
        f'.inv circle{{fill:{th["bg"]}}}'
        f'.olh circle,.co{{fill:none;stroke:{L};stroke-width:{f(OL_W)}}}'
    )
    css_ol += (
        f'.olb use{{fill:none;stroke:{th["bottom"]};stroke-width:{f(TW_B + 2 * OL_W)};stroke-linecap:round}}'
        f'.inb use{{fill:none;stroke:{th["bg"]};stroke-width:{f(TW_B)};stroke-linecap:round}}')
    css = css_ol + (
        f'.clr use{{fill:none;stroke:{th["bg"]};stroke-width:{f(TW + 2 * PG)};stroke-linecap:round;stroke-linejoin:round}}'
        f'.clr circle,.clr rect{{fill:{th["bg"]}}}'
        f'.st{{fill:{th["pad"]};fill-opacity:{th["stitch_op"]}}}'
        f'.cu use{{fill:none;stroke:{th["copper"]};stroke-opacity:{th["copper_op"]};'
        f'stroke-width:{TW};stroke-linecap:round;stroke-linejoin:round}}'
        f'.bl use{{fill:none;stroke:{th["bottom"]};stroke-opacity:{th["bottom_op"]};'
        f'stroke-width:{TW_B};stroke-linecap:round}}'
        f'.pu use{{fill:none;stroke-linecap:round}}'
        f'.intro use{{fill:none;stroke:{th["glow"]};stroke-opacity:{th["intro_op"]};stroke-width:{f(TW - 1.6)};stroke-linecap:round}}'
        f'.intro use.b{{stroke:{th["bglow"]};stroke-width:{f(TW_B - 1)}}}'
        f'.pad{{fill:{th["pad"]};fill-opacity:{th["pad_op"]}}}'
        f'.via{{fill:{th["pad"]};fill-opacity:{th["pad_op"]}}}'
        f'.hole{{fill:{th["bg"]}}}'
        f'.silk{{fill:none;stroke:{th["silk"]};stroke-opacity:{th["silk_op"]};stroke-width:1}}'
    )
    fade_in = (f'<animate attributeName="opacity" from="0" to="1" begin="{INTRO_END - 0.8:.2f}s" '
               f'dur="1.2s" fill="freeze"/>') if INTRO else ''
    beats = [BEAT0 + b * BEAT for b, lv in enumerate(ORDER) if lv is not None]
    hb = flash_keys(beats, rise=0.25, fall=1.5, peak=1.0)
    chip = (
        f'<rect x="{CX - BODY + 2}" y="{CY - BODY + 2}" width="{2 * BODY - 4}" '
        f'height="{2 * BODY - 4}" rx="3" class="silk"/>'
        f'<rect x="{CX - BODY + 2}" y="{CY - BODY + 2}" width="{2 * BODY - 4}" '
        f'height="{2 * BODY - 4}" rx="3" fill="url(#core)" opacity="0">'
        + anim('opacity', hb, lambda v: f'{v:.2f}'.rstrip('0').rstrip('.') or '0') + '</rect>'
        f'<circle cx="{CX - BODY + 12}" cy="{CY - BODY + 12}" r="4.5" class="silk"/>'
        f'<g fill="{th["silk"]}" fill-opacity="{th["silk_op"]}" stroke="{th["silk"]}">{"".join(silk_extra)}</g>'
    )
    core = (f'<radialGradient id="core"><stop offset="0" stop-color="{th["glow"]}" stop-opacity=".16"/>'
            f'<stop offset="1" stop-color="{th["glow"]}" stop-opacity="0"/></radialGradient>')
    halo = core + (f'<radialGradient id="halo"><stop offset="0" stop-color="{th["glow"]}" stop-opacity=".55"/>'
            f'<stop offset=".35" stop-color="{th["glow"]}" stop-opacity=".18"/>'
            f'<stop offset="1" stop-color="{th["glow"]}" stop-opacity="0"/></radialGradient>')
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">
<style>{css}</style>
<defs>{''.join(defs)}{halo}<clipPath id="clip"><rect width="{W}" height="{H}" rx="16"/></clipPath></defs>
<rect width="{W}" height="{H}" rx="16" fill="{th['bg']}"/>
<g clip-path="url(#clip)">
{copper}
<g class="intro">{''.join(intro)}</g>
<g opacity="{0 if INTRO else 1}">{fade_in}<g class="pu">{''.join(pulses)}</g>
<g>{''.join(vglow)}</g></g>
{chip}
</g>
<rect x=".5" y=".5" width="{W - 1}" height="{H - 1}" rx="15.5" fill="none" stroke="{th['border']}"/>
</svg>
"""
    return svg


def main():
    traces, vias = build_layout()
    gap, where = drc(traces, vias)
    print(f'DRC min gap {gap:.2f} (need {CLEAR}) at {where}')
    schedule(traces)
    for theme in THEMES:
        svg = build_svg(theme, traces, vias)
        path = os.path.join(HERE, f'banner-{theme}.svg')
        with open(path, 'w') as fh:
            fh.write(svg)
        print(path, len(svg.encode()) // 1024, 'KB')


if __name__ == '__main__':
    main()
