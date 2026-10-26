#!/usr/bin/env python3
"""Animated Norse knotwork banner.

The interlace is generated, not drawn: knot.py builds it as the medial graph
of a lattice with breaks (every lattice edge becomes a crossing, strands turn
around lattice vertices), which guarantees a strictly alternating over/under
weave.  geom.py turns each strand into smooth cubics and splits it into one
piece per crossing; pieces that pass *under* are painted first, pieces that
pass *over* are painted last on top of a background-coloured halo, so the
weave is correct at every moment of the animation.

Each strand is a ribbon: two contour lines (offset cubics) over a faint body.
Motion (one 24 s SMIL loop):
  * the ribbon draws itself from two pens that start near the middle of each
    strand and run in opposite directions (stroke-dashoffset per piece,
    pathLength-normalised so both contours advance together), a short warm
    spark riding at each pen;
  * once complete, a slanted band of warm light glides across the metal (one
    animated userSpaceOnUse gradient shared by every contour and body);
  * the pens then run again and the ribbon retracts, leaving the faint
    engraved ghost of the knot until the next pass.
A row of Elder Futhark (the fuþark itself, purely decorative) is drawn as
stroked paths beneath.

Usage:  python generate.py   -> banner-dark.svg, banner-light.svg
"""
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from knot import build_graph, trace, over_under          # noqa: E402
from geom import (strand_curves, pieces, path_d, offset_cubic, bez_sub,  # noqa: E402
                  bez_len, fmt)
from runes import RUNES, AETTIR, width as rune_width      # noqa: E402

# ----------------------------------------------------------------- layout
W, H = 1200, 320
CARD_RX = 16
COLS, ROWS = 15, 3
UNIT_X, UNIT_Y = 62, 60
KNOT_CY = 148
SWELL = 0.2            # the band swells gently towards the middle
SWELL_SIGMA = 330

RIBBON = 13.0          # ribbon width (centre to centre of contour lines)
LINE_W = 1.6           # contour line width
GROOVE_W = 0.8
GAP = 4.5              # clearance around an over-passing ribbon
HALO_SPAN = 0.4        # halo: c1 from t=HALO_SPAN, c2 up to t=1-HALO_SPAN

# ----------------------------------------------------------------- timing
DUR = 24.0
DRAW_END = 6.8
DRAW_MAX = 6.0         # longest strand takes this long
UNDRAW_START = 19.6
UNDRAW_MAX = 3.0
UNDRAW_END = UNDRAW_START + UNDRAW_MAX
TIP = 26               # length of the warm glow behind each pen (0 = off)
SWEEP = (8.2, 18.6)    # a band of warm light crosses the knot
SWEEP_X = (20, 1180)   # band centre travels between these x (before skew)
SWEEP_WIDTH = 440

THEMES = {
    'dark': dict(bg='#0f141b', border='#262f3a', line='#b9c3ce', fill=0.06,
                 glint='#e3a654', glint_line='#ffdcaa', fill_hot=0.12, line_w=1.6,
                 ghost=0.085, tip_op=0.9, rune='#8f9aa7', rune_op=0.55),
    'light': dict(bg='#fbfaf6', border='#e3e0d6', line='#27303a', fill=0.07,
                  glint='#d39a3a', glint_line='#9c6614', fill_hot=0.12, line_w=1.8,
                  fill_tint='#9a8460',
                  ghost=0.07, tip_op=0.85, rune='#5a636e', rune_op=0.6),
}


def E(a, b):
    return frozenset((a, b))


def lattice_breaks():
    """Medallions joined by narrow necks, symmetric under a half turn."""
    rem = set()
    c = COLS - 1
    for i in (3, 7):
        for ii in {i, c - i}:
            for j in range(ROWS - 1):
                rem.add(E((ii, j), (ii, j + 1)))
    for i in (1, 5):
        for ii in {i, c - 1 - i}:
            rem.add(E((ii, 0), (ii + 1, 0)))
            rem.add(E((ii, ROWS - 1), (ii + 1, ROWS - 1)))
    return rem


def ease(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * (3 - 2 * u) if u < 1 else 1.0


def ease_inv(y):
    lo, hi = 0.0, 1.0
    for _ in range(50):
        mid = (lo + hi) / 2
        if ease(mid) < y:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def build():
    g = build_graph(COLS, ROWS, lattice_breaks())
    strands = trace(g)
    ou = over_under(strands)

    def xf(x, y):
        X = W / 2 + (x - (COLS - 1) / 2) * UNIT_X
        Y = KNOT_CY - (y - (ROWS - 1) / 2) * UNIT_Y
        Y = KNOT_CY + (Y - KNOT_CY) * (1 + SWELL * math.exp(-((X - W / 2) / SWELL_SIGMA) ** 2))
        return X, Y

    params = dict(k_straight=1 / 3, k_bounce=1.0, k_corner=1.0)
    out = []
    for si, st in enumerate(strands):
        segs = strand_curves(st, xf, params)
        pcs, total = pieces(segs)
        # choose the pen origin: the piece boundary closest to the banner centre
        best, bk = None, 0
        for k, (cub, s0, L) in enumerate(pcs):
            p = cub[0][0]
            d = math.hypot(p[0] - W / 2, (p[1] - KNOT_CY) * 1.5)
            if best is None or d < best:
                best, bk = d, k
        order = list(range(bk, len(pcs))) + list(range(0, bk))
        recs = []
        s = 0.0
        for k in order:
            cub, _, L = pcs[k]
            recs.append(dict(cub=cub, L=L, s0=s, over=ou[si][k]))
            s += L
        # split the strand into a forward half and a backward half at a piece boundary
        acc, m = 0.0, 0
        for idx, r in enumerate(recs):
            if acc + r['L'] / 2 > s / 2:
                m = idx
                break
            acc += r['L']
        out.append(dict(recs=recs, total=s, split=m, fwd_len=acc, bwd_len=s - acc))
    return out


def timing(strands):
    """Attach keyframe times (seconds) for drawing/undrawing every piece."""
    longest = max(max(st['fwd_len'], st['bwd_len']) for st in strands)
    for st in strands:
        half = max(st['fwd_len'], st['bwd_len'])
        frac = (half / longest) ** 0.6
        d_draw = DRAW_MAX * frac
        d_und = UNDRAW_MAX * frac
        t0 = DRAW_END - d_draw
        u0 = UNDRAW_START + (UNDRAW_MAX - d_und) * 0.5

        def tdraw(x, n):
            return t0 + d_draw * ease_inv(min(max(x / n, 0), 1))

        def tund(x, n):
            return u0 + d_und * ease_inv(min(max(x / n, 0), 1))

        m = st['split']
        for idx, r in enumerate(st['recs']):
            if idx < m:
                a = r['s0']
                b = a + r['L']
                n = st['fwd_len']
                r['dir'] = 1
            else:
                # distance travelled by the backward pen when it reaches the piece end / start
                a = st['total'] - (r['s0'] + r['L'])
                b = st['total'] - r['s0']
                n = st['bwd_len']
                r['dir'] = -1
            r['td'] = (tdraw(a, n), tdraw(b, n))
            r['tu'] = (tund(a, n), tund(b, n))
    return strands


def kt(times):
    """Normalised, strictly increasing keyTimes."""
    out = []
    prev = -1
    for t in times:
        v = round(t / DUR, 5)
        if v <= prev:
            v = prev + 0.00002
        out.append(v)
        prev = v
    out[0] = 0
    out[-1] = 1
    return ';'.join(f'{v:.5f}'.rstrip('0').rstrip('.') if v not in (0, 1) else str(int(v)) for v in out)


def dash_anim(L, direction, td, tu):
    """stroke-dashoffset animation for a dash [L, L+4] on a path of length L."""
    # hidden states sit one unit clear of the path ends, so no zero-length
    # sliver of a dash can ever be painted there
    h = L + 1
    if direction == 1:
        vals = [h, h, 0, 0, -h, -h]
    else:
        vals = [-h, -h, 0, 0, h, h]
    times = [0, td[0], td[1], tu[0], tu[1], DUR]
    v = ';'.join(fmt(x) for x in vals)
    return (f'<animate attributeName="stroke-dashoffset" dur="{fmt(DUR)}s" repeatCount="indefinite" '
            f'values="{v}" keyTimes="{kt(times)}"/>')


def halo_geometry(r):
    """Sub-curve around the crossing (joint of the two cubics) and its offsets."""
    c1, c2 = r['cub']
    h1 = bez_sub(c1, HALO_SPAN, 1.0)
    h2 = bez_sub(c2, 0.0, 1 - HALO_SPAN)
    l_before = bez_len(c1) - bez_len(h1)
    return [h1, h2], l_before, bez_len(h1) + bez_len(h2)


def mix(c1, c2, t):
    a = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
    return '#' + ''.join(f'{round(x + (y - x) * t):02x}' for x, y in zip(a, b))


def sweep_gradient(gid, base, hot, th):
    """A soft slanted band of light, carried across the card once per loop."""
    kts = kt([0, SWEEP[0], SWEEP[1], DUR])
    hw = SWEEP_WIDTH / 2
    stops = ''.join(f'<stop offset="{o}" stop-color="{c}"/>' for o, c in (
        (0, base), (.2, base), (.5, hot), (.8, base), (1, base)))
    return (f'<linearGradient id="{gid}" gradientUnits="userSpaceOnUse" x1="{fmt(-hw)}" y1="0" x2="{fmt(hw)}" y2="0" '
            f'gradientTransform="skewX(-24)">{stops}'
            f'<animateTransform attributeName="gradientTransform" type="translate" additive="sum" '
            f'dur="{fmt(DUR)}s" repeatCount="indefinite" values="{fmt(SWEEP_X[0] - hw)};{fmt(SWEEP_X[0] - hw)};{fmt(SWEEP_X[1] + hw)};{fmt(SWEEP_X[1] + hw)}" '
            f'keyTimes="{kts}" calcMode="spline" keySplines="0 0 1 1;.3 0 .7 1;0 0 1 1"/></linearGradient>')


def tip_el(pid, r):
    """A short warm glow riding just behind each pen while it draws."""
    if not TIP:
        return ''
    L = r['L']
    ta, tb = r['td']
    tx = tb + (tb - ta) * TIP / L          # time at which the tip has fully left the piece
    if r['dir'] == 1:
        vals = [TIP + 1, TIP + 1, TIP, -L, -L - 1, -L - 1]
    else:
        vals = [-L - 1, -L - 1, -L, TIP, TIP + 1, TIP + 1]
    times = [0, ta - 0.001, ta, tx, tx + 0.001, DUR]
    v = ';'.join(fmt(x) for x in vals)
    return (f'<use href="#c{pid}" class="t" stroke-dasharray="{TIP} {fmt(L + 40)}" stroke-dashoffset="{fmt(vals[0])}">'
            f'<animate attributeName="stroke-dashoffset" dur="{fmt(DUR)}s" repeatCount="indefinite" '
            f'values="{v}" keyTimes="{kt(times)}"/></use>')


def render(theme_name, strands):
    th = THEMES[theme_name]
    defs = []
    ghost_under, ghost_over = [], []
    anim_under, anim_over = [], []
    pid = 0
    for st in strands:
        for r in st['recs']:
            pid += 1
            L = r['L']
            pl = fmt(L)
            cub = r['cub']
            ea = path_d(sum([offset_cubic(c, RIBBON / 2) for c in cub], []))
            eb = path_d(sum([offset_cubic(c, -RIBBON / 2) for c in cub], []))
            cd = path_d(cub)
            defs.append(f'<path id="a{pid}" d="{ea}" pathLength="{pl}"/>'
                        f'<path id="b{pid}" d="{eb}" pathLength="{pl}"/>'
                        f'<path id="c{pid}" d="{cd}"/>')
            lines = f'<use href="#a{pid}"/><use href="#b{pid}"/>'
            ghost = lines
            dd = r['dir']
            body = (f'<g stroke-dasharray="{pl} {fmt(L + 4)}">{dash_anim(L, dd, r["td"], r["tu"])}'
                    f'<use href="#c{pid}" class="f"/>{lines}</g>')
            if r['over']:
                hc, hb, hl = halo_geometry(r)
                defs.append(f'<path id="h{pid}" d="{path_d(hc)}"/>')
                ghost = f'<use href="#h{pid}" class="ha"/>' + ghost
                # the halo is revealed by the same pens: interpolate inside the piece
                f0, f1 = hb / L, (hb + hl) / L
                if dd == -1:
                    f0, f1 = 1 - f1, 1 - f0
                td = tuple(r['td'][0] + (r['td'][1] - r['td'][0]) * f for f in (f0, f1))
                tu = tuple(r['tu'][0] + (r['tu'][1] - r['tu'][0]) * f for f in (f0, f1))
                halo = (f'<use href="#h{pid}" class="ha" stroke-dasharray="{fmt(hl)} {fmt(hl + 4)}">'
                        f'{dash_anim(hl, dd, td, tu)}</use>')
                ghost_over.append(ghost)
                anim_over.append(halo + body + tip_el(pid, r))
            else:
                ghost_under.append(ghost)
                anim_under.append(body + tip_el(pid, r))

    runes = rune_row(th)
    fill_base = mix(th['bg'], th.get('fill_tint', th['line']), th['fill'])
    fill_hot = mix(fill_base, th['glint'], th['fill_hot'])
    line_hot = mix(th['line'], th['glint_line'], .85)
    css = (f'.k use{{fill:none;stroke:url(#sl);stroke-width:{th['line_w']}}}'
           f'.k use.f{{stroke:url(#sf);stroke-width:{fmt(RIBBON - LINE_W + .2)}}}'
           f'.k use.ha{{stroke:{th["bg"]};stroke-width:{fmt(RIBBON + LINE_W + 2 * GAP)}}}'
           f'.k use.t{{stroke:{th["glint"]};stroke-opacity:{th["tip_op"]};stroke-width:3;stroke-linecap:round}}'
           f'.g use{{fill:none;stroke:{th["line"]};stroke-width:{th['line_w']}}}'
           f'.g use.ha{{stroke:{th["bg"]};stroke-width:{fmt(RIBBON + LINE_W + 2 * GAP)}}}'
           f'.r path{{fill:none;stroke:{th["rune"]};stroke-width:1.2;stroke-linecap:round;stroke-linejoin:round}}')
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">',
        f'<style>{css}</style>',
        '<defs>',
        f'<clipPath id="cl"><rect width="{W}" height="{H}" rx="{CARD_RX}"/></clipPath>',
        sweep_gradient('sl', th['line'], line_hot, th),
        sweep_gradient('sf', fill_base, fill_hot, th),
        ''.join(defs),
        '</defs>',
        f'<rect x=".5" y=".5" width="{W - 1}" height="{H - 1}" rx="{CARD_RX}" fill="{th["bg"]}" stroke="{th["border"]}"/>',
        '<g clip-path="url(#cl)">',
        f'<g class="g" opacity="{th["ghost"]}">{"".join(ghost_under)}{"".join(ghost_over)}</g>',
        f'<g class="k">{"".join(anim_under)}{"".join(anim_over)}</g>',
        runes,
        '</g>',
        '</svg>',
    ]
    return '\n'.join(svg)


def rune_row(th):
    size = 6.2          # px per rune unit (stave height = 2 units)
    gapu = 0.95         # gap between runes (units)
    sepu = 2.2          # separator slot width
    items = []
    x = 0.0
    for ai, aett in enumerate(AETTIR):
        if ai:
            items.append(('sep', x + sepu / 2))
            x += sepu
        for name in aett:
            w, x0 = rune_width(name)
            items.append((name, x - x0))
            x += w + gapu
        x -= gapu
        x += gapu
    total = (x - gapu) * size
    left = W / 2 - total / 2
    base = 286
    parts = []
    n = len([i for i in items if i[0] != 'sep'])
    k = 0
    for name, ux in items:
        if name == 'sep':
            cx = left + ux * size - gapu * size / 2
            d = ''.join(f'M{fmt(cx)} {fmt(base - yy * size)}h0' for yy in (0.45, 1.0, 1.55))
            parts.append(f'<path d="{d}" stroke-width="1.7"/>')
            continue
        d = ''
        for line in RUNES[name]:
            d += 'M' + 'L'.join(f'{fmt(left + (ux + px) * size)} {fmt(base - py * size)}' for px, py in line)
        k += 1
        parts.append(f'<path d="{d}"/>')
    # quiet reveal: the row fades in as the knot completes and out as it retracts
    ktimes = kt([0, DRAW_END - 1.2, DRAW_END + 0.8, UNDRAW_START, UNDRAW_START + 2.0, DUR])
    op = th['rune_op']
    return (f'<g class="r" opacity="0"><animate attributeName="opacity" dur="{fmt(DUR)}s" repeatCount="indefinite" '
            f'values="0;0;{op};{op};0;0" keyTimes="{ktimes}"/>{"".join(parts)}</g>')


def main():
    strands = timing(build())
    for name in THEMES:
        svg = render(name, strands)
        p = os.path.join(HERE, f'banner-{name}.svg')
        with open(p, 'w') as f:
            f.write(svg)
        print(p, len(svg.encode()), 'bytes;', len(strands), 'strands;',
              sum(len(s['recs']) for s in strands), 'crossings')


if __name__ == '__main__':
    main()
