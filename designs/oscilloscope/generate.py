#!/usr/bin/env python3
"""Oscilloscope banner: a row of Lissajous figures (X-Y mode) whose phase sweeps
slowly, so each figure turns like a wire object, while a phosphor beam with a
decaying tail traces every curve.

Deterministic. Writes banner-dark.svg and banner-light.svg next to this file.

Technique notes
- Each figure is ONE <path> in <defs> whose `d` is morphed with SMIL over a full
  2*pi phase sweep (keyframes are exact cubic Hermite fits of the curve, so the
  same command structure is used in every keyframe and the loop is seamless).
- Glow is faked with stacked <use> strokes of decreasing width (no filters on
  anything that moves). The beam is a set of dashes on the same path
  (pathLength=1000) animated with stroke-dashoffset.
"""
import math
import os

import uharfbuzz as hb
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont

HERE = os.path.dirname(os.path.abspath(__file__))

W, H = 1200, 340
DIV = 40                      # graticule division (px)
SCR_X, SCR_Y = 40, 30         # screen origin
COLS, ROWS = 28, 7            # divisions
SCR_W, SCR_H = COLS * DIV, ROWS * DIV
CY = SCR_Y + SCR_H / 2

RATIOS = [(1, 1), (1, 2), (2, 3), (3, 4), (4, 5)]   # x-freq : y-freq
AMP = 92                      # figure half-size
SWEEP_DUR = 32.0              # seconds for a full 2*pi phase sweep
KEYS = 24                     # keyframes per sweep
SEG_PER_CYCLE = 8             # cubic segments per cycle of the faster channel
BEAM_PERIOD_PER_LOBE = 1.15   # seconds of beam travel per (a+b): equal beam speed

FONT_MONO = "/usr/share/fonts/fonts-go/Go-Mono.ttf"


def f1(v):
    s = f"{v:.1f}"
    if s.endswith(".0"):
        s = s[:-2]
    if s == "-0":
        s = "0"
    return s


# --------------------------------------------------------------------------- curves
def lissajous_d(cx, cy, a, b, delta, amp):
    """Closed Lissajous x=sin(a t + delta), y=sin(b t) as cubic Beziers."""
    n = SEG_PER_CYCLE * max(a, b)
    T = 2 * math.pi
    dt = T / n

    def P(t):
        return (cx + amp * math.sin(a * t + delta), cy - amp * math.sin(b * t))

    def D(t):
        return (amp * a * math.cos(a * t + delta), -amp * b * math.cos(b * t))

    # start slightly off zero so the seam of the path never sits on a cusp
    t0 = 0.1234
    x0, y0 = P(t0)
    parts = [f"M{f1(x0)} {f1(y0)}"]
    for i in range(n):
        ta, tb = t0 + i * dt, t0 + (i + 1) * dt
        pa, pb = P(ta), P(tb)
        da, db = D(ta), D(tb)
        c1 = (pa[0] + da[0] * dt / 3, pa[1] + da[1] * dt / 3)
        c2 = (pb[0] - db[0] * dt / 3, pb[1] - db[1] * dt / 3)
        parts.append("C" + " ".join(f1(v) for v in (*c1, *c2, *pb)))
    return "".join(parts) + "Z"


# --------------------------------------------------------------------------- text
def text_path(txt, font_path, size, x, y, anchor="middle", tracking=0.0):
    blob = hb.Blob.from_file_path(font_path)
    face = hb.Face(blob)
    font = hb.Font(face)
    upem = face.upem
    buf = hb.Buffer()
    buf.add_str(txt)
    buf.guess_segment_properties()
    hb.shape(font, buf, {"kern": True})
    tt = TTFont(font_path)
    gs = tt.getGlyphSet()
    order = tt.getGlyphOrder()
    scale = size / upem
    adv_total = sum(p.x_advance for p in buf.glyph_positions) + tracking * upem * (len(txt) - 1)
    width = adv_total * scale
    if anchor == "middle":
        x -= width / 2
    elif anchor == "end":
        x -= width
    pen = SVGPathPen(gs, ntos=lambda v: f1(v))
    cur = 0
    for info, pos in zip(buf.glyph_infos, buf.glyph_positions):
        name = order[info.codepoint]
        tp = TransformPen(pen, (scale, 0, 0, -scale, x + (cur + pos.x_offset) * scale,
                                y - pos.y_offset * scale))
        gs[name].draw(tp)
        cur += pos.x_advance + tracking * upem
    return pen.getCommands()


# --------------------------------------------------------------------------- themes
# tail: (colour, width, opacity per step, steps) -> `steps` stacked dashes that all
#       end at the beam head, lengths TAIL_PX*k/steps, giving a smooth decay.
# head: (colour, width, opacity) -> a dot (tiny dash with round caps) at the head.
THEMES = {
    "dark": dict(
        card="#05080a",
        card_edge="#1b232a",
        screen_hi="#0a1410",       # vignette centre
        screen_lo="#05080a",
        paper=False,
        grid="#7fe6b8", grid_op=0.16, grid_major_op=0.20,
        label="#7fe6b8", label_op=0.46,
        trace=[("#20ff9a", 8.0, 0.03), ("#2cffa0", 4.4, 0.06), ("#3dffa6", 2.2, 0.15),
               ("#93ffc9", 1.0, 0.55)],
        tail_px=190,
        tail=[("#22ff9c", 11.0, 0.04, 3), ("#40ffaa", 5.0, 0.075, 4), ("#8dffcb", 2.4, 0.17, 4),
              ("#eafff4", 1.3, 0.26, 5)],
        head=[("#30ffa4", 18.0, 0.10), ("#7dffc6", 9.0, 0.22), ("#d8fff0", 4.6, 0.75),
              ("#ffffff", 2.6, 1.0)],
    ),
    "light": dict(
        card="#fcfbf7",
        card_edge="#e4e1d8",
        screen_hi="#fcfbf7",
        screen_lo="#fcfbf7",
        paper=True,
        grid="#3d8aa6", grid_op=0.10, grid_major_op=0.24,
        label="#2a4656", label_op=0.6,
        trace=[("#1d3c78", 3.6, 0.045), ("#1d3c78", 1.8, 0.12), ("#1a3570", 0.95, 0.55)],
        tail_px=170,
        tail=[("#1a2f66", 3.0, 0.045, 3), ("#14285a", 1.35, 0.2, 5)],
        head=[("#c0402c", 8.0, 0.12), ("#b8392a", 3.4, 0.95)],
    ),
}


def graticule(th):
    g = []
    x0, y0, x1, y1 = SCR_X, SCR_Y, SCR_X + SCR_W, SCR_Y + SCR_H
    minor = DIV / 5
    cx_ = x0 + SCR_W / 2
    col = th["grid"]
    if th["paper"]:
        # graph paper: fine millimetre lines + division lines
        fine, major = [], []
        for k in range(1, COLS * 5):
            x = x0 + k * minor
            (major if k % 5 == 0 else fine).append(f"M{f1(x)} {y0}V{y1}")
        for k in range(1, ROWS * 5):
            y = y0 + k * minor
            (major if k % 5 == 0 else fine).append(f"M{x0} {f1(y)}H{x1}")
        g.append(f'<path d="{"".join(fine)}" stroke="{col}" stroke-opacity="{th["grid_op"]:.3f}" stroke-width="0.6"/>')
        g.append(f'<path d="{"".join(major)}" stroke="{col}" stroke-opacity="{th["grid_major_op"] * 0.85:.3f}" stroke-width="0.8"/>')
        g.append(f'<path d="M{x0} {f1(CY)}H{x1}M{f1(cx_)} {y0}V{y1}" stroke="{col}" '
                 f'stroke-opacity="{th["grid_major_op"] * 1.5:.3f}" stroke-width="1"/>')
    else:
        # etched graticule: dotted division lines (a dot at every minor tick)
        dots = []
        for c in range(1, COLS):
            x = x0 + c * DIV
            for k in range(1, ROWS * 5):
                dots.append(f"M{f1(x)} {f1(y0 + k * minor)}h0")
        for r in range(1, ROWS):
            y = y0 + r * DIV
            for k in range(1, COLS * 5):
                if k % 5:
                    dots.append(f"M{f1(x0 + k * minor)} {f1(y)}h0")
        g.append(f'<path d="{"".join(dots)}" stroke="{col}" stroke-opacity="{th["grid_op"]:.3f}" '
                 f'stroke-width="1.4" stroke-linecap="round"/>')
        # centre axes with minor ticks
        ticks = []
        for k in range(1, COLS * 5):
            if k % 5:
                ticks.append(f"M{f1(x0 + k * minor)} {f1(CY - 4)}v8")
        for k in range(1, ROWS * 5):
            if k % 5:
                ticks.append(f"M{f1(cx_ - 4)} {f1(y0 + k * minor)}h8")
        ticks.append(f"M{x0} {f1(CY)}H{x1}M{f1(cx_)} {y0}V{y1}")
        g.append(f'<path d="{"".join(ticks)}" stroke="{col}" stroke-opacity="{th["grid_major_op"]:.3f}" '
                 f'stroke-width="1"/>')
    g.append(f'<rect x="{x0}" y="{y0}" width="{SCR_W}" height="{SCR_H}" '
             f'stroke="{col}" stroke-opacity="{th["grid_major_op"]:.3f}" stroke-width="1"/>')
    return "\n".join(g)


def mean_length(a, b, amp, samples=4000):
    tot = 0.0
    deltas = [2 * math.pi * k / 12 for k in range(12)]
    for d in deltas:
        px, py = None, None
        for i in range(samples + 1):
            t = 2 * math.pi * i / samples
            x, y = amp * math.sin(a * t + d), amp * math.sin(b * t)
            if px is not None:
                tot += math.hypot(x - px, y - py)
            px, py = x, y
    return tot / len(deltas)


def use(i, attrs, child=""):
    return f'<use href="#f{i}" xlink:href="#f{i}" {attrs}>{child}</use>' if child else \
        f'<use href="#f{i}" xlink:href="#f{i}" {attrs}/>'


def build(theme):
    th = THEMES[theme]
    out = []
    out.append(f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
               f'viewBox="0 0 {W} {H}" width="{W}" height="{H}" fill="none">')
    out.append("<defs>")
    out.append(f'<radialGradient id="vig" cx="50%" cy="50%" r="60%">'
               f'<stop offset="0" stop-color="{th["screen_hi"]}"/>'
               f'<stop offset="1" stop-color="{th["screen_lo"]}"/></radialGradient>')
    out.append(f'<clipPath id="card"><rect width="{W}" height="{H}" rx="16"/></clipPath>')

    n = len(RATIOS)
    cell = SCR_W / n
    centers = [SCR_X + cell * (i + 0.5) for i in range(n)]
    for i, (a, b) in enumerate(RATIOS):
        cx = centers[i]
        vals = [lissajous_d(cx, CY, a, b, 2 * math.pi * k / KEYS, AMP) for k in range(KEYS)]
        vals.append(vals[0])
        out.append(f'<path id="f{i}" pathLength="1000" d="{vals[0]}">'
                   f'<animate attributeName="d" dur="{SWEEP_DUR}s" repeatCount="indefinite" '
                   f'values="{";".join(vals)}"/></path>')
    out.append("</defs>")

    out.append('<g clip-path="url(#card)">')
    out.append(f'<rect width="{W}" height="{H}" fill="{th["card"]}"/>')
    out.append(f'<rect x="{SCR_X}" y="{SCR_Y}" width="{SCR_W}" height="{SCR_H}" fill="url(#vig)"/>')
    out.append(graticule(th))

    lab = []
    for i, (a, b) in enumerate(RATIOS):
        lab.append(text_path(f"{a}:{b}", FONT_MONO, 11.5, centers[i], SCR_Y + SCR_H + 19, tracking=0.08))
    out.append(f'<path d="{"".join(lab)}" fill="{th["label"]}" fill-opacity="{th["label_op"]}"/>')

    for i, (a, b) in enumerate(RATIOS):
        g = ['<g stroke-linecap="round" stroke-linejoin="round">']
        for col, w, op in th["trace"]:
            g.append(use(i, f'stroke="{col}" stroke-width="{w}" stroke-opacity="{op}"'))
        unit = 1000.0 / mean_length(a, b, AMP)        # pathLength units per px
        per = BEAM_PERIOD_PER_LOBE * (a + b)
        dashes = []
        for col, w, op, steps in th["tail"]:
            for k in range(1, steps + 1):
                dashes.append((col, w, op, th["tail_px"] * k / steps * unit))
        for col, w, op in th["head"]:
            dashes.append((col, w, op, 0.6 * unit))
        for col, w, op, L in dashes:
            L = round(L, 2)
            anim = (f'<animate attributeName="stroke-dashoffset" from="{L}" to="{round(L - 1000, 2)}" '
                    f'dur="{per:.3f}s" repeatCount="indefinite"/>')
            g.append(use(i, f'stroke="{col}" stroke-width="{w}" stroke-opacity="{op}" '
                            f'stroke-dasharray="{L} {round(1000 - L, 2)}" stroke-dashoffset="{L}"', anim))
        g.append("</g>")
        out.append("\n".join(g))

    out.append("</g>")
    out.append(f'<rect x="0.5" y="0.5" width="{W - 1}" height="{H - 1}" rx="15.5" '
               f'stroke="{th["card_edge"]}"/>')
    out.append("</svg>")
    return "\n".join(out)


if __name__ == "__main__":
    for t in THEMES:
        p = os.path.join(HERE, f"banner-{t}.svg")
        s = build(t)
        with open(p, "w") as fh:
            fh.write(s)
        print(p, f"{len(s.encode()) / 1024:.1f} KB")
