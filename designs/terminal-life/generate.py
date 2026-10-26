#!/usr/bin/env python3
"""Animated terminal banner, sibling of ../terminal: a shell types a command,
then Conway's Game of Life blooms in the alternate screen from a handful of
small methuselahs placed with four-fold mirror symmetry, then the program
exits and `clear` resets the loop.

The field is drawn the way terminal Life programs use braille: every cell is
one braille dot (2 x 4 dots per character cell). Each generation is a
precomputed frame; a frame is one <path> per age bucket whose vertices stamp a
dot marker. Only the top-left quadrant is stored; three mirrored <use>s
complete the field. When a generation is replaced it is not hidden at once but
steps down in opacity for a few generations, so dying cells leave a short
phosphor trail. All text is Go Mono / Inter outlines.

    python generate.py            # writes banner-dark.svg and banner-light.svg
"""
import random
import sys
from pathlib import Path

import numpy as np
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont

OUT = Path(__file__).resolve().parent
MONO = "/usr/share/fonts/fonts-go/Go-Mono.ttf"
SANS = "/usr/share/fonts/opentype/inter/Inter-Medium.otf"

# ---------------------------------------------------------------- geometry
W, H = 1200, 400
CW, LH = 9, 18                    # terminal cell, px
FONT_PX = CW / 0.6                # Go Mono advance is 0.6 em
BAR = 36                          # title bar height
PADX, PADY = 20, 12
X0 = PADX + 4
Y0 = BAR + PADY
COLS = (W - 2 * X0) // CW         # 128
ROWS = (H - Y0 - PADY) // LH      # 18
BASE = round(LH * 0.74, 2)

FROWS = ROWS - 1                  # last row is the program's status line
DX, DY = CW / 2, LH / 4           # braille dot pitch (4.5 x 4.5)
GW, GH = COLS * 2, FROWS * 4      # life grid: 256 x 68
QW, QH = GW // 2, GH // 2         # stored quadrant

T = 24.0                          # master loop, s
GENS = 196                        # generations shown
GPS = 12.0                        # generations per second
GHOST = (0.3, 0.12)               # opacity of a generation 1, 2, 3 steps later
SEED = (85, 3)                    # (rng seed, number of methuselahs per quadrant)

# ---------------------------------------------------------------- fonts
mono = TTFont(MONO)
mono_gs = mono.getGlyphSet()
mono_cmap = mono.getBestCmap()
MONO_UPEM = mono["head"].unitsPerEm


def fmt(v):
    s = f"{v:.2f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def glyph_d(ch, px=FONT_PX):
    name = mono_cmap[ord(ch)]
    pen = SVGPathPen(mono_gs, fmt)
    s = px / MONO_UPEM
    mono_gs[name].draw(TransformPen(pen, (s, 0, 0, -s, 0, 0)))
    return pen.getCommands()


def sans_text_d(text, px, x, y):
    import uharfbuzz as hb
    blob = hb.Blob.from_file_path(SANS)
    font = hb.Font(hb.Face(blob))
    buf = hb.Buffer()
    buf.add_str(text)
    buf.guess_segment_properties()
    hb.shape(font, buf, {"kern": True, "liga": True})
    tt = TTFont(SANS)
    gs = tt.getGlyphSet()
    order = tt.getGlyphOrder()
    s = px / tt["head"].unitsPerEm
    pen = SVGPathPen(gs, fmt)
    cx = 0
    for info, pos in zip(buf.glyph_infos, buf.glyph_positions):
        gx = x + (cx + pos.x_offset) * s
        gy = y - pos.y_offset * s
        gs[order[info.codepoint]].draw(TransformPen(pen, (s, 0, 0, -s, gx, gy)))
        cx += pos.x_advance
    return pen.getCommands(), cx * s


# ---------------------------------------------------------------- themes
# age buckets: newborn, young (2-3), adult (4-11), settled (12+)
AGE_EDGES = (1, 2, 9)

THEMES = {
    "dark": dict(
        card="#0b0f15", bar="#11161d", border="#262c36", hair="#1b2129",
        title="#7d8590", fg="#c9d1d9", dim="#6e7681", faint="#3d444d",
        dir="#79c0ff", chev="#c297ff", cursor="#d6dbe1",
        lights=("#ff5f57", "#febc2e", "#28c840"),
        ages=("#ffd6a8", "#d4d8f6", "#8b9de6", "#4a58a0"),
        glow="#4a5fd0", glow_a=0.12,
    ),
    "light": dict(
        card="#fbfbfc", bar="#f3f4f6", border="#d8dee4", hair="#e6e8ec",
        title="#6e7781", fg="#24292f", dim="#8c959f", faint="#c4cad1",
        dir="#0969da", chev="#8250df", cursor="#24292f",
        lights=("#ff5f57", "#febc2e", "#28c840"),
        ages=("#a85f2e", "#3a4074", "#6f80c8", "#aab4dc"),
        glow="#7d8fd6", glow_a=0.08,
    ),
}

# ---------------------------------------------------------------- life
PAT = {
    "acorn": [".O.....", "...O...", "OO..OOO"],
    "r": [".OO", "OO.", ".O."],
    "b": ["O.OO", "OOO.", ".O.."],
    "pi": ["OOO", "O.O", "O.O"],
    "rabbits": ["O...OOO", "OOO..O.", ".O....."],
    "thunder": ["OOO", "...", ".O.", ".O.", ".O."],
    "herschel": ["O..", "OOO", "O.O", "..O"],
}


def step(A):
    N = sum(np.roll(np.roll(A, i, 0), j, 1)
            for i in (-1, 0, 1) for j in (-1, 0, 1) if (i, j) != (0, 0))
    return ((N == 3) | ((A == 1) & (N == 2))).astype(np.uint8)


def seed_field(seed, k):
    """k small methuselahs in the top-left quadrant, mirrored four ways."""
    rng = np.random.default_rng(seed)
    F = np.zeros((GH, GW), np.uint8)
    for _ in range(k):
        n = rng.choice(list(PAT))
        p = np.array([[ch == "O" for ch in r] for r in PAT[n]], np.uint8)
        p = np.rot90(p, rng.integers(4))
        if rng.random() < .5:
            p = p[:, ::-1]
        r = rng.integers(GH // 2 - 14, GH // 2 - 1 - p.shape[0])
        c = rng.integers(GW // 2 - 80, GW // 2 - 3 - p.shape[1])
        F[r:r + p.shape[0], c:c + p.shape[1]] |= p
    F[:, GW // 2:] = F[:, :GW // 2][:, ::-1]
    F[GH // 2:] = F[:GH // 2][::-1]
    return F


def simulate(F0, gens, pad=120):
    """Field is a window onto a larger board; the board's rim is kept empty
    so escaping gliders simply leave."""
    A = np.zeros((GH + 2 * pad, GW + 2 * pad), np.uint8)
    A[pad:pad + GH, pad:pad + GW] = F0
    out = [F0]
    for _ in range(gens):
        A = step(A)
        A[:2] = 0; A[-2:] = 0; A[:, :2] = 0; A[:, -2:] = 0
        out.append(A[pad:pad + GH, pad:pad + GW].copy())
    return out


def ages(frames):
    age = np.zeros(frames[0].shape, int)
    out = []
    for F in frames:
        age = np.where(F == 1, age + 1, 0)
        out.append(age.copy())
    return out


def bucket(a):
    for i, e in enumerate(AGE_EDGES):
        if a <= e:
            return i
    return len(AGE_EDGES)


def cells_path(cells):
    """Path in dot units whose vertices sit on each live cell."""
    cells = sorted(cells, key=lambda rc: (rc[0], rc[1]))
    out = []
    pr = pc = None
    last = ""
    for r, c in cells:
        if pr is None:
            out.append(f"M{c} {r}"); last = "M"
        elif r == pr:
            # implicit repetition of h saves a byte per cell
            out.append((" " if last == "h" else "h") + str(c - pc)); last = "h"
        else:
            out.append(f"m{c - pc} {r - pr}"); last = "m"
        pr, pc = r, c
    s = "".join(out)
    return s.replace(" -", "-")


# ---------------------------------------------------------------- timeline
def kt(t):
    return f"{t / T:.5f}".rstrip("0").rstrip(".") if t else "0"


def discrete(attr, events, extra=""):
    assert events[0][0] == 0
    vals = ";".join(str(v) for _, v in events)
    keys = ";".join(kt(t) for t, _ in events)
    return (f'<animate attributeName="{attr}" dur="{T}s" repeatCount="indefinite" '
            f'calcMode="discrete" values="{vals}" keyTimes="{keys}"{extra}/>')


def linear(attr, events):
    if events[-1][0] < T:
        events = events + [(T, events[-1][1])]
    vals = ";".join(str(v) for _, v in events)
    keys = ";".join(kt(t) for t, _ in events)
    return (f'<animate attributeName="{attr}" dur="{T}s" repeatCount="indefinite" '
            f'values="{vals}" keyTimes="{keys}"/>')


def typing_times(text, start, rng):
    times = []
    t = start
    prev = ""
    for ch in text:
        d = rng.uniform(0.045, 0.085)
        if ch == " ":
            d = rng.uniform(0.07, 0.11)
        if prev == " ":
            d += rng.uniform(0.02, 0.06)
        if ch in "-./" and prev != ch:
            d += rng.uniform(0.04, 0.08)
        if ch.isupper() or ch.isdigit():
            d += rng.uniform(0.03, 0.06)        # shift / number row
        if ch == prev and ch in "-":
            d = rng.uniform(0.035, 0.05)
        t += d
        times.append(round(t, 3))
        prev = ch
    return times


# ---------------------------------------------------------------- build
def build(theme_name, frames, agef):
    th = THEMES[theme_name]
    rng = random.Random(11)
    parts, defs = [], []
    used = set()

    def g(ch):
        used.add(ch)
        return f"#c{ord(ch):x}"

    def text_uses(s, col, row, color=None):
        out = []
        for i, ch in enumerate(s):
            if ch == " ":
                continue
            out.append(f'<use href="{g(ch)}" x="{X0 + (col + i) * CW}" '
                       f'y="{Y0 + row * LH + BASE}"/>')
        inner = "".join(out)
        return f'<g fill="{color}">{inner}</g>' if color else inner

    # ---- timeline -------------------------------------------------------
    PROMPT_DIR = "~/src"
    PROMPT_COLS = len(PROMPT_DIR) + 3
    CMD = "./life --rule B3/S23 bloom.rle"
    t_type0 = 0.6
    ttimes = typing_times(CMD, t_type0, rng)
    t_enter = ttimes[-1] + 0.4
    t_alt = t_enter + 0.14
    t_run = t_alt + 0.55                       # gen 0 holds a beat
    gen_t = [t_alt] + [t_run + (k - 1) / GPS for k in range(1, len(frames))]
    t_last = gen_t[-1]
    t_exit = t_last + 0.9
    t_prompt2 = t_exit + 0.05
    ctimes = typing_times("clear", t_prompt2 + 0.7, rng)
    t_enter2 = ctimes[-1] + 0.36
    t_clear = t_enter2 + 0.06
    assert t_clear < T - 0.4, (t_clear, T)
    print(f'  enter {t_enter:.2f}  run {t_run:.2f}-{t_last:.2f}  exit {t_exit:.2f}  clear {t_clear:.2f} / {T}', file=sys.stderr)

    # ---- card + chrome --------------------------------------------------
    parts.append(f'<rect x=".5" y=".5" width="{W - 1}" height="{H - 1}" rx="14" '
                 f'fill="{th["card"]}" stroke="{th["border"]}"/>')
    parts.append(f'<path d="M.5 {BAR}V14.5A14 14 0 0 1 14.5 .5H{W - 14.5}'
                 f'A14 14 0 0 1 {W - .5} 14.5V{BAR}Z" fill="{th["bar"]}"/>')
    parts.append(f'<path d="M1 {BAR}.5H{W - 1}" stroke="{th["hair"]}"/>')
    for i, col in enumerate(th["lights"]):
        parts.append(f'<circle cx="{22 + i * 20}" cy="{BAR / 2}" r="6" fill="{col}"/>')
    size = f"{COLS}×{ROWS}"
    for label, vis in (
        (f"{PROMPT_DIR} — zsh — {size}",
         [(0, "visible"), (t_alt, "hidden"), (t_exit, "visible")]),
        (f"{PROMPT_DIR} — life — {size}",
         [(0, "hidden"), (t_alt, "visible"), (t_exit, "hidden")]),
    ):
        d, wpx = sans_text_d(label, 12.5, 0, 0)
        tx = (W - wpx) / 2
        parts.append(f'<path transform="translate({tx:.1f} {BAR / 2 + 4.4})" d="{d}" '
                     f'fill="{th["title"]}">{discrete("visibility", vis)}</path>')

    # ---- normal screen --------------------------------------------------
    def chevron(col, row):
        x = X0 + col * CW + 1.2
        y = Y0 + row * LH + LH / 2 - 0.3
        return (f'<path d="M{x:.1f} {y - 3.6:.1f}l3.6 3.6-3.6 3.6" fill="none" '
                f'stroke="{th["chev"]}" stroke-width="1.6" stroke-linecap="round" '
                f'stroke-linejoin="round"/>')

    def prompt(row):
        return text_uses(PROMPT_DIR, 0, row, th["dir"]) + chevron(len(PROMPT_DIR) + 1, row)

    def clip_typed(cid, row, times, col0, t_off):
        ev = [(0, 0)] + [(t, (i + 1) * CW) for i, t in enumerate(times)] + [(t_off, 0)]
        x = X0 + col0 * CW
        defs.append(f'<clipPath id="{cid}"><rect x="{x}" y="{Y0 + row * LH}" width="0" '
                    f'height="{LH}">{discrete("width", ev)}</rect></clipPath>')

    normal = [prompt(0)]
    clip_typed("k0", 0, ttimes, PROMPT_COLS, t_clear)
    normal.append(f'<g clip-path="url(#k0)">{text_uses(CMD, PROMPT_COLS, 0, th["fg"])}</g>')
    normal.append(f'<g opacity="0">{prompt(1)}'
                  f'{discrete("opacity", [(0, 0), (t_prompt2, 1), (t_clear, 0)])}</g>')
    clip_typed("k1", 1, ctimes, PROMPT_COLS, t_clear)
    normal.append(f'<g clip-path="url(#k1)">{text_uses("clear", PROMPT_COLS, 1, th["fg"])}</g>')
    parts.append(f'<g>{"".join(normal)}'
                 f'{discrete("opacity", [(0, 1), (t_alt, 0), (t_exit, 1)])}</g>')

    # ---- cursor ---------------------------------------------------------
    def blink(t0, t1):
        ev, t, on = [], t0, True
        while t < t1 - 1e-6:
            ev.append((round(t, 3), 1 if on else 0))
            t += 0.53
            on = not on
        return ev

    base_x = X0 + PROMPT_COLS * CW
    cx_ev, cy_ev, op_ev = [(0, base_x)], [(0, Y0)], []
    op_ev += blink(0, t_type0 - 0.1)
    op_ev.append((round(t_type0 - 0.1, 3), 1))
    for i, t in enumerate(ttimes):
        cx_ev.append((t, base_x + (i + 1) * CW))
    cx_ev.append((t_enter, X0)); cy_ev.append((t_enter, Y0 + LH))
    op_ev.append((t_alt, 0))
    cx_ev.append((t_prompt2, base_x)); cy_ev.append((t_prompt2, Y0 + LH))
    op_ev += blink(t_prompt2, ctimes[0] - 0.05)
    op_ev.append((round(ctimes[0] - 0.05, 3), 1))
    for i, t in enumerate(ctimes):
        cx_ev.append((t, base_x + (i + 1) * CW))
    cx_ev.append((t_enter2, X0)); cy_ev.append((t_enter2, Y0 + 2 * LH))
    cx_ev.append((t_clear, base_x)); cy_ev.append((t_clear, Y0))
    op_ev += blink(t_clear, T)
    parts.append(f'<rect x="{base_x}" y="{Y0}" width="{CW}" height="{LH}" '
                 f'fill="{th["cursor"]}" opacity=".85">'
                 f'{discrete("x", cx_ev)}{discrete("y", cy_ev)}'
                 f'{discrete("opacity", [(t, .85 if o else 0) for t, o in op_ev])}</rect>')

    # ---- life field (alternate screen) ----------------------------------
    for b, col in enumerate(th["ages"]):
        defs.append(f'<marker id="m{b}" markerUnits="userSpaceOnUse" markerWidth="1" '
                    f'markerHeight="1" overflow="visible" orient="0">'
                    f'<circle cx=".5" cy=".5" r=".37" fill="{col}"/></marker>')
    n = len(frames)
    frame_svg = []
    for k in range(n):
        F = frames[k][:QH, :QW]
        A = agef[k][:QH, :QW]
        by = {}
        for r, c in zip(*np.nonzero(F)):
            by.setdefault(bucket(A[r, c]), []).append((int(r), int(c)))
        # draw old first, newborn on top
        paths = "".join(f'<path class="m{b}" d="{cells_path(by[b])}"/>'
                        for b in sorted(by, reverse=True))
        t0 = gen_t[k]
        if k + 1 < n:
            t1 = gen_t[k + 1]
            ev = [(0, 0), (t0, 1), (t1, GHOST[0])]
            for j, gop in enumerate(GHOST[1:], 2):
                if k + j < n:
                    ev.append((gen_t[k + j], gop))
            ev.append((gen_t[k + len(GHOST) + 1] if k + len(GHOST) + 1 < n else t_exit, 0))
        else:
            ev = [(0, 0), (t0, 1), (t_exit, 0)]
        ev = [(t, v) for t, v in ev if t < T]
        frame_svg.append(f'<g opacity="0">{paths}{discrete("opacity", ev)}</g>')
    defs.append(f'<g id="Q">{"".join(frame_svg)}</g>')

    cx, cy = X0 + COLS * CW / 2, Y0 + FROWS * LH / 2
    defs.append(f'<radialGradient id="glow" cx="{cx}" cy="{cy}" r="300" '
                f'gradientUnits="userSpaceOnUse" gradientTransform="translate({cx} {cy}) '
                f'scale(1.9 1) translate({-cx} {-cy})">'
                f'<stop offset="0" stop-color="{th["glow"]}" stop-opacity="{th["glow_a"]}"/>'
                f'<stop offset=".6" stop-color="{th["glow"]}" stop-opacity="{th["glow_a"] * 0.3:.3f}"/>'
                f'<stop offset="1" stop-color="{th["glow"]}" stop-opacity="0"/></radialGradient>')
    glow = f'<rect x="1" y="{BAR + 1}" width="{W - 2}" height="{H - BAR - 2}" fill="url(#glow)"/>'
    field = (f'<g transform="translate({X0} {Y0}) scale({DX})">'
             f'<use href="#Q"/>'
             f'<use href="#Q" transform="matrix(-1 0 0 1 {GW} 0)"/>'
             f'<use href="#Q" transform="matrix(1 0 0 -1 0 {GH})"/>'
             f'<use href="#Q" transform="matrix(-1 0 0 -1 {GW} {GH})"/></g>')

    # ---- status line ------------------------------------------------------
    srow = ROWS - 1
    status = [text_uses("bloom.rle", 0, srow, th["dim"]),
              text_uses("B3/S23", 11, srow, th["faint"])]
    # counters: digit strips scrolled by discrete translate
    pops = [int(F.sum()) for F in frames]

    def counter(values, col, width, color):
        out = []
        for p in range(width):
            place = 10 ** (width - 1 - p)
            ev, prev = [], None
            for k, v in enumerate(values):
                dgt = 10 if (v < place and place > 1) else (v // place) % 10
                if dgt != prev:
                    ev.append((0 if k == 0 else gen_t[k], f"0 {-dgt * LH}"))
                    prev = dgt
            strip = "".join(f'<use href="{g(str(i))}" x="{X0 + (col + p) * CW}" '
                            f'y="{Y0 + (srow + i) * LH + BASE}"/>' for i in range(10))
            if len(ev) == 1:
                anim = ""
                tr = f' transform="translate({ev[0][1]})"'
            else:
                vals = ";".join(v for _, v in ev)
                keys = ";".join(kt(t) for t, _ in ev)
                anim = (f'<animateTransform attributeName="transform" type="translate" '
                        f'dur="{T}s" repeatCount="indefinite" calcMode="discrete" '
                        f'values="{vals}" keyTimes="{keys}"/>')
                tr = ""
            out.append(f'<g{tr}>{strip}{anim}</g>')
        return f'<g fill="{color}">{"".join(out)}</g>'

    gcol = COLS - 18
    status.append(text_uses("gen", gcol, srow, th["faint"]))
    status.append(counter(list(range(n)), gcol + 4, 3, th["dim"]))
    status.append(text_uses("pop", gcol + 9, srow, th["faint"]))
    status.append(counter(pops, gcol + 13, 4, th["dim"]))
    defs.append(f'<clipPath id="srow"><rect x="0" y="{Y0 + srow * LH}" width="{W}" '
                f'height="{LH}"/></clipPath>')

    # paint-in wipe, as the program draws its first screen
    rows_ev = [(0, 0)]
    for r in range(ROWS + 1):
        rows_ev.append((round(t_alt + r * 0.02, 3), r * LH))
    rows_ev.append((t_exit, 0))
    defs.append(f'<clipPath id="wipe"><rect x="0" y="{Y0}" width="{W}" height="0">'
                f'{discrete("height", rows_ev)}</rect></clipPath>')
    fade = [(0, 0), (t_alt, 0), (t_alt + 0.1, 1), (t_exit - 0.25, 1), (t_exit, 0)]
    parts.append(f'<g opacity="0">{glow}<g clip-path="url(#wipe)">{field}'
                 f'<g clip-path="url(#srow)">{"".join(status)}</g></g>'
                 f'{linear("opacity", fade)}</g>')

    for ch in sorted(used):
        defs.insert(0, f'<path id="c{ord(ch):x}" d="{glyph_d(ch)}"/>')
    style = "".join(f".m{i}{{marker:url(#m{i})}}" for i in range(len(th["ages"])))
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
            f'width="{W}" height="{H}" fill="none">'
            f'<style>{style}</style>'
            f'<defs>{"".join(defs)}</defs>{"".join(parts)}</svg>')


if __name__ == "__main__":
    frames = simulate(seed_field(*SEED), GENS)
    agef = ages(frames)
    for name in (sys.argv[1:] or THEMES):
        svg = build(name, frames, agef)
        p = OUT / f"banner-{name}.svg"
        p.write_text(svg)
        print(p.name, f"{len(svg.encode()) / 1024:.1f} KB",
              f"grid {GW}x{GH}", f"pop end {int(frames[-1].sum())}")
