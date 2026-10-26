#!/usr/bin/env python3
"""Animated terminal banner: a shell types a command, then a shaded ASCII torus
tumbles in the alternate screen, then the shell returns and `clear` resets it.

All glyphs are Go Mono outlines. The torus frames are precomputed; every frame
is one <path> per luminance level whose vertices stamp a glyph marker, so a
frame costs ~2-3 bytes per character.

    python generate.py            # writes banner-dark.svg and banner-light.svg
"""
import math
import random
import sys
import warnings
from pathlib import Path

import numpy as np
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont

warnings.filterwarnings("ignore", category=RuntimeWarning)
OUT = Path(__file__).resolve().parent
MONO = "/usr/share/fonts/fonts-go/Go-Mono.ttf"
SANS = "/usr/share/fonts/opentype/inter/Inter-Medium.otf"

# ---------------------------------------------------------------- geometry
W, H = 1200, 400
CW, LH = 9, 18                    # terminal cell, px (integer so paths stay tiny)
FONT_PX = CW / 0.6                # Go Mono advance is 0.6 em
BAR = 36                          # title bar height
PADX, PADY = 20, 12
X0 = PADX + 4                     # left edge of column 0 (cells start here)
Y0 = BAR + PADY                   # top edge of row 0
COLS = (W - 2 * X0) // CW
ROWS = (H - Y0 - PADY) // LH
BASE = round(LH * 0.74, 2)        # baseline offset inside a cell

T = 18.0                          # master loop, s
FPS_FRAMES = 72                   # frames per full precession cycle
SPIN = 4.5                        # seconds per cycle (T is a multiple)
AXIS = (1.0, 0.25, 0.62)          # tumble axis in torus coords (z = symmetry axis)
RAMP = ".,-~:;=!*#$@"

# ---------------------------------------------------------------- fonts
mono = TTFont(MONO)
mono_gs = mono.getGlyphSet()
mono_cmap = mono.getBestCmap()
MONO_UPEM = mono["head"].unitsPerEm


def fmt(v):
    s = f"{v:.2f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def glyph_d(ch, px=FONT_PX):
    """Outline of `ch` at size px, origin at (left, baseline)."""
    name = mono_cmap[ord(ch)]
    pen = SVGPathPen(mono_gs, fmt)
    s = px / MONO_UPEM
    mono_gs[name].draw(TransformPen(pen, (s, 0, 0, -s, 0, 0)))
    return pen.getCommands()


def sans_text_d(text, px, x, y):
    """Shaped Inter text as one path, left-aligned at (x, baseline y)."""
    import uharfbuzz as hb
    blob = hb.Blob.from_file_path(SANS)
    face = hb.Face(blob)
    font = hb.Font(face)
    buf = hb.Buffer()
    buf.add_str(text)
    buf.guess_segment_properties()
    hb.shape(font, buf, {"kern": True, "liga": True})
    tt = TTFont(SANS)
    gs = tt.getGlyphSet()
    order = tt.getGlyphOrder()
    upem = tt["head"].unitsPerEm
    s = px / upem
    pen = SVGPathPen(gs, fmt)
    cx = 0
    for info, pos in zip(buf.glyph_infos, buf.glyph_positions):
        gx = x + (cx + pos.x_offset) * s
        gy = y - pos.y_offset * s
        gs[order[info.codepoint]].draw(TransformPen(pen, (s, 0, 0, -s, gx, gy)))
        cx += pos.x_advance
    return pen.getCommands(), cx * s


# ---------------------------------------------------------------- colour
def hex2rgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)])


def srgb2lin(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def lin2srgb(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
               [0.2119034982, 0.6806995451, 0.1073969566],
               [0.0883024619, 0.2817188376, 0.6299787005]])
M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
               [1.9779984951, -2.4285922050, 0.4505937099],
               [0.0259040371, 0.7827717662, -0.8086757660]])


def to_oklab(h):
    return M2 @ np.cbrt(M1 @ srgb2lin(hex2rgb(h)))


def from_oklab(lab):
    lms = np.linalg.inv(M2) @ lab
    rgb = lin2srgb(np.linalg.inv(M1) @ (lms ** 3))
    return "#" + "".join(f"{int(round(v * 255)):02x}" for v in rgb)


def ramp(stops, n):
    labs = [to_oklab(s) for s in stops]
    out = []
    for i in range(n):
        u = i / (n - 1) * (len(labs) - 1)
        k = min(int(u), len(labs) - 2)
        f = u - k
        out.append(from_oklab(labs[k] * (1 - f) + labs[k + 1] * f))
    return out


THEMES = {
    "dark": dict(
        page="#0d1117", card="#0b0f15", bar="#11161d", border="#262c36",
        hair="#1b2129", title="#7d8590", fg="#c9d1d9", dim="#6e7681",
        dir="#79c0ff", chev="#c297ff", cursor="#d6dbe1",
        lights=("#ff5f57", "#febc2e", "#28c840"),
        ramp=ramp(["#1d2440", "#34437e", "#5a72c8", "#9fb0ee", "#e9e6f5",
                   "#ffe3c4"], len(RAMP)),
        glow="#4a5fd0", glow_a=0.16, glow_r=230,
    ),
    "light": dict(
        page="#ffffff", card="#fbfbfc", bar="#f3f4f6", border="#d8dee4",
        hair="#e6e8ec", title="#6e7781", fg="#24292f", dim="#8c959f",
        dir="#0969da", chev="#8250df", cursor="#24292f",
        lights=("#ff5f57", "#febc2e", "#28c840"),
        ramp=ramp(["#dfe3f3", "#b5bfe6", "#7d8fd6", "#4b5cb8", "#2a2f6e",
                   "#16152e"], len(RAMP)),
        glow="#7d8fd6", glow_a=0.10, glow_r=230,
    ),
}

# ---------------------------------------------------------------- torus
def render_torus(n_frames, cols, rows, seed=0):
    """Return list of frames; each frame is a dict level -> list of (col,row)."""
    R1, R2 = 1.0, 2.05
    SS = 4                               # subcells per cell, each axis
    nt, npph = 720, 360                  # samples around tube / around ring
    th = np.linspace(0, 2 * np.pi, nt, endpoint=False)
    ph = np.linspace(0, 2 * np.pi, npph, endpoint=False)
    TH, PH = np.meshgrid(th, ph, indexing="ij")
    TH = TH.ravel(); PH = PH.ravel()
    # torus with symmetry axis z
    cx = R2 + R1 * np.cos(TH)
    P = np.stack([cx * np.cos(PH), cx * np.sin(PH), R1 * np.sin(TH)], 1)
    N = np.stack([np.cos(TH) * np.cos(PH), np.cos(TH) * np.sin(PH), np.sin(TH)], 1)

    def rx(a):
        c, s = math.cos(a), math.sin(a)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    def ry(a):
        c, s = math.cos(a), math.sin(a)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    def rot(axis, a):
        k = np.array(axis, float); k /= np.linalg.norm(k)
        K_ = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        return np.eye(3) + math.sin(a) * K_ + (1 - math.cos(a)) * K_ @ K_

    def rz(a):
        c, s = math.cos(a), math.sin(a)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    # tumble about a tilted axis (precesses; period 2pi), then a fixed
    # view rotation so the motion reads as a diagonal wobble.
    VIEW = rz(math.radians(-28)) @ rx(math.radians(-22)) @ ry(math.radians(18))
    light = np.array([-0.38, 0.55, -1.0]); light /= np.linalg.norm(light)
    fill = np.array([0.7, -0.35, -0.6]); fill /= np.linalg.norm(fill)
    half = light - np.array([0, 0, 1.0]); half /= np.linalg.norm(half)

    D = 9.0                               # camera distance
    unit_rows = (rows - 0.6) / 2 / 3.05   # rows per unit at depth D (fit +-3.05)
    K = unit_rows * D
    aspect = LH / CW
    gw, gh = cols * SS, rows * SS
    frames = []
    for f in range(n_frames):
        a = 2 * math.pi * f / n_frames
        Mx = VIEW @ rot(AXIS, a)
        p = P @ Mx.T
        n = N @ Mx.T
        z = p[:, 2] + D
        # screen coords in cell units
        u = cols / 2 + K * aspect * p[:, 0] / z
        v = rows / 2 - K * p[:, 1] / z
        gx = np.floor(u * SS).astype(int)
        gy = np.floor(v * SS).astype(int)
        ok = (gx >= 0) & (gx < gw) & (gy >= 0) & (gy < gh)
        # visible only if facing camera roughly (helps z-buffer at silhouette)
        diff = np.clip(n @ light, 0, 1)
        spec = np.clip(n @ half, 0, 1) ** 24
        rim = np.clip(n @ fill, 0, 1)
        lum = 0.10 + 0.74 * diff ** 1.25 + 0.18 * rim + 0.32 * spec
        idx = gy * gw + gx
        idx = idx[ok]; zz = z[ok]; ll = lum[ok]
        order = np.lexsort((zz, idx))
        idx_s = idx[order]
        first = np.ones(len(idx_s), bool)
        first[1:] = idx_s[1:] != idx_s[:-1]
        sel = order[first]
        buf_l = np.full(gw * gh, np.nan)
        buf_l[idx[sel]] = ll[sel]
        buf_l = buf_l.reshape(rows, SS, cols, SS)
        cov = np.isfinite(buf_l).mean(axis=(1, 3))
        with np.errstate(all="ignore"):
            mean_l = np.nanmean(buf_l, axis=(1, 3))
        cells = {}
        for r in range(rows):
            for c in range(cols):
                cv = cov[r, c]
                if cv < 0.34:
                    continue
                L = mean_l[r, c] * (0.55 + 0.45 * min(1, cv / 0.8))
                lev = int(np.clip(L, 0, 0.999) * len(RAMP))
                cells.setdefault(lev, []).append((c, r))
        frames.append(cells)
    return frames


def cells_path(cells, ox, oy):
    """Path whose vertices sit on each cell's (left, baseline)."""
    cells = sorted(cells, key=lambda cr: (cr[1], cr[0]))
    out = []
    px = py = None
    for c, r in cells:
        x = ox + c * CW
        y = oy + r * LH
        if px is None:
            out.append(f"M{x} {y}")
        elif y == py:
            out.append(f"h{x - px}")
        else:
            out.append(f"m{x - px} {y - py}")
        px, py = x, y
    return "".join(out)


# ---------------------------------------------------------------- timeline
def kt(t):
    return f"{t / T:.5f}".rstrip("0").rstrip(".") if t else "0"


def discrete(attr, events, extra=""):
    """events: [(t, value)], t ascending, first t must be 0."""
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
    """Natural cadence: per-char times at which char i appears."""
    times = []
    t = start
    prev = ""
    for ch in text:
        d = rng.uniform(0.045, 0.085)
        if ch == " ":
            d = rng.uniform(0.07, 0.11)
        if prev == " ":
            d += rng.uniform(0.02, 0.06)        # a beat after each word
        if ch in "&-./" and prev != ch:
            d += rng.uniform(0.04, 0.08)        # reach for a symbol
        if ch == prev and ch in "&-":
            d = rng.uniform(0.035, 0.05)        # repeated keys are quick
        t += d
        times.append(round(t, 3))
        prev = ch
    return times


# ---------------------------------------------------------------- build
def build(theme_name):
    th = THEMES[theme_name]
    rng = random.Random(7)
    parts = []
    defs = []

    # glyph defs (fill inherited)
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
    PROMPT_COLS = len(PROMPT_DIR) + 3         # "~/src ❯ "
    CMD = "make torus && ./torus"
    MAKE_OUT = "cc -O3 -march=native -o torus torus.c -lm"
    t_type0 = 0.55
    ttimes = typing_times(CMD, t_type0, rng)
    t_enter = ttimes[-1] + 0.42
    t_make = t_enter + 0.12
    t_alt = t_make + 0.62
    t_exit = 15.3
    t_prompt2 = t_exit + 0.05
    ctimes = typing_times("clear", t_prompt2 + 0.75, rng)
    t_enter2 = ctimes[-1] + 0.38
    t_clear = t_enter2 + 0.06

    # ---- card + chrome --------------------------------------------------
    parts.append(f'<rect x=".5" y=".5" width="{W - 1}" height="{H - 1}" rx="14" '
                 f'fill="{th["card"]}" stroke="{th["border"]}"/>')
    parts.append(f'<path d="M.5 {BAR}V14.5A14 14 0 0 1 14.5 .5H{W - 14.5}'
                 f'A14 14 0 0 1 {W - .5} 14.5V{BAR}Z" fill="{th["bar"]}"/>')
    parts.append(f'<path d="M1 {BAR}.5H{W - 1}" stroke="{th["hair"]}"/>')
    for i, col in enumerate(th["lights"]):
        parts.append(f'<circle cx="{22 + i * 20}" cy="{BAR / 2}" r="6" fill="{col}"/>')
    # title switches with the foreground process
    size = f"{COLS}×{ROWS}"
    for label, vis in (
        (f"{PROMPT_DIR} — zsh — {size}",
         [(0, "visible"), (t_alt, "hidden"), (t_exit, "visible")]),
        (f"{PROMPT_DIR} — torus — {size}",
         [(0, "hidden"), (t_alt, "visible"), (t_exit, "hidden")]),
    ):
        d, wpx = sans_text_d(label, 12.5, 0, 0)
        tx = (W - wpx) / 2
        parts.append(f'<path transform="translate({tx:.1f} {BAR / 2 + 4.4})" d="{d}" '
                     f'fill="{th["title"]}">{discrete("visibility", vis)}</path>')

    # ---- prompt glyph (custom chevron, drawn in cell) -------------------
    def chevron(col, row):
        x = X0 + col * CW + 1.2
        y = Y0 + row * LH + LH / 2 - 0.3
        return (f'<path d="M{x:.1f} {y - 3.6:.1f}l3.6 3.6-3.6 3.6" fill="none" '
                f'stroke="{th["chev"]}" stroke-width="1.6" stroke-linecap="round" '
                f'stroke-linejoin="round"/>')

    def prompt(row):
        return text_uses(PROMPT_DIR, 0, row, th["dir"]) + chevron(len(PROMPT_DIR) + 1, row)

    def clip_typed(cid, row, times, col0, t_off):
        """clipPath that reveals typed chars at `times`, hides at t_off."""
        ev = [(0, 0)] + [(t, (i + 1) * CW) for i, t in enumerate(times)] + [(t_off, 0)]
        x = X0 + col0 * CW
        defs.append(f'<clipPath id="{cid}"><rect x="{x}" y="{Y0 + row * LH}" width="0" '
                    f'height="{LH}">{discrete("width", ev)}</rect></clipPath>')

    normal = []
    # line 0: prompt, typed command
    normal.append(prompt(0))
    clip_typed("k0", 0, ttimes, PROMPT_COLS, t_clear)
    normal.append(f'<g clip-path="url(#k0)">{text_uses(CMD, PROMPT_COLS, 0, th["fg"])}</g>')
    # line 1: make echo (appears after enter, gone after clear)
    normal.append(f'<g fill="{th["dim"]}" opacity="0">'
                  f'{text_uses(MAKE_OUT, 0, 1)}'
                  f'{discrete("opacity", [(0, 0), (t_make, 1), (t_clear, 0)])}</g>')
    # line 2: prompt after torus exits, types `clear`
    normal.append(f'<g opacity="0">{prompt(2)}'
                  f'{discrete("opacity", [(0, 0), (t_prompt2, 1), (t_clear, 0)])}</g>')
    clip_typed("k2", 2, ctimes, PROMPT_COLS, t_clear)
    normal.append(f'<g clip-path="url(#k2)">{text_uses("clear", PROMPT_COLS, 2, th["fg"])}</g>')

    parts.append(f'<g>{"".join(normal)}'
                 f'{discrete("opacity", [(0, 1), (t_alt, 0), (t_exit, 1)])}</g>')

    # ---- cursor ---------------------------------------------------------
    def blink(t0, t1, phase_on=True):
        ev = []
        t = t0
        on = phase_on
        while t < t1 - 1e-6:
            ev.append((round(t, 3), 1 if on else 0))
            t += 0.53
            on = not on
        return ev

    cx_ev, op_ev = [], []
    base_x = X0 + PROMPT_COLS * CW
    cx_ev.append((0, base_x))
    cy_ev = [(0, Y0)]
    op_ev += blink(0, t_type0 - 0.1)
    op_ev.append((round(t_type0 - 0.1, 3), 1))
    for i, t in enumerate(ttimes):
        cx_ev.append((t, base_x + (i + 1) * CW))
    # enter: cursor to next line col 0, then below make output
    cx_ev.append((t_enter, X0)); cy_ev.append((t_enter, Y0 + LH))
    cy_ev.append((t_make, Y0 + 2 * LH))
    op_ev.append((t_alt, 0))
    # back at prompt line 2
    cx_ev.append((t_prompt2, base_x)); cy_ev.append((t_prompt2, Y0 + 2 * LH))
    op_ev += blink(t_prompt2, ctimes[0] - 0.05)
    op_ev.append((round(ctimes[0] - 0.05, 3), 1))
    for i, t in enumerate(ctimes):
        cx_ev.append((t, base_x + (i + 1) * CW))
    cx_ev.append((t_enter2, X0)); cy_ev.append((t_enter2, Y0 + 3 * LH))
    cx_ev.append((t_clear, base_x)); cy_ev.append((t_clear, Y0))
    op_ev += blink(t_clear, T)
    # keep last blink phase consistent with t=0 (on) -> fine, short gap
    parts.append(f'<rect x="{base_x}" y="{Y0}" width="{CW}" height="{LH}" '
                 f'fill="{th["cursor"]}" opacity=".85">'
                 f'{discrete("x", cx_ev)}{discrete("y", cy_ev)}'
                 f'{discrete("opacity", [(t, .85 if o else 0) for t, o in op_ev])}</rect>')

    # ---- torus in the alternate screen ---------------------------------
    frames = render_torus(FPS_FRAMES, COLS, ROWS)
    for lev, col in enumerate(th["ramp"]):
        defs.append(f'<marker id="m{lev}" markerUnits="userSpaceOnUse" markerWidth="1" '
                    f'markerHeight="1" overflow="visible" orient="0">'
                    f'<use href="{g(RAMP[lev])}" fill="{col}"/></marker>')
    frame_svg = []
    n = len(frames)
    for i, cells in enumerate(frames):
        paths = []
        for lev in sorted(cells):
            d = cells_path(cells[lev], X0, Y0 + BASE)
            paths.append(f'<path class="m{lev}" d="{d}"/>')
        if i == 0:
            vals, keys = "visible;hidden", f"0;{1 / n:.5f}"
        else:
            vals, keys = "hidden;visible;hidden", f"0;{i / n:.5f};{(i + 1) / n:.5f}"
        frame_svg.append(
            f'<g visibility="{"visible" if i == 0 else "hidden"}">{"".join(paths)}'
            f'<animate attributeName="visibility" dur="{SPIN}s" repeatCount="indefinite" '
            f'calcMode="discrete" values="{vals}" keyTimes="{keys}"/></g>')
    fade = [(0, 0), (t_alt, 0), (t_alt + 0.12, 1), (t_exit - 0.3, 1), (t_exit, 0)]
    gcx, gcy = X0 + COLS * CW / 2, Y0 + ROWS * LH / 2
    defs.append(f'<radialGradient id="glow" cx="{gcx}" cy="{gcy}" r="{th["glow_r"]}" '
                f'gradientUnits="userSpaceOnUse" gradientTransform="translate({gcx} {gcy}) '
                f'scale(1.25 1) translate({-gcx} {-gcy})">'
                f'<stop offset="0" stop-color="{th["glow"]}" stop-opacity="{th["glow_a"]}"/>'
                f'<stop offset=".55" stop-color="{th["glow"]}" stop-opacity="{th["glow_a"] * 0.35:.3f}"/>'
                f'<stop offset="1" stop-color="{th["glow"]}" stop-opacity="0"/></radialGradient>')
    glow = (f'<rect x="1" y="{BAR + 1}" width="{W - 2}" height="{H - BAR - 2}" '
            f'fill="url(#glow)"/>')
    # paint-in: the first frames arrive row by row, like a terminal redraw
    rows_ev = [(0, 0)]
    for r in range(ROWS + 1):
        rows_ev.append((round(t_alt + r * 0.022, 3), r * LH))
    rows_ev.append((t_exit, 0))
    defs.append(f'<clipPath id="wipe"><rect x="0" y="{Y0}" width="{W}" height="0">'
                f'{discrete("height", rows_ev)}</rect></clipPath>')
    parts.append(f'<g fill="none" opacity="0">{glow}'
                 f'<g clip-path="url(#wipe)">{"".join(frame_svg)}</g>'
                 f'{linear("opacity", fade)}</g>')

    # ---- glyph defs ------------------------------------------------------
    for ch in sorted(used):
        defs.insert(0, f'<path id="c{ord(ch):x}" d="{glyph_d(ch)}"/>')

    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
           f'width="{W}" height="{H}" fill="none">'
           f'<style>{"".join(f".m{i}{{marker:url(#m{i})}}" for i in range(len(RAMP)))}</style>'
           f'<defs>{"".join(defs)}</defs>{"".join(parts)}</svg>')
    return svg


if __name__ == "__main__":
    for name in (sys.argv[1:] or THEMES):
        svg = build(name)
        p = OUT / f"banner-{name}.svg"
        p.write_text(svg)
        print(p.name, f"{len(svg.encode()) / 1024:.1f} KB", f"{COLS}x{ROWS}")
