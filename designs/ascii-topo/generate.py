#!/usr/bin/env python3
"""Terminal banner: `topo` flies over a contour map drawn in braille.

A shell types a command; the program takes the alternate screen and plots a
contour map with braille characters (2 x 4 dots per cell, the way terminal
map viewers do), flying slowly east over the terrain while a status line
reads out the position; then the shell comes back and `clear` resets it.

Contours are rasterised onto the braille dot grid like a line renderer
(flat runs scanned per dot column, steep runs per dot row), so every line is
exactly one dot thick.  Because braille dots sit on a uniform 4.5 px lattice,
re-rendering the map one dot further east is pixel-identical to shifting the
previous frame by one dot: the map is therefore stored once and stepped with
a discrete translate, one dot per frame.

Colour: contour dots are tinted by elevation (muted slate ramp, the summit
band in warm amber) and lit Tanaka-style, brighter on slopes facing the
upper-left light and dimmer on the far side, which gives the flat map relief.
Summits carry spot heights and a few index contours carry their elevation.

All text is Go Mono outlines; the dots are stamped via SVG markers, so the
whole map costs a couple of bytes per dot.

    python generate.py            # writes banner-dark.svg and banner-light.svg
"""
import math
import random
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
CW, LH = 9, 18                   # terminal cell, px
DP = CW / 2                      # braille dot pitch (= LH / 4)
FONT_PX = CW / 0.6               # Go Mono advance is 0.6 em
BAR = 36
PADX, PADY = 20, 12
X0 = PADX + 4
Y0 = BAR + PADY
COLS = (W - 2 * X0) // CW
ROWS = (H - Y0 - PADY) // LH
BASE = round(LH * 0.74, 2)
MAP_ROWS = ROWS - 1              # last row is the status line
GX, GY = 2 * COLS, 4 * MAP_ROWS  # view size in dots

# ---------------------------------------------------------------- timing
T = 24.0
FPS = 8
STEP_Y_EVERY = 0                 # one dot north every n frames (0: due east)

# ---------------------------------------------------------------- fonts
mono = TTFont(MONO)
mono_gs = mono.getGlyphSet()
mono_cmap = mono.getBestCmap()
MONO_UPEM = mono["head"].unitsPerEm


def fmt(v):
    s = f"{v:.2f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def glyph_d(ch, px=FONT_PX):
    pen = SVGPathPen(mono_gs, fmt)
    s = px / MONO_UPEM
    mono_gs[mono_cmap[ord(ch)]].draw(TransformPen(pen, (s, 0, 0, -s, 0, 0)))
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


# ---------------------------------------------------------------- terrain
SEED = 2


def terrain_params(seed):
    rng = np.random.default_rng(seed)
    feats = []
    for _ in range(9):
        lam = math.exp(rng.uniform(math.log(420), math.log(1300)))
        ang = rng.uniform(0, math.pi)
        k = 2 * math.pi / lam
        feats.append((k * math.cos(ang), k * math.sin(ang) * 1.3,
                      (lam / 1000) ** 1.1, rng.uniform(0, 2 * math.pi)))
    return feats


FEATS = terrain_params(SEED)
# massifs along the flight strip, world px: (x, y, sx, sy, rot, amp)
MASSIFS = [
    (380, 150, 170, 105, 0.35, 1.25),
    (980, 230, 210, 120, -0.25, 1.5),
    (1530, 110, 180, 115, 0.15, 1.15),
    (700, 420, 220, 120, 0.0, -0.6),
]


def field(X, Y):
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    wx = 26 * np.sin(Y / 210 + 1.1 * np.cos(X / 380))
    wy = 20 * np.sin(X / 260 + 0.8 * np.sin(Y / 300))
    X = X + wx
    Y = Y + wy
    f = np.zeros_like(X)
    for kx, ky, a, p in FEATS:
        f += 0.4 * a * np.cos(kx * X + ky * Y + p)
    for cx, cy, sx, sy, rot, amp in MASSIFS:
        c, s = math.cos(rot), math.sin(rot)
        dx, dy = X - cx, Y - cy
        u = (c * dx + s * dy) / sx
        v = (-s * dx + c * dy) / sy
        f += amp * np.exp(-0.5 * (u * u + v * v))
    return f


LEV = 0.115                 # contour interval (field units)
INDEX = 5                    # every 5th contour is an index contour
MIN_GAP = 3.0 * DP           # where contours crowd tighter, keep only index ones
M_PER_LEV = 20               # metres per contour interval (labels)
M_BASE = 1900                # metres at field level 0


LIGHT = (-0.6, -0.8)         # light comes from the upper left (screen NW)


def grad(X, Y, h=0.5):
    return ((field(X + h, Y) - field(X - h, Y)) / (2 * h),
            (field(X, Y + h) - field(X, Y - h)) / (2 * h))


def _crossings(F):
    """Level crossings along axis 0 of F: (fractional index, line, k)."""
    q = np.floor(F / LEV)
    pos, line, kk = [], [], []
    ii, jj = np.nonzero(q[1:] != q[:-1])
    for i, j in zip(ii, jj):
        a, b = F[i, j], F[i + 1, j]
        lo, hi = sorted((q[i, j], q[i + 1, j]))
        for k in range(int(lo) + 1, int(hi) + 1):
            pos.append(i + (k * LEV - a) / (b - a))
            line.append(j)
            kk.append(k)
    return np.array(pos), np.array(line, int), np.array(kk, int)


def rasterise_dots(nx, ny, sub=4):
    """Contour dots on the world dot lattice [0,nx) x [0,ny).
    Returns {(gx, gy): k}.  Dot (gx, gy) sits at world ((gx+.5)DP, (gy+.5)DP)."""
    dots = {}

    def put(gx, gy, k, err, lit):
        if 0 <= gx < nx and 0 <= gy < ny:
            old = dots.get((gx, gy))
            if old is None or err < old[1]:
                dots[(gx, gy)] = (k, err, lit)

    def light(fx, fy):
        """Tanaka illumination: +1 on slopes facing the light (NW), -1 away."""
        g = np.hypot(fx, fy) + 1e-12
        return (-fx * LIGHT[0] - fy * LIGHT[1]) / g

    # flat runs: scan each dot column, finely in y
    ys = (np.arange(-2 * sub, (ny + 2) * sub) + 0.5) / sub * DP
    xc = (np.arange(nx) + 0.5) * DP
    F = field(xc[None, :], ys[:, None])
    pos, col, kk = _crossings(F)
    y = ys[0] + pos * DP / sub
    x = xc[col]
    fx, fy = grad(x, y)
    gap = LEV / np.hypot(fx, fy)
    flat = np.abs(fx) <= np.abs(fy)
    lit = light(fx, fy)
    for i in np.nonzero(flat)[0]:
        if gap[i] < MIN_GAP and kk[i] % INDEX:
            continue
        gy = int(round(y[i] / DP - 0.5))
        put(col[i], gy, kk[i], abs(y[i] / DP - 0.5 - gy), lit[i])
    # steep runs: scan each dot row, finely in x
    xs = (np.arange(-2 * sub, (nx + 2) * sub) + 0.5) / sub * DP
    yc = (np.arange(ny) + 0.5) * DP
    F = field(xs[:, None], yc[None, :])
    pos, row, kk = _crossings(F)
    x = xs[0] + pos * DP / sub
    y = yc[row]
    fx, fy = grad(x, y)
    gap = LEV / np.hypot(fx, fy)
    steep = np.abs(fx) > np.abs(fy)
    lit = light(fx, fy)
    for i in np.nonzero(steep)[0]:
        if gap[i] < MIN_GAP and kk[i] % INDEX:
            continue
        gx = int(round(x[i] / DP - 0.5))
        put(gx, row[i], kk[i], abs(x[i] / DP - 0.5 - gx), lit[i])
    return {key: (v[0], v[2]) for key, v in dots.items()}


def tier_of(k, kmin, kmax):
    u = (k - kmin) / max(1, kmax - kmin)
    return min(NT - 1, int(u * NT))


def place_labels(dots):
    """Spot heights on summits and a few index-contour labels, in lattice
    coordinates (dot units).  Dots under each label are removed.
    Returns (labels, dots) with labels = [(text, cx, cy, kind)]."""
    gx = np.arange(NX)
    gy = np.arange(NY)
    Xp, Yp = np.meshgrid((gx + 0.5) * DP, (gy + 0.5) * DP)
    F = field(Xp, Yp)
    labels = []
    # summits: lattice maxima within a 24-dot neighbourhood
    R = 24
    pad = np.pad(F, R, constant_values=-1e9)
    mx = np.full_like(F, -1e9)
    for dy in range(-R, R + 1, 2):
        for dx in range(-R, R + 1, 2):
            mx = np.maximum(mx, pad[R + dy:R + dy + NY, R + dx:R + dx + NX])
    ks = [k for k, _ in dots.values()]
    kmin, kmax = min(ks), max(ks)
    top = min(k for k in range(kmin, kmax + 1) if tier_of(k, kmin, kmax) == NT - 1) * LEV
    for y, x in zip(*np.nonzero((F >= mx) & (F > top))):
        if 6 <= y < GY - 6 and 4 <= x < NX - 4:
            if any(abs(x - lx) < R and abs(y - ly) < R for _, lx, ly, _ in labels):
                continue
            m = M_BASE + F[y, x] / LEV * M_PER_LEV
            labels.append((f"{int(round(m))}", int(x), int(y), "peak"))
    # index contours: near-horizontal, uncluttered stretches, well apart
    cands = []
    for (x, y), (k, _) in dots.items():
        if k % INDEX or not (8 <= y < GY - 8) or not (6 <= x < NX - 10):
            continue
        fx, fy = grad(np.array([(x + .5) * DP]), np.array([(y + .5) * DP]))
        if abs(fy[0]) < 4 * abs(fx[0]):
            continue
        txt = str(M_BASE + k * M_PER_LEV)
        w = len(txt) * 2 + 2                    # dots
        box = [(xx, yy) for xx in range(x - w // 2, x + w // 2 + 1)
               for yy in range(y - 2, y + 3)]
        clutter = sum(1 for b in box if b in dots and dots[b][0] != k)
        if clutter:
            continue
        cands.append((x, y, txt))
    rng = random.Random(3)
    rng.shuffle(cands)
    for x, y, txt in cands:
        if all(abs(x - lx) > 70 or abs(y - ly) > 14 for _, lx, ly, _ in labels):
            if all(abs(x - lx) > 46 for _, lx, ly, kd in labels if kd == "contour"):
                labels.append((txt, x, y, "contour"))
    # clear the dots under every label
    keep = dict(dots)
    for txt, x, y, kind in labels:
        w = len(txt) + 0.6                      # half-width in dots (2 per char)
        for (dx, dy) in list(keep):
            if abs(dx - x) <= w and abs(dy - y) <= (2.6 if kind == "peak" else 1.6):
                del keep[(dx, dy)]
    return labels, keep


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


def mix(a, b, f):
    return from_oklab(to_oklab(a) * (1 - f) + to_oklab(b) * f)


NT = 6                       # elevation tiers
LIT_CUT = 0.3                # Tanaka: |illumination| beyond this is lit/shaded

THEMES = {
    "dark": dict(
        page="#0d1117", card="#0b0f15", bar="#11161d", border="#262c36",
        hair="#1b2129", title="#7d8590", fg="#c9d1d9", dim="#6e7681",
        faint="#3d4653", dir="#79c0ff", chev="#c297ff", cursor="#d6dbe1",
        lights=("#ff5f57", "#febc2e", "#28c840"),
        tiers=ramp(["#33465f", "#4a668d", "#7393b9", "#b3c6de"], NT - 1) + ["#eda769"],
        index_to="#f2f6fb", index_boost=0.3, label="#8291a6",
        lit_to="#eef3fa", lit=0.38, shade_to="#0b0f15", shade=0.4,
    ),
    "light": dict(
        page="#ffffff", card="#fbfbfc", bar="#f3f4f6", border="#d8dee4",
        hair="#e6e8ec", title="#6e7781", fg="#24292f", dim="#8c959f",
        faint="#c4cad2", dir="#0969da", chev="#8250df", cursor="#24292f",
        lights=("#ff5f57", "#febc2e", "#28c840"),
        tiers=ramp(["#bcc6d3", "#8e9eb4", "#62748e", "#3e4f68"], NT - 1) + ["#c8682e"],
        index_to="#0f1722", index_boost=0.3, label="#5f6d80",
        lit_to="#fbfbfc", lit=0.3, shade_to="#0b1018", shade=0.12,
    ),
}


# ---------------------------------------------------------------- timeline
def kt(t):
    return f"{t / T:.5f}".rstrip("0").rstrip(".") if t else "0"


def discrete(attr, events, extra=""):
    assert events[0][0] == 0
    vals = ";".join(str(v) for _, v in events)
    keys = ";".join(kt(t) for t, _ in events)
    return (f'<animate attributeName="{attr}" dur="{T}s" repeatCount="indefinite" '
            f'calcMode="discrete" values="{vals}" keyTimes="{keys}"{extra}/>')


def typing_times(text, start, rng):
    times = []
    t = start
    prev = ""
    for ch in text:
        d = rng.uniform(0.05, 0.09)
        if ch == " ":
            d = rng.uniform(0.07, 0.11)
        if prev == " ":
            d += rng.uniform(0.02, 0.06)
        if ch in "-./" and prev != ch:
            d += rng.uniform(0.04, 0.08)
        if ch == prev and ch in "-":
            d = rng.uniform(0.035, 0.05)
        t += d
        times.append(round(t, 3))
        prev = ch
    return times


PROMPT_DIR = "~/src"
PROMPT_COLS = len(PROMPT_DIR) + 3
CMD = "topo --fly east"
_rng = random.Random(5)
T_TYPE0 = 0.6
TTIMES = typing_times(CMD, T_TYPE0, _rng)
T_ENTER = TTIMES[-1] + 0.4
T_ALT = T_ENTER + 0.18
T_EXIT = 20.7
T_PROMPT2 = T_EXIT + 0.05
CTIMES = typing_times("clear", T_PROMPT2 + 0.7, _rng)
T_ENTER2 = CTIMES[-1] + 0.38
T_CLEAR = T_ENTER2 + 0.06
N_FRAMES = int(math.floor((T_EXIT - T_ALT) * FPS))


NX = GX + N_FRAMES + 2
NY = GY + (N_FRAMES // STEP_Y_EVERY + 2 if STEP_Y_EVERY else 0)


def frame_offset(i):
    """View origin (top-left) in lattice dots at frame i: east, maybe drifting north."""
    if not STEP_Y_EVERY:
        return i, 0
    return i, (N_FRAMES - 1) // STEP_Y_EVERY - i // STEP_Y_EVERY


# status line: position read-out (decimal degrees) of the view centre
LAT0, LON0 = 61.6342, 8.1517
M_PER_PX = 2.0
STATUS_EVERY = 2             # status line refreshes every n frames


def status_pos(i):
    ox, oy = frame_offset(i)
    xm = (ox + GX / 2) * DP * M_PER_PX
    ym = (oy + GY / 2) * DP * M_PER_PX
    lat = LAT0 - ym / 111320
    lon = LON0 + xm / (111320 * math.cos(math.radians(LAT0)))
    return f"{lat:.4f}°N {lon:.4f}°E"


HEADING = round(math.degrees(math.atan2(1, 1 / STEP_Y_EVERY))) if STEP_Y_EVERY else 90


# ---------------------------------------------------------------- build
def chain_path(pts):
    """Relative path through the given lattice points (marker vertices)."""
    pts = sorted(pts, key=lambda p: (p[1], p[0]))
    out, px, py = [], None, None
    for x, y in pts:
        if px is None:
            out.append(f"M{x} {y}")
        elif y == py:
            out.append(f"h{x - px}")
        else:
            out.append(f"m{x - px} {y - py}")
        px, py = x, y
    return "".join(out)


def build(theme_name, dots, labels):
    th = THEMES[theme_name]
    ks = [k for k, _ in dots.values()]
    kmin, kmax = min(ks), max(ks)
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
         [(0, "visible"), (T_ALT, "hidden"), (T_EXIT, "visible")]),
        (f"{PROMPT_DIR} — topo — {size}",
         [(0, "hidden"), (T_ALT, "visible"), (T_EXIT, "hidden")]),
    ):
        d, wpx = sans_text_d(label, 12.5, 0, 0)
        tx = (W - wpx) / 2
        parts.append(f'<path transform="translate({tx:.1f} {BAR / 2 + 4.4})" d="{d}" '
                     f'fill="{th["title"]}">{discrete("visibility", vis)}</path>')

    # ---- shell ------------------------------------------------------------
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
    clip_typed("k0", 0, TTIMES, PROMPT_COLS, T_CLEAR)
    normal.append(f'<g clip-path="url(#k0)">{text_uses(CMD, PROMPT_COLS, 0, th["fg"])}</g>')
    normal.append(f'<g opacity="0">{prompt(1)}'
                  f'{discrete("opacity", [(0, 0), (T_PROMPT2, 1), (T_CLEAR, 0)])}</g>')
    clip_typed("k1", 1, CTIMES, PROMPT_COLS, T_CLEAR)
    normal.append(f'<g clip-path="url(#k1)">{text_uses("clear", PROMPT_COLS, 1, th["fg"])}</g>')
    parts.append(f'<g>{"".join(normal)}'
                 f'{discrete("opacity", [(0, 1), (T_ALT, 0), (T_EXIT, 1)])}</g>')

    # ---- cursor -------------------------------------------------------------
    def blink(t0, t1, on=True):
        ev, t = [], t0
        while t < t1 - 1e-6:
            ev.append((round(t, 3), 1 if on else 0))
            t += 0.53
            on = not on
        return ev

    base_x = X0 + PROMPT_COLS * CW
    cx_ev, cy_ev, op_ev = [(0, base_x)], [(0, Y0)], []
    op_ev += blink(0, T_TYPE0 - 0.1)
    op_ev.append((round(T_TYPE0 - 0.1, 3), 1))
    for i, t in enumerate(TTIMES):
        cx_ev.append((t, base_x + (i + 1) * CW))
    cx_ev.append((T_ENTER, X0)); cy_ev.append((T_ENTER, Y0 + LH))
    op_ev.append((T_ALT, 0))
    cx_ev.append((T_PROMPT2, base_x)); cy_ev.append((T_PROMPT2, Y0 + LH))
    op_ev += blink(T_PROMPT2, CTIMES[0] - 0.05)
    op_ev.append((round(CTIMES[0] - 0.05, 3), 1))
    for i, t in enumerate(CTIMES):
        cx_ev.append((t, base_x + (i + 1) * CW))
    cx_ev.append((T_ENTER2, X0)); cy_ev.append((T_ENTER2, Y0 + 2 * LH))
    cx_ev.append((T_CLEAR, base_x)); cy_ev.append((T_CLEAR, Y0))
    op_ev += blink(T_CLEAR, T)
    parts.append(f'<rect x="{base_x}" y="{Y0}" width="{CW}" height="{LH}" '
                 f'fill="{th["cursor"]}" opacity=".85">'
                 f'{discrete("x", cx_ev)}{discrete("y", cy_ev)}'
                 f'{discrete("opacity", [(t, .85 if o else 0) for t, o in op_ev])}</rect>')

    # ---- status line ---------------------------------------------------------
    srow = ROWS - 1
    stat = []
    times = [T_ALT + i / FPS for i in range(0, N_FRAMES, STATUS_EVERY)]
    strs = [status_pos(i) for i in range(0, N_FRAMES, STATUS_EVERY)]
    n = len(strs[0])
    assert all(len(x) == n for x in strs)
    sy = Y0 + srow * LH + BASE
    for j in range(n):
        seq = [x[j] for x in strs]
        if len(set(seq)) == 1:
            stat.append(text_uses(seq[0], j, srow))
            continue
        # odometer: the column's glyphs stacked in a strip, stepped into view
        chars = sorted(set(seq))
        strip = "".join(f'<use href="{g(ch)}" x="{X0 + j * CW}" y="{sy + m * LH}"/>'
                        for m, ch in enumerate(chars))
        ev = [(0, 0)]
        for t, ch in zip(times, seq):
            v = -chars.index(ch) * LH
            if v != ev[-1][1]:
                ev.append((t, v))
        vals = ";".join(f"0 {v}" for _, v in ev)
        keys = ";".join(kt(t) for t, _ in ev)
        stat.append(f'<g>{strip}<animateTransform attributeName="transform" '
                    f'type="translate" dur="{T}s" repeatCount="indefinite" '
                    f'calcMode="discrete" values="{vals}" keyTimes="{keys}"/></g>')
    hdg = f"hdg {HEADING:03d}°"
    stat.append(text_uses(hdg, n + 3, srow))
    bar_m = 200
    bar_cols = round(bar_m / M_PER_PX / CW)
    scale = "├" + "─" * (bar_cols - 2) + "┤" + f" {bar_m} m"
    stat.append(text_uses(scale, COLS - len(scale), srow))
    defs.append(f'<clipPath id="srow"><rect x="0" y="{Y0 + srow * LH}" width="{W}" '
                f'height="{LH}"/></clipPath>')
    status_g = f'<g clip-path="url(#srow)" fill="{th["dim"]}">{"".join(stat)}</g>'

    # ---- the map: one static dot lattice, stepped one dot per frame ---------
    groups = {}
    for (x, y), (k, lit) in dots.items():
        t = tier_of(k, kmin, kmax)
        lv = 0 if lit < -LIT_CUT else (2 if lit > LIT_CUT else 1)
        groups.setdefault((t, k % INDEX == 0, lv), []).append((x, y))
    map_paths = []
    for (t, idx, lv), pts in sorted(groups.items()):
        col = th["tiers"][t]
        if idx and t < NT - 1:
            col = mix(col, th["index_to"], th["index_boost"])
        if lv == 2:
            col = mix(col, th["lit_to"], th["lit"])
        elif lv == 0:
            col = mix(col, th["shade_to"], th["shade"])
        r = 0.29 if idx else 0.24           # in dot units
        mid = f"d{t}{'i' if idx else ''}{lv}"
        defs.append(f'<marker id="{mid}" markerUnits="userSpaceOnUse" markerWidth="1" '
                    f'markerHeight="1" overflow="visible" orient="0">'
                    f'<circle r="{r}" fill="{col}"/></marker>')
        map_paths.append(f'<path marker-start="url(#{mid})" marker-mid="url(#{mid})" '
                         f'marker-end="url(#{mid})" d="{chain_path(pts)}"/>')
    steps = []
    for i in range(N_FRAMES):
        dx, dy = frame_offset(i)
        steps.append((T_ALT + i / FPS, f"{-dx} {-dy}"))
    ev = [(0, "0 0")] + steps
    tr = (f'<animateTransform attributeName="transform" type="translate" dur="{T}s" '
          f'repeatCount="indefinite" calcMode="discrete" '
          f'values="{";".join(v for _, v in ev)}" '
          f'keyTimes="{";".join(kt(t) for t, _ in ev)}"/>')
    mx0, my0 = X0 + DP / 2, Y0 + DP / 2
    defs.append(f'<clipPath id="view"><rect x="{X0}" y="{Y0}" width="{COLS * CW}" '
                f'height="{MAP_ROWS * LH}"/></clipPath>')
    # paint-in: the first frame arrives row by row, like a terminal redraw
    rows_ev = [(0, 0)]
    for r in range(ROWS + 1):
        rows_ev.append((round(T_ALT + r * 0.02, 3), r * LH))
    rows_ev.append((T_EXIT, 0))
    defs.append(f'<clipPath id="wipe"><rect x="0" y="{Y0}" width="{W}" height="0">'
                f'{discrete("height", rows_ev)}</rect></clipPath>')
    lab = []
    ink = 5.2                                   # digit ink centre above baseline
    for txt, x, y, kind in labels:
        col = th["tiers"][-1] if kind == "peak" else th["label"]
        lx = x * DP - len(txt) * CW / 2
        ly = y * DP + ink
        lab.append(f'<g fill="{col}">' + "".join(
            f'<use href="{g(ch)}" x="{fmt(lx + i * CW)}" y="{fmt(ly)}"/>'
            for i, ch in enumerate(txt)) + "</g>")
    map_g = (f'<g clip-path="url(#view)"><g transform="translate({fmt(mx0)} {fmt(my0)}) '
             f'scale({DP})"><g>{"".join(map_paths)}'
             f'<g transform="scale({fmt(1 / DP)})">{"".join(lab)}</g>{tr}</g></g></g>')
    parts.append(f'<g clip-path="url(#wipe)">{map_g}{status_g}</g>')

    for ch in sorted(used):
        defs.insert(0, f'<path id="c{ord(ch):x}" d="{glyph_d(ch)}"/>')

    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
           f'width="{W}" height="{H}" fill="none">'
           f'<defs>{"".join(defs)}</defs>{"".join(parts)}</svg>')
    return svg


if __name__ == "__main__":
    dots = rasterise_dots(NX, NY)
    labels, dots = place_labels(dots)
    ks = [k for k, _ in dots.values()]
    print(f"{N_FRAMES} frames, lattice {NX}x{NY}, {len(dots)} dots, k {min(ks)}..{max(ks)}")
    for name in THEMES:
        svg = build(name, dots, labels)
        p = OUT / f"banner-{name}.svg"
        p.write_text(svg)
        print(p.name, f"{len(svg.encode()) / 1024:.1f} KB")
