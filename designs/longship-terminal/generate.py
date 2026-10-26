#!/usr/bin/env python3
"""Terminal banner: `cargo run --release` builds a little program that draws a
moonlit fjord in braille, and a longship rows across it.

A shell types the command, cargo compiles, the program takes the alternate
screen and renders the scene on the braille dot lattice (2 x 4 dots per
terminal cell): layered fjord walls dithered with an ordered (Bayer) matrix and
outlined along their skylines, a moon with a halo, its glitter path on the
water, mirrored cliffs broken into ripple rows, and a longship (clinker hull,
spiral stems, striped square sail, shields, oars) rowing east with a Kelvin
wake.  Then the program exits, the shell comes back and `clear` resets it.

Cost tricks (as in the ascii-topo design): every dot is a vertex of a path
whose marker stamps it, so a dot costs ~2-3 bytes; the scenery is stored once;
the ship is stored once and stepped one dot east per frame with a discrete
translate (the dot lattice is uniform, so this is pixel-identical to
re-rendering), carrying its own knock-out mask; only the oars, the wake and
the glitter have a handful of looping frames.

All text is Go Mono outlines.

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
MONO_B = "/usr/share/fonts/fonts-go/Go-Mono-Bold.ttf"
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
GX, GY = 2 * COLS, 4 * ROWS      # scene size in dots (256 x 72)
HZ = 40                          # first water dot row (horizon)
WL = 60                          # ship's water line (dot row)

# ---------------------------------------------------------------- fonts
_fonts = {}


def font(path):
    if path not in _fonts:
        f = TTFont(path)
        _fonts[path] = (f.getGlyphSet(), f.getBestCmap(), f["head"].unitsPerEm)
    return _fonts[path]


def fmt(v):
    s = f"{v:.2f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def glyph_d(ch, path=MONO, px=FONT_PX):
    gs, cmap, upem = font(path)
    pen = SVGPathPen(gs, fmt)
    s = px / upem
    gs[cmap[ord(ch)]].draw(TransformPen(pen, (s, 0, 0, -s, 0, 0)))
    return pen.getCommands()


def sans_text_d(text, px, x, y):
    import uharfbuzz as hb
    blob = hb.Blob.from_file_path(SANS)
    hfont = hb.Font(hb.Face(blob))
    buf = hb.Buffer()
    buf.add_str(text)
    buf.guess_segment_properties()
    hb.shape(hfont, buf, {"kern": True, "liga": True})
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


# ---------------------------------------------------------------- raster helpers
BAYER = np.array([[0, 32, 8, 40, 2, 34, 10, 42],
                  [48, 16, 56, 24, 50, 18, 58, 26],
                  [12, 44, 4, 36, 14, 46, 6, 38],
                  [60, 28, 52, 20, 62, 30, 54, 22],
                  [3, 35, 11, 43, 1, 33, 9, 41],
                  [51, 19, 59, 27, 49, 17, 57, 25],
                  [15, 47, 7, 39, 13, 45, 5, 37],
                  [63, 31, 55, 23, 61, 29, 53, 21]]) / 64 + 1 / 128


def bayer(shape, ox=0, oy=0):
    ys = (np.arange(shape[0]) + oy) % 8
    xs = (np.arange(shape[1]) + ox) % 8
    return BAYER[ys[:, None], xs[None, :]]


def value_noise(x, scale, seed):
    r = np.random.default_rng(seed)
    n = int(np.max(x) / scale) + 4
    lat = r.uniform(-1, 1, n)
    u = np.asarray(x, float) / scale
    i = np.floor(u).astype(int)
    t = u - i
    t = t * t * t * (t * (t * 6 - 15) + 10)
    return lat[i] * (1 - t) + lat[i + 1] * t


def fbm(x, scale, seed, octaves=6, gain=0.5):
    out = np.zeros_like(np.asarray(x, float))
    amp = tot = 0.0
    amp = 1.0
    for o in range(octaves):
        out += amp * value_noise(x, scale / 2 ** o, seed * 31 + o)
        tot += amp
        amp *= gain
    return out / tot


def noise2(X, Y, scale, seed):
    """Cheap 2-D value noise on a lattice (bilinear, smoothstep)."""
    r = np.random.default_rng(seed)
    lat = r.uniform(-1, 1, (64, 64))
    u, v = X / scale + 7, Y / scale + 7
    i, j = np.floor(u).astype(int), np.floor(v).astype(int)
    fu, fv = u - i, v - j
    fu = fu * fu * (3 - 2 * fu)
    fv = fv * fv * (3 - 2 * fv)
    i %= 63
    j %= 63
    a, b = lat[j, i], lat[j, i + 1]
    c, d = lat[j + 1, i], lat[j + 1, i + 1]
    return (a * (1 - fu) + b * fu) * (1 - fv) + (c * (1 - fu) + d * fu) * fv


def poly_inside(poly, X, Y):
    """Even-odd point-in-polygon for arrays X, Y."""
    inside = np.zeros(X.shape, bool)
    px, py = poly[:, 0], poly[:, 1]
    n = len(poly)
    for k in range(n):
        x1, y1 = px[k], py[k]
        x2, y2 = px[(k + 1) % n], py[(k + 1) % n]
        if y1 == y2:
            continue
        cond = (y1 > Y) != (y2 > Y)
        xi = x1 + (Y - y1) * (x2 - x1) / (y2 - y1)
        inside ^= cond & (X < xi)
    return inside


def coverage(inside_fn, xs, ys, ss=4):
    """Fraction of each dot cell (centred on integer coords) inside the shape."""
    o = (np.arange(ss) + 0.5) / ss - 0.5
    X = xs[None, :, None, None] + o[None, None, None, :]
    Y = ys[:, None, None, None] + o[None, None, :, None]
    X, Y = np.broadcast_arrays(X, Y)
    return inside_fn(X, Y).mean(axis=(2, 3))


def dilate(m, n=1):
    for _ in range(n):
        p = np.pad(m, 1)
        m = (p[1:-1, 1:-1] | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]
             | p[:-2, :-2] | p[:-2, 2:] | p[2:, :-2] | p[2:, 2:])
    return m


def erode4(m):
    p = np.pad(m, 1, constant_values=True)
    return m & p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]


# ---------------------------------------------------------------- scene
# Fjord walls: glacial U-profiles (steep faces, fjell plateaus) alternating
# left and right, receding toward the moon.  Designed in a 1200-px frame.
PXS = 1200 / GX                  # design px per dot, horizontally
VYS = 0.19                       # dots per design px, vertically
YH_PX = 226.0

LAYERS = [
    dict(kind="range", top=170, seed=3),
    dict(side="R", c=716, w=30, top=140, rough=8, seed=11),
    dict(side="L", c=556, w=30, top=110, rough=10, seed=5),
    dict(side="R", c=880, w=30, top=80, rough=12, seed=21),
    dict(side="L", c=360, w=28, top=56, rough=13, seed=8),
    dict(side="R", c=1070, w=26, top=40, rough=14, seed=33, trend=0.1),
    dict(side="L", c=140, w=26, top=22, rough=14, seed=41, trend=0.08),
]


def cliff(x, c, w, side):
    s = (x - c) / w * (1 if side == "L" else -1)
    s = np.clip(s, -1.6, 1.6)
    return 0.5 * (1 - np.tanh(2.2 * s))


def layer_top(L, x):
    """Skyline height in design px (y down) for design-px x."""
    if L.get("kind") == "range":
        n = fbm(x, 120, L["seed"], 7, 0.55)
        hump = np.exp(-((x - 700) / 230) ** 2)
        return YH_PX - (YH_PX - L["top"]) * (0.45 + 0.55 * hump) * (0.7 + 0.3 * n)
    sd = L["seed"]
    xw = x + 1.4 * L["w"] * fbm(x, 110, sd + 7, 4)
    env = cliff(xw, L["c"], L["w"], L["side"])
    plate = L["top"] + L["rough"] * fbm(x, 190, sd, 7, 0.55)
    r = np.random.default_rng(sd)
    inland = np.clip((L["c"] - x) if L["side"] == "L" else (x - L["c"]), 0, 500)
    plate -= L.get("trend", 0.05) * inland - 3
    for _ in range(4):
        px = L["c"] + (-1 if L["side"] == "L" else 1) * r.uniform(1.5, 11) * L["w"]
        plate -= r.uniform(4, 14) * np.exp(-((x - px) / r.uniform(14, 38)) ** 2)
    nx = L["c"] + (-1 if L["side"] == "L" else 1) * r.uniform(3, 7) * L["w"]
    plate += r.uniform(4, 9) * np.exp(-((x - nx) / r.uniform(10, 18)) ** 2)
    y = YH_PX - (YH_PX - plate) * env
    face = 4 * env * (1 - env)
    y += 7 * face * fbm(x, 22, sd + 99, 4)
    return np.minimum(y, YH_PX)


MOON = (142.0, 15.5, 5.6)        # centre x, y and radius, in dots
GLITTER_FRAMES = 6
GLITTER_STEP = 0.3               # s per glitter frame


def build_scene():
    """Returns {colour key: bool array (GY, GX)} for the static scenery and the
    list of glitter frames (same format)."""
    gy, gx = np.mgrid[0:GY, 0:GX].astype(float)
    xpx = (np.arange(GX) + 0.5) * PXS
    B = bayer((GY, GX))
    sky = gy < HZ
    layers = {}
    occupied = np.zeros((GY, GX), bool)

    # stars
    rng = np.random.default_rng(12)
    stars = np.zeros((GY, GX), bool)
    stars_b = np.zeros((GY, GX), bool)
    for _ in range(70):
        x, y = rng.integers(0, GX), rng.integers(0, HZ - 6)
        if math.hypot(x - MOON[0], (y - MOON[1]) * 1.0) < 22:
            continue
        (stars_b if rng.random() < 0.18 else stars)[y, x] = True

    # moon, halo
    mx, my, mr = MOON
    rr = np.hypot(gx - mx, gy - my)
    disc = coverage(lambda X, Y: np.hypot(X - mx, Y - my) < mr, np.arange(GX), np.arange(GY)) > 0.45
    mare = disc & (noise2(gx, gy, 2.6, 4) > 0.25) & (rr < mr - 1.2)
    halo_i = 0.32 * np.exp(-(rr - mr) / 2.6) + 0.08 * np.exp(-(rr - mr) / 9)
    halo = (~disc) & (halo_i > B) & (rr < mr + 18)

    # walls, far to near
    tops = []
    for k, L in enumerate(LAYERS):
        ypx = layer_top(L, xpx)
        top = HZ - (YH_PX - ypx) * VYS
        tops.append(top)
    masks = [(gy >= t[None, :] - 0.5) & (gy < HZ) for t in tops]
    nl = len(LAYERS)
    walls = {}
    for k in range(nl):
        m = masks[k].copy()
        for j in range(k + 1, nl):          # occluded by nearer walls
            m &= ~masks[j]
        nearer = np.zeros_like(m)
        for j in range(k + 1, nl):
            nearer |= masks[j]
        m &= ~(dilate(nearer) & ~nearer)    # 1-dot gap in front of nearer walls
        full = masks[k]
        rim = full & ~erode4(full | ~sky)   # outline (not along the horizon)
        rim &= m
        depth = k / (nl - 1)                # 0 far .. 1 near
        # density: far walls misty and bright, near walls darker; mist pools
        # toward the water; vertical gullies give the faces texture
        dens = 0.5 - 0.34 * depth
        dens = dens + (0.3 - 0.16 * depth) * np.exp(-(HZ - gy) / 5.0)
        gul = noise2(gx * 1.0, gy * 0.22, 3.0, 20 + k)
        dens = dens * (0.75 + 0.45 * gul)
        # a lighter cap below the skyline (moonlit plateau edge)
        below = gy - tops[k][None, :]
        dens = dens + 0.25 * np.exp(-below / 2.0) * (below > 0)
        body = m & ~rim & (dens > B)
        walls[k] = (body, rim, m)
        occupied |= m
    stars &= ~occupied
    stars_b &= ~occupied
    halo &= ~occupied
    disc_v = disc & ~occupied

    out = {"star": stars, "star_b": stars_b, "halo": halo,
           "moon": disc_v & ~mare, "mare": disc_v & mare}
    for k in range(nl):
        out[f"w{k}"] = walls[k][0]
        out[f"r{k}"] = walls[k][1]

    # ---- water -------------------------------------------------------
    water = gy >= HZ
    # shoreline: a bright line where walls meet the water
    shore = np.zeros((GY, GX), bool)
    shore[HZ, :] = occupied[HZ - 1, :]
    # reflections of the walls, mirrored about the horizon, in ripple rows
    refl = {k: np.zeros((GY, GX), bool) for k in range(nl)}
    frim = np.zeros((GY, GX), bool)
    rng = np.random.default_rng(31)
    for y in range(HZ + 1, GY):
        d = y - HZ
        if d % 2 == 1 and d > 1:
            continue
        ys = HZ - 1 - d                       # mirrored source row
        if ys < 0:
            continue
        shift = int(round(rng.normal(0, 0.35 + d * 0.035)))
        for k in range(nl):
            depth = k / (nl - 1)
            dens = (0.3 - 0.16 * depth) * (1 - d / 40) + 0.15 * math.exp(-d / 2.5)
            row = np.roll(walls[k][2][ys] & ~walls[k][1][ys], shift) & (dens > B[y])
            refl[k][y] = row
            frim[y] |= np.roll(walls[k][1][ys], shift) & (rng.random(GX) < 0.75 - d * 0.012)
    for k in range(nl):
        out[f"f{k}"] = refl[k] & ~frim
    out["frim"] = frim
    out["shore"] = shore & ~np.zeros_like(shore)
    # faint ripple dashes across the open water
    rip = np.zeros((GY, GX), bool)
    rng = np.random.default_rng(8)
    for y in range(HZ + 2, GY):
        d = (y - HZ) / (GY - HZ)
        n = int(1 + 4 * d)
        for _ in range(n):
            x = rng.integers(0, GX)
            ln = int(rng.integers(2, 4 + int(6 * d)))
            rip[y, x:x + ln] = True
    any_refl = np.zeros((GY, GX), bool)
    for k in range(nl):
        any_refl |= refl[k] | frim
    out["ripple"] = rip & ~dilate(any_refl) & water

    # ---- glitter path under the moon (frames) -------------------------
    frames = []
    rng = np.random.default_rng(5)
    for f in range(GLITTER_FRAMES):
        hi = np.zeros((GY, GX), bool)
        lo = np.zeros((GY, GX), bool)
        for y in range(HZ + 1, GY):
            d = y - HZ
            half = 1.2 + d * 0.2
            if rng.random() > 0.2:
                # a streak centred under the moon, its length shimmering
                c = mx + rng.normal(0, 0.5 + d * 0.025)
                L = half * rng.uniform(0.3, 1.0)
                for x in range(int(round(c - L)), int(round(c + L)) + 1):
                    (hi if abs(x - c) <= max(0.6, 0.5 * L) else lo)[y, x] = True
            for _ in range(rng.poisson(0.25 + d * 0.035)):
                x = int(round(mx + rng.normal(0, 0.6 * half + 1.5)))
                lo[y, x:x + int(rng.integers(1, 3))] = True
        lo &= ~hi
        frames.append({"glint": hi, "glint_lo": lo})
    # keep the reflections and ripples out of the glitter column
    col = np.zeros((GY, GX), bool)
    for fr in frames:
        col |= fr["glint"] | fr["glint_lo"]
    for k in out:
        if k[0] == "f" or k == "ripple":
            out[k] = out[k] & ~col
    return out, frames


# ---------------------------------------------------------------- ship
SX0, SX1, SY0, SY1 = -46, 46, -46, 14      # sprite box, dots relative to (x, WL)


def bez(p0, p1, p2, p3, n):
    t = np.linspace(0, 1, n)[:, None]
    p0, p1, p2, p3 = map(np.array, (p0, p1, p2, p3))
    return ((1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1
            + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3)


def stem_line(sign, foot, top, r0, turns):
    """Raked stem out of the water ending in an inward spiral (sign +1: bow)."""
    cx, cy = top
    S = (cx + sign * r0, cy)
    rise = bez(foot, (foot[0] + sign * 6.5, foot[1] - 1.0),
               (S[0] + sign * 0.8, S[1] + 6), S, 50)
    th = np.linspace(0, 2 * math.pi * turns, 80)
    r = r0 * (1 - 0.72 * th / th[-1]) ** 1.1
    sp = np.stack([cx + sign * r * np.cos(th), cy - r * np.sin(th)], 1)
    return np.vstack([rise, sp[1:]])


def thick(cl, w0, w1):
    d = np.gradient(cl, axis=0)
    d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
    nrm = np.stack([-d[:, 1], d[:, 0]], 1)
    s = np.linspace(0, 1, len(cl))
    w = w0 + (w1 - w0) * s
    return np.vstack([cl + nrm * (w / 2)[:, None], (cl - nrm * (w / 2)[:, None])[::-1]])


def seg_inside(a, b, w):
    a, b = np.array(a, float), np.array(b, float)

    def f(X, Y):
        d = b - a
        t = np.clip(((X - a[0]) * d[0] + (Y - a[1]) * d[1]) / (d @ d), 0, 1)
        return np.hypot(X - a[0] - t * d[0], Y - a[1] - t * d[1]) < w / 2
    return f


MAST_X = 1.0
MAST_TOP = -36.0
YARD_Y = -32.0
SAIL_L, SAIL_R = -12.0, 14.0
SAIL_BOT = -9.5
N_OARS = 8
OAR_X = np.linspace(-13, 15, N_OARS)
OAR_R = 9.5
STROKE = 2.0
OAR_FRAMES = 8
WAKE_FRAMES = 4
WAKE_STEP = 0.35


def sheer(x):
    return -3.6 - 5.4 * np.clip(np.abs(x - 0.5) / 21.0, 0, 1.2) ** 3.6


def build_ship():
    xs = np.arange(SX0, SX1 + 1)
    ys = np.arange(SY0, SY1 + 1)
    gx, gy = np.meshgrid(xs.astype(float), ys.astype(float))
    bow = stem_line(+1, (16.0, 0.6), (24.5, -15.0), 3.0, 1.15)
    stern = stem_line(-1, (-15.0, 0.6), (-23.5, -14.0), 2.8, 1.1)

    def at(cl, h):
        i = int(np.argmin(np.abs(cl[:50, 1] - h)))
        return i
    ib = at(bow, sheer(np.array(21.5)))
    ia = at(stern, sheer(np.array(-20.5)))
    sx = np.linspace(stern[ia][0], bow[ib][0], 120)
    hull_poly = np.vstack([[[-15.0, 0.6], [16.0, 0.6]], bow[1:ib + 1],
                           np.stack([sx, sheer(sx)], 1)[::-1][1:-1],
                           stern[:ia + 1][::-1]])
    cov_hull = coverage(lambda X, Y: poly_inside(hull_poly, X, Y), xs, ys)
    bow_p = thick(bow, 2.3, 1.25)
    stern_p = thick(stern, 2.2, 1.2)
    cov_stem = np.maximum(coverage(lambda X, Y: poly_inside(bow_p, X, Y), xs, ys),
                          coverage(lambda X, Y: poly_inside(stern_p, X, Y), xs, ys))
    hull = (cov_hull > 0.42) & (gy <= 0)
    stems = (cov_stem > 0.38) & ~hull
    hull_all = hull | stems
    # clinker strakes follow the sheer: banded by fraction of freeboard
    sh = sheer(gx)
    frac = (gy - sh) / (0.6 - sh)
    rail = hull & (gy - sh < 0.9)
    strake = hull & ~rail & ((np.floor(frac * 4.0) % 2) == 1)
    plank = hull & ~rail & ~strake
    # sail: square, braced, bellied slightly to leeward (east)
    def sail_in(X, Y):
        v = (Y - YARD_Y) / (SAIL_BOT - YARD_Y)             # 0 at yard, 1 at foot
        belly = 1.4 * np.sin(np.pi * np.clip(v, 0, 1))
        left = SAIL_L + 0.6 * v
        right = SAIL_R + belly - 0.4 * v
        foot = SAIL_BOT - 1.2 * np.sin(np.pi * np.clip((X - SAIL_L) / (SAIL_R - SAIL_L), 0, 1))
        return (Y > YARD_Y + 0.3) & (Y < foot) & (X > left) & (X < right)
    sail = coverage(sail_in, xs, ys) > 0.5
    stripe_w = (SAIL_R - SAIL_L) / 7
    band = np.floor((gx - SAIL_L + 0.1 * (gy - YARD_Y)) / stripe_w).astype(int)
    v = (gy - YARD_Y) / (SAIL_BOT - YARD_Y)
    red = sail & (band % 2 == 1)
    cream = sail & ~red
    shade = v > 0.62                                     # lower third a shade darker
    yard = coverage(seg_inside((SAIL_L - 3, YARD_Y), (SAIL_R + 3, YARD_Y), 1.0), xs, ys) > 0.4
    mast = coverage(seg_inside((MAST_X, sheer(np.array(MAST_X)) + 0.5), (MAST_X, MAST_TOP), 0.9),
                    xs, ys) > 0.4
    mast &= ~sail
    yard &= ~sail
    # stays to the stem heads, sheets from the clews
    rig = np.zeros_like(sail)
    for a, b in (((MAST_X, MAST_TOP + 1), (22.5, -17.0)),
                 ((MAST_X, MAST_TOP + 1), (-21.5, -16.0)),
                 ((SAIL_R, SAIL_BOT - 0.5), (19, sheer(np.array(19.0)) - 0.5)),
                 ((SAIL_L, SAIL_BOT - 0.5), (-17, sheer(np.array(-17.0)) - 0.5))):
        rig |= coverage(seg_inside(a, b, 0.55), xs, ys, ss=6) > 0.2
    # shields on the rail
    shields_a = np.zeros_like(sail)
    shields_b = np.zeros_like(sail)
    for i, x in enumerate(np.arange(-15, 17, 3.0)):
        y = float(sheer(np.array(x))) + 1.0
        c = coverage(lambda X, Y: np.hypot(X - x, (Y - y) * 1.0) < 1.15, xs, ys) > 0.5
        (shields_a if i % 2 == 0 else shields_b)[:] |= c
    sail_any = sail | yard | mast
    rig &= ~(sail_any | hull_all | shields_a | shields_b)
    rail &= ~(shields_a | shields_b)
    plank &= ~(shields_a | shields_b)
    strake &= ~(shields_a | shields_b)
    hull_all_ = hull_all | shields_a | shields_b
    parts = {
        "plank": plank, "strake": strake, "rail": rail | stems,
        "shield_a": shields_a, "shield_b": shields_b,
        "mast": mast | yard, "rig": rig,
        "sail_r": red & ~shade, "sail_r2": red & shade,
        "sail_c": cream & ~shade, "sail_c2": cream & shade,
    }
    body = hull_all_ | sail_any
    # reflection: hull, stems and sail mirrored in ripple rows
    refl_h = np.zeros_like(sail)
    refl_s = np.zeros_like(sail)
    rng = np.random.default_rng(2)
    iy0 = -SY0                                      # index of y = 0
    for j, y in enumerate(ys):
        if y < 1:
            continue
        d = int(y)
        if d % 2 == 0 and d > 2:
            continue
        src = iy0 - d
        sh_ = int(round(rng.normal(0, 0.3 + 0.05 * d)))
        keep = rng.random(len(xs)) < 0.8 - 0.035 * d
        refl_h[j] = np.roll(hull_all_[src], sh_) & keep
        refl_s[j] = np.roll(sail[src] | mast[src], sh_) & keep
    refl_s &= ~refl_h
    parts["refl_h"] = refl_h
    parts["refl_s"] = refl_s
    # knock-out mask: the silhouette plus a 1-dot gap, and the reflection band
    above = body.copy()
    mask = dilate(above) | dilate(refl_h | refl_s)
    # oars
    oar_frames = []
    drive = 0.45
    a0, a1 = 32.0, -26.0
    for f in range(OAR_FRAMES):
        p = f / OAR_FRAMES
        if p <= drive:
            s = 0.5 - 0.5 * math.cos(math.pi * p / drive)
            th = math.radians(a0 + (a1 - a0) * s)
            lift = 0.0
        else:
            s = (p - drive) / (1 - drive)
            e = 0.5 - 0.5 * math.cos(math.pi * s)
            th = math.radians(a1 + (a0 - a1) * e)
            lift = 2.6 * math.sin(math.pi * s) ** 0.7
        loom = np.zeros_like(sail)
        blade = np.zeros_like(sail)
        for x in OAR_X:
            py = float(sheer(np.array(x))) + 1.6
            dx = OAR_R * math.sin(th)
            dy = 1.6 + 0.62 * OAR_R * math.cos(th) - lift
            tip = (x + dx, py + dy)
            mid = (x + 0.7 * dx, py + 0.7 * dy)
            loom |= coverage(seg_inside((x, py), mid, 0.5), xs, ys, ss=6) > 0.18
            blade |= coverage(seg_inside(mid, tip, 1.3), xs, ys, ss=6) > 0.3
        loom &= ~blade
        # a splash ring where each blade sits in the water during the drive
        splash = np.zeros_like(sail)
        if p <= drive and p > 0:
            for x in OAR_X:
                py = float(sheer(np.array(x))) + 1.6
                tx = x + OAR_R * math.sin(th)
                ty = py + 1.6 + 0.62 * OAR_R * math.cos(th)
                splash |= coverage(lambda X, Y: np.abs(np.hypot((X - tx) / 2.2, Y - ty - 0.6) - 1) < 0.25,
                                   xs, ys) > 0.2
            splash &= ~(blade | loom | hull_all_)
            splash &= ~mask
        oar_frames.append({"oar": loom, "blade": blade, "splash": splash & (gy > 0)})
    # wake: Kelvin arms from the stern, a churned centre trail, a bow wave
    wake_frames = []
    for f in range(WAKE_FRAMES):
        rng = np.random.default_rng(100 + f)
        hi = np.zeros_like(sail)
        lo = np.zeros_like(sail)
        for arm, slope in ((0, -0.13), (1, 0.21)):
            for d in np.arange(0, 26, 0.5):
                x = -16 - d
                y = 1.0 + slope * d
                fade = 1 - d / 26
                if rng.random() < 0.6 * fade + 0.15:
                    ln = rng.integers(1, 4)
                    yy = int(round(y + rng.normal(0, 0.25)))
                    xx = int(round(x))
                    tgt = hi if rng.random() < 0.45 * fade else lo
                    ii, jj = yy - SY0, xx - SX0
                    if 0 <= ii < tgt.shape[0] and 0 <= jj < tgt.shape[1]:
                        tgt[ii, max(0, jj - ln + 1):jj + 1] = True
        for d in np.arange(0, 24, 0.7):
            x = -15 - d
            y = 1.0 + 0.04 * d + rng.normal(0, 0.4)
            if rng.random() < 0.5 * (1 - d / 24):
                ii, jj = int(round(y)) - SY0, int(round(x)) - SX0
                lo[ii, jj] = True
        for d in range(0, 4):               # bow wave
            for s_ in (-1, 1):
                if rng.random() < 0.7:
                    ii = 1 - SY0 + (1 if s_ > 0 else 0)
                    jj = 18 + d - SX0
                    hi[ii, jj] = True
        lo &= ~hi
        wake_frames.append({"wake": hi & ~mask, "wake_lo": lo & ~mask})
    return parts, mask, oar_frames, wake_frames


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


def mix(a, b, f):
    return from_oklab(to_oklab(a) * (1 - f) + to_oklab(b) * f)


def ramp(a, b, n):
    return [mix(a, b, i / (n - 1)) for i in range(n)]


NL = len(LAYERS)
THEMES = {
    "dark": dict(
        page="#0d1117", card="#0b0f15", bar="#11161d", border="#262c36",
        hair="#1b2129", title="#7d8590", fg="#c9d1d9", dim="#6e7681",
        dir="#79c0ff", chev="#c297ff", cursor="#d6dbe1", ok="#3fb950",
        lights=("#ff5f57", "#febc2e", "#28c840"),
        walls=ramp("#6f829c", "#34445a", NL), rims=ramp("#a9b8cb", "#7d8ea6", NL),
        refl=ramp("#46546a", "#283345", NL), frim="#61728a",
        star="#56637a", star_b="#aab6c8", halo="#8f8a78", moon="#f1e9d4", mare="#c9c0a6",
        shore="#8193aa", ripple="#2a3545", glint="#f4e6c2", glint_lo="#a99a78",
        plank="#8c7765", strake="#6f5d4f", rail="#c7ad8d", shield_a="#d8b56a",
        shield_b="#a8553f", mast="#a58e74", rig="#5d6676",
        sail_r="#c0603f", sail_r2="#9c4c34", sail_c="#dccdac", sail_c2="#b9ab8c",
        refl_h="#3e3832", refl_s="#5a3a30", oar="#b39b7d", blade="#d9c4a2",
        splash="#9fb0c4", wake="#c5d0dd", wake_lo="#6c7b8f",
    ),
    "light": dict(
        page="#ffffff", card="#fbfbfc", bar="#f3f4f6", border="#d8dee4",
        hair="#e6e8ec", title="#6e7781", fg="#24292f", dim="#8c959f",
        dir="#0969da", chev="#8250df", cursor="#24292f", ok="#1a7f37",
        lights=("#ff5f57", "#febc2e", "#28c840"),
        walls=ramp("#a9b5c4", "#4f5f74", NL), rims=ramp("#7d8b9e", "#2f3b4b", NL),
        refl=ramp("#cdd4dd", "#a3afbe", NL), frim="#8e9bad",
        star="#c9d0d9", star_b="#9aa5b3", halo="#e8cf9f", moon="#d9a24e", mare="#bf883b",
        shore="#6f7e92", ripple="#dde2e8", glint="#d29a45", glint_lo="#ebd3a8",
        plank="#5d4a3b", strake="#7a6452", rail="#3a2d23", shield_a="#c9962e",
        shield_b="#a8432c", mast="#4a3b2f", rig="#a3abb6",
        sail_r="#b4472c", sail_r2="#953a24", sail_c="#cfb487", sail_c2="#b89b6b",
        refl_h="#d6d0c8", refl_s="#e2c5b6", oar="#6b5747", blade="#4a3b2f",
        splash="#8e9bad", wake="#7e8ca0", wake_lo="#bcc5d1",
    ),
}


def colour_of(th, key):
    if key[0] in "wrf" and key[1:].isdigit():
        k = int(key[1:])
        return {"w": th["walls"], "r": th["rims"], "f": th["refl"]}[key[0]][k]
    return th[key]


DOT_R = {"star": 0.22, "star_b": 0.28, "halo": 0.26, "ripple": 0.26, "rig": 0.24,
         "f": 0.27, "wake_lo": 0.27, "splash": 0.27}


def dot_r(key):
    if key in DOT_R:
        return DOT_R[key]
    if key[0] == "f" and key[1:].isdigit():
        return DOT_R["f"]
    return 0.33


# ---------------------------------------------------------------- timeline
T = 24.0


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


PROMPT_DIR = "~/src/fjord"
PROMPT_COLS = len(PROMPT_DIR) + 3
CMD = "cargo run --release"
CARGO = [("Compiling", "fjord v0.1.0 (~/src/fjord)"),
         ("Finished", "`release` profile [optimized] target(s) in 0.81s"),
         ("Running", "`target/release/fjord`")]
_rng = random.Random(11)
T_TYPE0 = 0.6
TTIMES = typing_times(CMD, T_TYPE0, _rng)
T_ENTER = TTIMES[-1] + 0.4
T_COMP = T_ENTER + 0.1
T_FIN = T_COMP + 0.84
T_RUN = T_FIN + 0.06
T_ALT = T_RUN + 0.3
T_EXIT = 21.2
T_PROMPT2 = T_EXIT + 0.05
CTIMES = typing_times("clear", T_PROMPT2 + 0.7, _rng)
T_ENTER2 = CTIMES[-1] + 0.38
T_CLEAR = T_ENTER2 + 0.06

SHIP_FPS = 8                     # one dot east per frame
SHIP_X0 = 50
N_STEPS = int((T_EXIT - T_ALT) * SHIP_FPS)


# ---------------------------------------------------------------- svg helpers
def chain_path(pts):
    """Relative path through lattice points (each vertex stamps a dot)."""
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


def runs_path(mask, ox, oy):
    """Mask as horizontal runs of unit squares (dot cells)."""
    out = []
    for j in range(mask.shape[0]):
        row = mask[j]
        i = 0
        while i < len(row):
            if row[i]:
                k = i
                while k < len(row) and row[k]:
                    k += 1
                out.append(f"M{fmt(i + ox - 0.5)} {fmt(j + oy - 0.5)}h{k - i}v1h{i - k}z")
                i = k
            else:
                i += 1
    return "".join(out)


def pts_of(m, ox=0, oy=0):
    ys, xs = np.nonzero(m)
    return [(int(x) + ox, int(y) + oy) for x, y in zip(xs, ys)]


def frames_group(frames, period, step, order):
    """Looping frame groups (visibility), `step` s per frame."""
    n = len(frames)
    out = []
    for i, fr in enumerate(frames):
        paths = "".join(f'<path class="{k}" d="{chain_path(pts)}"/>'
                        for k in order if (pts := fr.get(k)))
        if i == 0:
            vals, keys = "visible;hidden", f"0;{1 / n:.5f}"
        else:
            vals, keys = "hidden;visible;hidden", f"0;{i / n:.5f};{(i + 1) / n:.5f}"
        out.append(f'<g visibility="{"visible" if i == 0 else "hidden"}">{paths}'
                   f'<animate attributeName="visibility" dur="{fmt(n * step)}s" '
                   f'repeatCount="indefinite" calcMode="discrete" values="{vals}" '
                   f'keyTimes="{keys}"/></g>')
    return "".join(out)


# ---------------------------------------------------------------- build
def build(theme_name, scene, glitter, ship):
    th = THEMES[theme_name]
    parts, defs = [], []
    used = set()
    used_b = set()
    keys_used = set()

    def g(ch, bold=False):
        (used_b if bold else used).add(ch)
        return f"#{'b' if bold else 'c'}{ord(ch):x}"

    def text_uses(s, col, row, color=None, bold=False):
        out = []
        for i, ch in enumerate(s):
            if ch == " ":
                continue
            out.append(f'<use href="{g(ch, bold)}" x="{X0 + (col + i) * CW}" '
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
        (f"{PROMPT_DIR} — fjord — {size}",
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
    for i, (verb, rest) in enumerate(CARGO):
        t_on = (T_COMP, T_FIN, T_RUN)[i]
        normal.append(f'<g opacity="0">'
                      f'{text_uses(verb, 12 - len(verb), 1 + i, th["ok"], bold=True)}'
                      f'{text_uses(rest, 13, 1 + i, th["fg"])}'
                      f'{discrete("opacity", [(0, 0), (t_on, 1), (T_CLEAR, 0)])}</g>')
    normal.append(f'<g opacity="0">{prompt(4)}'
                  f'{discrete("opacity", [(0, 0), (T_PROMPT2, 1), (T_CLEAR, 0)])}</g>')
    clip_typed("k1", 4, CTIMES, PROMPT_COLS, T_CLEAR)
    normal.append(f'<g clip-path="url(#k1)">{text_uses("clear", PROMPT_COLS, 4, th["fg"])}</g>')
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
    cy_ev.append((T_COMP, Y0 + 2 * LH))
    cy_ev.append((T_FIN, Y0 + 3 * LH))
    cy_ev.append((T_RUN, Y0 + 4 * LH))
    op_ev.append((T_ALT, 0))
    cx_ev.append((T_PROMPT2, base_x)); cy_ev.append((T_PROMPT2, Y0 + 4 * LH))
    op_ev += blink(T_PROMPT2, CTIMES[0] - 0.05)
    op_ev.append((round(CTIMES[0] - 0.05, 3), 1))
    for i, t in enumerate(CTIMES):
        cx_ev.append((t, base_x + (i + 1) * CW))
    cx_ev.append((T_ENTER2, X0)); cy_ev.append((T_ENTER2, Y0 + 5 * LH))
    cx_ev.append((T_CLEAR, base_x)); cy_ev.append((T_CLEAR, Y0))
    op_ev += blink(T_CLEAR, T)
    parts.append(f'<rect x="{base_x}" y="{Y0}" width="{CW}" height="{LH}" '
                 f'fill="{th["cursor"]}" opacity=".85">'
                 f'{discrete("x", cx_ev)}{discrete("y", cy_ev)}'
                 f'{discrete("opacity", [(t, .85 if o else 0) for t, o in op_ev])}</rect>')

    # ---- the scene (dot units) ----------------------------------------------
    def paths(layer, order=None):
        out = []
        for k in (order or layer):
            m = layer.get(k)
            if m is None or not m.any():
                continue
            keys_used.add(k)
            out.append(f'<path class="{k}" d="{chain_path(pts_of(m))}"/>')
        return "".join(out)

    static_order = (["star", "star_b", "halo", "moon", "mare"]
                    + [f"w{k}" for k in range(NL)] + [f"r{k}" for k in range(NL)]
                    + ["shore"] + [f"f{k}" for k in range(NL)] + ["frim", "ripple"])
    scene_svg = paths(scene, static_order)
    gl = [{k: pts_of(v) for k, v in fr.items()} for fr in glitter]
    keys_used.update(["glint", "glint_lo"])
    glit_svg = frames_group(gl, GLITTER_FRAMES * GLITTER_STEP, GLITTER_STEP,
                            ["glint_lo", "glint"])

    parts_s, mask, oars, wakes = ship
    ox, oy = SX0, SY0 + WL
    sprite = [f'<path fill="{th["card"]}" d="{runs_path(mask, ox, oy)}"/>']
    s_order = ["refl_h", "refl_s", "rig", "plank", "strake", "rail", "shield_a", "shield_b",
               "mast", "sail_r", "sail_r2", "sail_c", "sail_c2"]
    for k in s_order:
        m = parts_s[k]
        if m.any():
            keys_used.add(k)
            sprite.append(f'<path class="{k}" d="{chain_path(pts_of(m, ox, oy))}"/>')
    wf = [{k: pts_of(v, ox, oy) for k, v in fr.items()} for fr in wakes]
    keys_used.update(["wake", "wake_lo"])
    sprite.append(frames_group(wf, WAKE_FRAMES * WAKE_STEP, WAKE_STEP, ["wake_lo", "wake"]))
    of = [{k: pts_of(v, ox, oy) for k, v in fr.items()} for fr in oars]
    keys_used.update(["oar", "blade", "splash"])
    sprite.append(frames_group(of, STROKE, STROKE / OAR_FRAMES, ["splash", "oar", "blade"]))
    steps = [(0, f"{SHIP_X0} 0")] + [(round(T_ALT + i / SHIP_FPS, 3), f"{SHIP_X0 + i} 0")
                                     for i in range(N_STEPS)]
    tr = (f'<animateTransform attributeName="transform" type="translate" dur="{T}s" '
          f'repeatCount="indefinite" calcMode="discrete" '
          f'values="{";".join(v for _, v in steps)}" '
          f'keyTimes="{";".join(kt(t) for t, _ in steps)}"/>')
    ship_svg = f'<g transform="translate({SHIP_X0} 0)">{"".join(sprite)}{tr}</g>'

    # twinkling stars: a few of the bright ones blink off briefly
    rng = random.Random(4)
    sb = pts_of(scene["star_b"])
    tw = []
    for x, y in rng.sample(sb, min(5, len(sb))):
        t0 = rng.uniform(T_ALT + 1, T_EXIT - 2)
        tw.append(f'<path class="card" d="M{x} {y}h0">'
                  f'{discrete("opacity", [(0, 0), (round(t0, 2), 1), (round(t0 + 0.4, 2), 0)])}'
                  f'</path>')
    keys_used.add("card")

    defs.append(f'<clipPath id="view"><rect x="{X0}" y="{Y0}" width="{COLS * CW}" '
                f'height="{ROWS * LH}"/></clipPath>')
    rows_ev = [(0, 0)]
    for r in range(ROWS + 1):
        rows_ev.append((round(T_ALT + r * 0.02, 3), r * LH))
    rows_ev.append((T_EXIT, 0))
    defs.append(f'<clipPath id="wipe"><rect x="0" y="{Y0}" width="{W}" height="0">'
                f'{discrete("height", rows_ev)}</rect></clipPath>')
    mx0, my0 = X0 + DP / 2, Y0 + DP / 2
    parts.append(f'<g clip-path="url(#wipe)"><g clip-path="url(#view)">'
                 f'<g transform="translate({fmt(mx0)} {fmt(my0)}) scale({DP})">'
                 f'{scene_svg}{glit_svg}{"".join(tw)}{ship_svg}</g></g></g>')

    # dot markers
    style = []
    for k in sorted(keys_used):
        col = th["card"] if k == "card" else colour_of(th, k)
        r = 0.42 if k == "card" else dot_r(k)
        defs.append(f'<marker id="{k}" markerUnits="userSpaceOnUse" markerWidth="1" '
                    f'markerHeight="1" overflow="visible" orient="0">'
                    f'<circle r="{r}" fill="{col}"/></marker>')
        style.append(f".{k}{{marker:url(#{k})}}")

    for ch in sorted(used_b):
        defs.insert(0, f'<path id="b{ord(ch):x}" d="{glyph_d(ch, MONO_B)}"/>')
    for ch in sorted(used):
        defs.insert(0, f'<path id="c{ord(ch):x}" d="{glyph_d(ch)}"/>')

    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
           f'width="{W}" height="{H}" fill="none">'
           f'<style>{"".join(style)}</style>'
           f'<defs>{"".join(defs)}</defs>{"".join(parts)}</svg>')
    return svg


if __name__ == "__main__":
    scene, glitter = build_scene()
    ship = build_ship()
    print(f"alt {T_ALT:.2f}s, {N_STEPS} ship steps, ship x {SHIP_X0}..{SHIP_X0 + N_STEPS - 1}")
    for name in THEMES:
        svg = build(name, scene, glitter, ship)
        p = OUT / f"banner-{name}.svg"
        p.write_text(svg)
        print(p.name, f"{len(svg.encode()) / 1024:.1f} KB")
