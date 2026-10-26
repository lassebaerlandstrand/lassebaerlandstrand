#!/usr/bin/env python3
"""Aurora banner generator.

Builds banner-dark.svg and banner-light.svg: luminous aurora curtains over a still
horizon, mirrored in the water below. All soft shapes are pre-rendered (numpy) into
small, horizontally periodic PNG layers that are embedded as data URIs; the SVG only
animates transforms and opacity (SMIL), so no filter is ever evaluated per frame.

Usage:  python generate.py            (needs only numpy)
"""
import base64
import math
import struct
import zlib
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
W, H = 1200, 340          # viewBox
P = 1200                  # horizontal period of every scrolling layer (viewBox units)
HORIZON = 252             # y of the horizon line
SX, SY = 0.5, 1 / 3       # raster resolution of curtain layers (px per unit)


# --------------------------------------------------------------------------- PNG
def png_bytes(rgba: np.ndarray) -> bytes:
    """Encode uint8 HxWx4 as PNG, choosing the best of the 5 filters per row."""
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
        f = int(np.argmin(scores))
        rows.append(bytes([f]) + cands[f].tobytes())
        prev = cur
    raw = b"".join(rows)
    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 6 if c == 4 else 4, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def data_uri(rgba):
    return "data:image/png;base64," + base64.b64encode(png_bytes(rgba)).decode()


# ------------------------------------------------------------------ noise helpers
def periodic_noise(x, rng, kmin, kmax, falloff=1.0):
    """Sum of sinusoids with integer harmonics of P -> exactly periodic in x. Range ~[-1,1]."""
    out = np.zeros_like(x, dtype=np.float64)
    for k in range(kmin, kmax + 1):
        amp = rng.normal() / (k ** falloff)
        out += amp * np.sin(2 * np.pi * k * x / P + rng.uniform(0, 2 * np.pi))
    return out / (np.abs(out).max() + 1e-9)


def smoothstep(e0, e1, v):
    t = np.clip((v - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def hex_rgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)])


def ramp(stops, t):
    """Piecewise-linear colour ramp. stops = [(t, '#hex'), ...]; t array -> (...,3)."""
    ts = np.array([s[0] for s in stops])
    cs = np.array([hex_rgb(s[1]) for s in stops])
    out = np.empty(t.shape + (3,))
    for ch in range(3):
        out[..., ch] = np.interp(t, ts, cs[:, ch])
    return out


# --------------------------------------------------------------- curtain raster
def curtain(seed, y0, y1, base, amp, height, colors, edge_sigma=7.0, ray_strength=0.7,
            ray_k=(14, 80), gain=1.0, patchiness=1.0, ray_seed=None, ray_reach=1.2, halo=0.0):
    """Render one aurora curtain covering x in [0, 2P) (two periods) and y in [y0, y1].

    base/amp: lower-edge line y_b(x) = base + amp * noise(x)
    height:   decay length of the rays above the edge
    colors:   ramp over normalised height above the edge (0 = edge)
    Returns (uint8 RGBA, (x, y, w, h) placement in viewBox units).
    """
    rng = np.random.default_rng(seed)
    nx, ny = int(2 * P * SX), int(round((y1 - y0) * SY))
    x = (np.arange(nx) + 0.5) / SX
    y = y0 + (np.arange(ny) + 0.5) / SY
    X, Y = np.meshgrid(x, y)

    yb = base + amp * periodic_noise(x, rng, 1, 7, 1.35)
    L = height * (1 + 0.35 * periodic_noise(x, rng, 1, 5, 1.0))
    bright = smoothstep(-0.4, 0.75, periodic_noise(x, rng, 1, 4, 1.0)) ** 1.6
    bright = (1 - patchiness) + patchiness * bright
    rrng = np.random.default_rng(ray_seed if ray_seed is not None else seed + 1000)
    rays = periodic_noise(x, rrng, ray_k[0], ray_k[1], 0.55)
    rays = 0.5 + 0.5 * rays                             # 0..1
    # rays come in clusters: a mid-frequency envelope switches whole groups on/off
    env = smoothstep(-0.5, 0.7, periodic_noise(x, rrng, 4, 14, 0.8))
    r = rays ** 2.4 * (0.25 + 0.75 * env)
    # bright rays reach higher than the diffuse sheet between them
    L = L * (0.7 + ray_reach * r)

    d = yb[None, :] - Y                                 # >0 above the edge
    below = np.exp(-(np.minimum(d, 0) / edge_sigma) ** 2)
    above = np.exp(-np.maximum(d, 0) / L[None, :])
    core = np.exp(-(d / (edge_sigma * 1.6)) ** 2) * 0.35   # hot lower edge
    ray_w = smoothstep(-4, 22, d) * ray_strength
    mod = (1 - ray_w) + ray_w * (0.12 + 1.5 * r[None, :])
    halo_t = halo * np.exp(-np.maximum(-d, 0) / 22) * (d < 0)   # scattered light under the edge
    inten = (below * above * mod + core * below + halo_t) * bright[None, :] * gain
    # soft fade at the very top/bottom of the raster so no hard cut-off shows
    inten *= smoothstep(y0, y0 + 30, Y) * (1 - smoothstep(y1 - 6, y1, Y))

    t = np.clip(d / (L[None, :] * 2.2), 0, 1)
    col = ramp(colors, t)
    a = np.clip(inten, 0, 1)
    rgba = np.concatenate([col, a[..., None]], axis=-1)
    # tiny ordered dither to prevent banding without hurting compression much
    dither = ((np.indices((ny, nx)).sum(0) % 2) - 0.5) * 0.6
    out = np.clip(rgba * 255 + dither[..., None], 0, 255).astype(np.uint8)
    # colour varies slowly and alpha carries the detail: quantise RGB (invisible) and
    # zero it under full transparency, which shrinks the PNG by ~40 %
    out[..., :3] = np.minimum(out[..., :3] // 4 * 4 + 2, 255)
    out[out[..., 3] == 0, :3] = 0
    return out, (0, y0, 2 * P, y1 - y0)


def ripple_layer(seed, color, strength):
    """Static water texture: thin horizontal streaks, denser near the horizon (perspective).
    Drawn in the water colour over the reflection, it breaks it into calm ripples."""
    rng = np.random.default_rng(seed)
    sy, sx = 2, 0.25                                     # px per viewBox unit (dashes are long)
    h, w = (H - HORIZON) * sy, int(W * sx)
    y = (np.arange(h) + 0.5) / sy                         # depth below horizon, units
    x = (np.arange(w) + 0.5) / sx
    # perspective: line index grows like sqrt(depth) -> spacing grows with depth
    u = 3.0 * np.sqrt(y + 2.0)
    phase = (u % 1.0)
    line = np.exp(-((phase - 0.5) / 0.16) ** 2)           # one soft streak per unit of u
    # break streaks into long dashes that differ per line
    k = np.floor(u).astype(int)
    alpha = np.zeros((h, w))
    for row in range(h):
        r2 = np.random.default_rng(seed * 1000 + int(k[row]))
        n = periodic_noise(x, r2, 3, 24, 0.7)
        dash = smoothstep(-0.15, 0.35, n)
        alpha[row] = line[row] * (0.35 + 0.65 * dash)
    depth_fade = smoothstep(0, 10, y) * (0.55 + 0.45 * smoothstep(0, 60, y))
    alpha *= depth_fade[:, None] * strength
    rgb = np.broadcast_to(hex_rgb(color), (h, w, 3))
    rgba = np.concatenate([rgb, alpha[..., None]], axis=-1)
    return np.clip(rgba * 255 + 0.5, 0, 255).astype(np.uint8)


def grain_tile(seed, n=128, dark=True):
    rng = np.random.default_rng(seed)
    v = rng.random((n, n))
    grey = np.full((n, n), 255 if dark else 0, np.uint8)
    alpha = (np.round(np.clip((v - 0.5) * 2, 0, 1) ** 1.5 * 4) * 63).astype(np.uint8)  # 5 levels
    return np.stack([grey, alpha], axis=-1)                # grey + alpha PNG


# --------------------------------------------------------------------- themes
THEMES = {
    "dark": dict(
        bg_top="#04060c", bg_mid="#070b16", bg_hor="#0b1322", water="#05080f",
        border="#ffffff", border_op=0.09,
        a_cols=[(0, "#b4ffe0"), (0.08, "#43f0ad"), (0.35, "#19c49a"), (0.62, "#2a7fd6"),
                (0.85, "#6d4df2"), (1, "#8a3fd9")],
        b_cols=[(0, "#c9f3ff"), (0.1, "#4fd3ff"), (0.45, "#3b7bff"), (0.8, "#7a5cff"),
                (1, "#b04ad6")],
        c_cols=[(0, "#f3c8ff"), (0.2, "#d77bf0"), (0.6, "#8a52ff"), (1, "#5a3bd9")],
        gain_a=1.0, gain_b=0.75, gain_c=0.38,
        glow1="#5b3df5", glow1_op=0.30, glow2="#12b5a0", glow2_op=0.10,
        glow3="#2b6cff", glow3_op=0.18,
        horizon="#bfefff", horizon_op=0.55, refl_op=0.78, stars=True,
        grain_op=0.055, vignette="#000000", vignette_op=0.55, rays=1.0, edge=1.0, hs=1.0, reach=1.0, halo=0.14,
        airglow="#1fa88a", airglow_op=0.10, topfade_op=0.75, ripple="#04070d", ripple_k=0.85,
        bloom="#9fe8ff", bloom_op=0.22, glint="#e6fbff",
    ),
    "light": dict(
        bg_top="#ffffff", bg_mid="#fbfcff", bg_hor="#f1f4fb", water="#f8f9fd",
        border="#0d1117", border_op=0.1,
        a_cols=[(0, "#00b886"), (0.1, "#00b59c"), (0.35, "#0ea5c6"), (0.62, "#3b6ff5"),
                (0.85, "#7457f5"), (1, "#b64ee6")],
        b_cols=[(0, "#1d9bf0"), (0.1, "#1f8ef0"), (0.45, "#3a5ef2"), (0.8, "#7448ee"),
                (1, "#b84bd0")],
        c_cols=[(0, "#e879f9"), (0.2, "#c062f2"), (0.6, "#8b5cf6"), (1, "#6b4af0")],
        gain_a=1.25, gain_b=0.95, gain_c=0.5,
        glow1="#8b6cff", glow1_op=0.14, glow2="#2dd4bf", glow2_op=0.05,
        glow3="#60a5fa", glow3_op=0.12,
        horizon="#0d1117", horizon_op=0.28, refl_op=0.38, stars=False,
        grain_op=0.022, vignette="#ffffff", vignette_op=0.7, rays=0.12, edge=1.0, hs=0.6, reach=0.15, halo=0.05,
        airglow="#9ad7f0", airglow_op=0.06, topfade_op=0.6, ripple="#ffffff", ripple_k=0.8,
        bloom="#5b8cff", bloom_op=0.12, glint="#3f7cff",
    ),
}


def fmt(v):
    return f"{v:.2f}".rstrip("0").rstrip(".")


def build(theme):
    T = THEMES[theme]
    rng = np.random.default_rng(7)

    # three curtain rasters (+ one alternate-ray copy of the main curtain for shimmer)
    A, a_box = curtain(11, 10, 250, base=220, amp=12, height=78 * T["hs"], colors=T["a_cols"],
                       gain=T["gain_a"], ray_strength=0.75 * T["rays"], edge_sigma=7 * T["edge"], ray_reach=1.2 * T["reach"], halo=T["halo"])
    A2, _ = curtain(11, 10, 250, base=220, amp=12, height=78 * T["hs"], colors=T["a_cols"],
                    gain=T["gain_a"], ray_strength=0.9 * T["rays"], edge_sigma=7 * T["edge"],
                    ray_seed=4242, ray_reach=1.2 * T["reach"], halo=T["halo"])
    B, b_box = curtain(23, 0, 214, base=160, amp=24, height=58 * T["hs"], colors=T["b_cols"],
                       gain=T["gain_b"], ray_strength=0.6 * T["rays"], edge_sigma=6 * T["edge"],
                       patchiness=1.0, ray_reach=1.0 * T["reach"], halo=T["halo"] * 0.6)
    C, c_box = curtain(37, 0, 140, base=98, amp=16, height=44 * T["hs"], colors=T["c_cols"],
                       gain=T["gain_c"], ray_strength=0.5 * T["rays"], edge_sigma=13 * T["edge"],
                       patchiness=1.0, ray_reach=0.8 * T["reach"],
                       ray_k=(10, 60))
    grain = grain_tile(5, dark=(theme == "dark"))

    def image(id_, arr, box):
        x, y, w, h = box
        return (f'<image id="{id_}" x="{x}" y="{y}" width="{w}" height="{h}" '
                f'preserveAspectRatio="none" xlink:href="{data_uri(arr)}"/>')

    def scroll(dur, direction):
        vals = "0 0;-1200 0" if direction < 0 else "-1200 0;0 0"
        return (f'<animateTransform attributeName="transform" type="translate" '
                f'values="{vals}" dur="{dur}s" repeatCount="indefinite"/>')

    def ease(attr, values, dur, typ=None, begin=0):
        n = len(values.split(";")) - 1
        ks = ";".join(["0.45 0 0.55 1"] * n)
        kt = ";".join(fmt(i / n) for i in range(n + 1))
        t = f' type="{typ}"' if typ else ""
        tag = "animateTransform" if typ else "animate"
        return (f'<{tag} attributeName="{attr}"{t} values="{values}" dur="{dur}s" '
                f'begin="{begin}s" calcMode="spline" keyTimes="{kt}" keySplines="{ks}" '
                f'repeatCount="indefinite"/>')

    hz = HORIZON
    s = []
    s.append(f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
             f'viewBox="0 0 {W} {H}" width="{W}" height="{H}" fill="none">')
    s.append("<defs>")
    s.append(f'<clipPath id="card"><rect width="{W}" height="{H}" rx="16"/></clipPath>')
    s.append(f'<linearGradient id="sky" x1="0" y1="0" x2="0" y2="{hz}" gradientUnits="userSpaceOnUse">'
             f'<stop offset="0" stop-color="{T["bg_top"]}"/><stop offset="0.6" stop-color="{T["bg_mid"]}"/>'
             f'<stop offset="1" stop-color="{T["bg_hor"]}"/></linearGradient>')
    # water: fades the mirrored sky into the water colour
    s.append(f'<linearGradient id="water" x1="0" y1="{hz}" x2="0" y2="{H}" gradientUnits="userSpaceOnUse">'
             f'<stop offset="0" stop-color="{T["water"]}" stop-opacity="{fmt(1 - T["refl_op"])}"/>'
             f'<stop offset="0.75" stop-color="{T["water"]}" stop-opacity="0.96"/>'
             f'<stop offset="1" stop-color="{T["water"]}"/></linearGradient>')
    s.append(f'<linearGradient id="hline" x1="0" y1="0" x2="{W}" y2="0" gradientUnits="userSpaceOnUse">'
             f'<stop offset="0" stop-color="{T["horizon"]}" stop-opacity="0"/>'
             f'<stop offset="0.5" stop-color="{T["horizon"]}" stop-opacity="1"/>'
             f'<stop offset="1" stop-color="{T["horizon"]}" stop-opacity="0"/></linearGradient>')
    for gid, col in (("g1", T["glow1"]), ("g2", T["glow2"]), ("g3", T["glow3"])):
        s.append(f'<radialGradient id="{gid}"><stop offset="0" stop-color="{col}"/>'
                 f'<stop offset="0.45" stop-color="{col}" stop-opacity="0.45"/>'
                 f'<stop offset="1" stop-color="{col}" stop-opacity="0"/></radialGradient>')
    s.append(f'<radialGradient id="glint"><stop offset="0" stop-color="{T["glint"]}"/>'
             f'<stop offset="1" stop-color="{T["glint"]}" stop-opacity="0"/></radialGradient>')
    s.append(f'<radialGradient id="vig" cx="0.5" cy="0.45" r="0.75">'
             f'<stop offset="0.55" stop-color="{T["vignette"]}" stop-opacity="0"/>'
             f'<stop offset="1" stop-color="{T["vignette"]}" stop-opacity="{T["vignette_op"]}"/></radialGradient>')
    s.append(f'<pattern id="grain" width="128" height="128" patternUnits="userSpaceOnUse">'
             f'<image width="128" height="128" xlink:href="{data_uri(grain)}"/></pattern>')
    s.append(f'<linearGradient id="air" x1="0" y1="{hz - 80}" x2="0" y2="{hz}" gradientUnits="userSpaceOnUse">'
             f'<stop offset="0" stop-color="{T["airglow"]}" stop-opacity="0"/>'
             f'<stop offset="0.7" stop-color="{T["airglow"]}" stop-opacity="0.55"/>'
             f'<stop offset="1" stop-color="{T["airglow"]}"/></linearGradient>')
    s.append(f'<linearGradient id="top" x1="0" y1="0" x2="0" y2="110" gradientUnits="userSpaceOnUse">'
             f'<stop offset="0" stop-color="{T["bg_top"]}"/>'
             f'<stop offset="1" stop-color="{T["bg_top"]}" stop-opacity="0"/></linearGradient>')
    s.append(f'<radialGradient id="bloom"><stop offset="0" stop-color="{T["bloom"]}"/>'
             f'<stop offset="0.35" stop-color="{T["bloom"]}" stop-opacity="0.35"/>'
             f'<stop offset="1" stop-color="{T["bloom"]}" stop-opacity="0"/></radialGradient>')
    s.append(image("imR", ripple_layer(9, T["ripple"], T["ripple_k"]), (0, hz, W, H - hz)))
    s.append(image("imA", A, a_box))
    s.append(image("imA2", A2, a_box))
    s.append(image("imB", B, b_box))
    s.append(image("imC", C, c_box))

    # ---- the animated aurora (defined once, drawn twice: sky + reflection)
    s.append('<g id="aurora">')
    ag = T["airglow_op"]
    s.append(f'<g opacity="{fmt(ag)}">{ease("opacity", f"{fmt(ag)};{fmt(ag * 1.6)};{fmt(ag)}", 31)}'
             f'<rect y="{hz - 80}" width="{W}" height="80" fill="url(#air)"/></g>')
    # broad colour fields (vector radial gradients, transform-animated)
    s.append(f'<g opacity="{T["glow1_op"]}">{ease("transform", "-60 0;70 10;-60 0", 41, "translate")}'
             f'<ellipse cx="760" cy="70" rx="520" ry="120" fill="url(#g1)"/></g>')
    s.append(f'<g opacity="{T["glow3_op"]}">{ease("transform", "50 0;-80 -6;50 0", 53, "translate")}'
             f'<ellipse cx="300" cy="120" rx="420" ry="110" fill="url(#g3)"/></g>')
    s.append(f'<g opacity="{T["glow2_op"]}">{ease("transform", "0 0;90 0;0 0", 37, "translate")}'
             f'<ellipse cx="520" cy="150" rx="560" ry="60" fill="url(#g2)"/></g>')
    # high magenta veil, drifting right
    s.append(f'<g>{scroll(260, +1)}<g opacity="1">{ease("opacity", "0.55;1;0.55", 29)}'
             f'<use xlink:href="#imC"/></g></g>')
    # blue curtain, drifting right
    s.append(f'<g>{scroll(190, +1)}<g>{ease("opacity", "0.75;1;0.75", 23, begin=-7)}'
             f'<g transform="translate(0 {b_box[1] + b_box[3]})"><g>'
             f'{ease("transform", "1 1;1 1.12;1 1", 19, "scale", begin=-5)}'
             f'<g transform="translate(0 {-(b_box[1] + b_box[3])})"><use xlink:href="#imB"/></g></g></g>'
             f'</g></g>')
    # main green curtain, drifting left, breathing, with a cross-faded ray layer for shimmer
    s.append(f'<g>{scroll(140, -1)}'
             f'<g transform="translate(0 {a_box[1] + a_box[3]})"><g>'
             f'{ease("transform", "1 1;1 1.1;1 1", 17, "scale")}'
             f'<g transform="translate(0 {-(a_box[1] + a_box[3])})">'
             f'<use xlink:href="#imA"/>'
             f'<g opacity="0">{ease("opacity", "0;0.85;0", 11, begin=-2)}<use xlink:href="#imA2"/></g>'
             f'</g></g></g></g>')
    s.append("</g>")
    s.append("</defs>")

    # ------------------------------------------------------------------ scene
    s.append('<g clip-path="url(#card)">')
    s.append(f'<rect width="{W}" height="{H}" fill="url(#sky)"/>')
    s.append(f'<rect y="{hz}" width="{W}" height="{H - hz}" fill="{T["water"]}"/>')

    if T["stars"]:
        srng = np.random.default_rng(3)
        stars = []
        for i in range(70):
            x = srng.uniform(12, W - 12)
            y = 8 + (hz - 50) * srng.random() ** 1.6
            r = srng.choice([0.55, 0.7, 0.9, 1.15], p=[0.45, 0.3, 0.17, 0.08])
            op = srng.uniform(0.25, 0.75) * (1 - 0.6 * y / hz)
            stars.append((x, y, r, op))
        s.append('<g fill="#ffffff">')
        for i, (x, y, r, op) in enumerate(stars):
            if i % 9 == 0:
                dur = 5 + (i % 5) * 1.7
                s.append(f'<circle cx="{fmt(x)}" cy="{fmt(y)}" r="{r}" opacity="{fmt(op)}">'
                         f'{ease("opacity", f"{fmt(op)};{fmt(op * 0.25)};{fmt(op)}", fmt(dur), begin=-i % 7)}'
                         f'</circle>')
            else:
                s.append(f'<circle cx="{fmt(x)}" cy="{fmt(y)}" r="{r}" opacity="{fmt(op)}"/>')
        s.append("</g>")

    # sky
    s.append('<use xlink:href="#aurora"/>')
    s.append(f'<rect width="{W}" height="110" fill="url(#top)" opacity="{T["topfade_op"]}"/>')
    # reflection: mirrored about the horizon, vertically compressed, then veiled by the water
    s.append(f'<g transform="translate(0 {hz}) scale(1 -0.62) translate(0 {-hz})">'
             f'<use xlink:href="#aurora"/></g>')
    s.append(f'<rect y="{hz}" width="{W}" height="{H - hz}" fill="url(#water)"/>')
    s.append('<use xlink:href="#imR"/>')
    s.append(f'<ellipse cx="{W / 2}" cy="{hz}" rx="{W * 0.42}" ry="7" fill="url(#bloom)" opacity="{T["bloom_op"]}"/>')
    # horizon line + travelling glint
    s.append(f'<rect x="0" y="{hz - 0.5}" width="{W}" height="1" fill="url(#hline)" opacity="{T["horizon_op"]}"/>')
    s.append(f'<g opacity="0">{ease("opacity", "0;0.7;0", 24)}'
             f'<g>{ease("transform", "-200 0;1400 0;-200 0", 48, "translate")}'
             f'<ellipse cx="0" cy="{hz}" rx="140" ry="2.2" fill="url(#glint)"/></g></g>')

    s.append(f'<rect width="{W}" height="{H}" fill="url(#grain)" opacity="{T["grain_op"]}"/>')
    if T["vignette_op"] > 0:
        s.append(f'<rect width="{W}" height="{H}" fill="url(#vig)"/>')
    s.append("</g>")
    s.append(f'<rect x="0.5" y="0.5" width="{W - 1}" height="{H - 1}" rx="15.5" '
             f'stroke="{T["border"]}" stroke-opacity="{T["border_op"]}"/>')
    s.append("</svg>")
    return "\n".join(s)


if __name__ == "__main__":
    for theme in ("dark", "light"):
        svg = build(theme)
        p = OUT / f"banner-{theme}.svg"
        p.write_text(svg)
        print(f"{p.name}: {len(svg.encode()) / 1024:.1f} KB")
