#!/usr/bin/env python3
"""Orbits banner: the Chenciner-Montgomery figure-eight three-body choreography.

Three equal masses chase each other along one closed figure-eight curve. The orbit is
integrated numerically (RK4), rotated onto its principal axis, and drawn as hairline
line-art with fading comet trails, a morphing triangle joining the bodies, isochrone
ticks at equal time intervals and a sparse star field.

Run:  python generate.py        -> banner-dark.svg, banner-light.svg
Everything is deterministic (fixed seed).
"""
import math
import os
import random

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------------------- canvas
W, H = 1200, 340
CX, CY = W / 2, H / 2
SCALE = 340.0          # px per simulation unit
PERIOD_S = 15.0        # seconds of wall time per orbital period
N_PATH = 720           # polyline vertices along the orbit (uniform in time)
N_KEYS = 360           # animation keyframes per period
SEED = 7

THEMES = {
    "dark": dict(
        bg="#0d1117", card="#0d1117", border="#30363d",
        ink="#e6edf3",          # bodies / trails
        line="#8b949e",         # hairlines
        accent="#d9b26f",       # warm gold, used sparingly
        steel="#4f86d9",        # blued steel, trail tails
        vig_op=0.07,
        star="#c9d1d9",
        orbit_op=0.30, tick_op=0.45, frame_op=0.38, tri_op=0.28, star_scale=1.0,
        trail_peak=0.95, halo=True, arrow_len=46, arrow_op=0.85, caption_op=0.55, pulse_op=0.6,
    ),
    "light": dict(
        bg="#ffffff", card="#ffffff", border="#d0d7de",
        ink="#1f2328",
        line="#57606a",
        accent="#9a6b1f",
        steel="#2a5db0",
        vig_op=0.0,
        star="#57606a",
        orbit_op=0.28, tick_op=0.45, frame_op=0.34, tri_op=0.24, star_scale=0.75,
        trail_peak=0.9, halo=False, arrow_len=46, arrow_op=0.9, caption_op=0.6, pulse_op=0.7,
    ),
}


# ----------------------------------------------------------------------------- physics
def accel(pos):
    a = np.zeros_like(pos)
    for i in range(3):
        for j in range(3):
            if i != j:
                d = pos[j] - pos[i]
                a[i] += d / (np.dot(d, d) ** 1.5)
    return a


def simulate():
    """Integrate one period of the figure-eight with RK4 (G = m = 1)."""
    p1 = np.array([0.97000436, -0.24308753])
    v3 = np.array([-0.93240737, -0.86473146])
    pos = np.array([p1, -p1, [0.0, 0.0]])
    vel = np.array([-v3 / 2, -v3 / 2, v3])
    T = 6.32591398
    steps_per_sample = 40
    dt = T / (N_PATH * steps_per_sample)
    out = [pos[0].copy()]
    y = np.concatenate([pos, vel])

    def f(y):
        return np.concatenate([y[3:], accel(y[:3])])

    for _ in range(N_PATH):
        for _ in range(steps_per_sample):
            k1 = f(y)
            k2 = f(y + dt / 2 * k1)
            k3 = f(y + dt / 2 * k2)
            k4 = f(y + dt * k3)
            y = y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        out.append(y[0].copy())
    traj = np.array(out)          # N_PATH + 1 points, last ~= first
    print("closure error:", np.linalg.norm(traj[-1] - traj[0]))
    return traj[:-1]


def to_screen(traj):
    # rotate onto principal axis so the lobes sit exactly horizontal
    c = traj - traj.mean(axis=0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    ang = math.atan2(vt[0][1], vt[0][0])
    R = np.array([[math.cos(-ang), -math.sin(-ang)], [math.sin(-ang), math.cos(-ang)]])
    r = c @ R.T
    return np.column_stack([CX + SCALE * r[:, 0], CY - SCALE * r[:, 1]])


# ----------------------------------------------------------------------------- helpers
def f2(v):
    s = f"{v:.2f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def f1(v):
    s = f"{v:.1f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def mix(c1, c2, u):
    a = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * u):02x}" for x, y in zip(a, b))


def f4(v):
    s = f"{v:.4f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def poly_d(pts, closed=True):
    d = "M" + " ".join(f"{f2(x)} {f2(y)}" for x, y in pts[:1])
    d += "L" + " ".join(f"{f2(x)} {f2(y)}" for x, y in pts[1:])
    return d + ("Z" if closed else "")


def rounded(pts):
    """Round to the same precision the SVG uses, so our arc lengths match the browser's."""
    return np.round(pts, 2)


def trail_layers(n, peak, gamma=1.6):
    """Opacity per stacked dash so the composite fades as peak*(1-u)^gamma toward the tail."""
    alpha = [peak * (1 - (j - 0.5) / n) ** gamma for j in range(1, n + 1)] + [0.0]
    a = []
    for j in range(n):
        a.append(1 - (1 - alpha[j]) / (1 - alpha[j + 1]))
    return a


# ----------------------------------------------------------------------------- text -> outlines
FONT = "/usr/share/fonts/opentype/inter/Inter-Regular.otf"
_font_cache = {}


def text_path(text, x, y, size, anchor="start", tracking=0.0, font=FONT, features=None):
    """Shape `text` with HarfBuzz and return an SVG path `d` in user units (baseline at y)."""
    import uharfbuzz as hb
    from fontTools.ttLib import TTFont
    from fontTools.pens.svgPathPen import SVGPathPen
    from fontTools.pens.transformPen import TransformPen

    if font not in _font_cache:
        blob = hb.Blob.from_file_path(font)
        _font_cache[font] = (TTFont(font), hb.Font(hb.Face(blob)))
    tt, hbfont = _font_cache[font]
    upem = tt["head"].unitsPerEm
    buf = hb.Buffer()
    buf.add_str(text)
    buf.guess_segment_properties()
    hb.shape(hbfont, buf, features or {"tnum": True})
    sc = size / upem
    trk = tracking * upem
    total = sum(p.x_advance + trk for p in buf.glyph_positions) - trk
    x0 = x - {"start": 0, "middle": total / 2, "end": total}[anchor] * sc
    gs = tt.getGlyphSet()
    order = tt.getGlyphOrder()
    pen = SVGPathPen(gs, ntos=lambda v: f2(v))
    cx = 0
    for info, pos in zip(buf.glyph_infos, buf.glyph_positions):
        tp = TransformPen(pen, (sc, 0, 0, -sc, x0 + (cx + pos.x_offset) * sc, y - pos.y_offset * sc))
        gs[order[info.codepoint]].draw(tp)
        cx += pos.x_advance + trk
    return pen.getCommands()


# ----------------------------------------------------------------------------- build
def build(theme_name, pts):
    t = THEMES[theme_name]
    rng = random.Random(SEED)
    out = []
    A = out.append

    closed = np.vstack([pts, pts[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    P = cum[-1]

    A(f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
      f'viewBox="0 0 {W} {H}" width="{W}" height="{H}" fill="none">')
    A("<defs>")
    A(f'<path id="orbit" d="{poly_d(pts)}"/>')
    A(f'<clipPath id="card"><rect x="0.5" y="0.5" width="{W-1}" height="{H-1}" rx="16"/></clipPath>')
    A(f'<radialGradient id="vig" cx="50%" cy="50%" r="50%">'
      f'<stop offset="0" stop-color="{t["line"]}" stop-opacity="{t["vig_op"]}"/>'
      f'<stop offset="1" stop-color="{t["line"]}" stop-opacity="0"/></radialGradient>')
    A(f'<radialGradient id="halo"><stop offset="0" stop-color="{t["ink"]}" stop-opacity="0.22"/>'
      f'<stop offset=".35" stop-color="{t["ink"]}" stop-opacity="0.07"/>'
      f'<stop offset="1" stop-color="{t["ink"]}" stop-opacity="0"/></radialGradient>')
    A("</defs>")

    # card
    A(f'<rect x="0.5" y="0.5" width="{W-1}" height="{H-1}" rx="16" fill="{t["card"]}"/>')
    A('<g clip-path="url(#card)">')
    A(f'<ellipse cx="{CX}" cy="{CY}" rx="520" ry="200" fill="url(#vig)"/>')

    # ---- star field (sparse, faint, a few twinkle slowly)
    stars = []
    while len(stars) < 90:
        x, y = rng.uniform(14, W - 14), rng.uniform(12, H - 12)
        # keep the region hugging the curve clean
        dmin = np.min(np.hypot(pts[::6, 0] - x, pts[::6, 1] - y))
        if dmin < 16:
            continue
        stars.append((x, y, dmin))
    A(f'<g fill="{t["star"]}">')
    for i, (x, y, dmin) in enumerate(stars):
        r = rng.choice([0.45, 0.55, 0.65, 0.8, 1.0])
        op = rng.uniform(0.12, 0.5) * t["star_scale"]
        if dmin < 60:
            op *= 0.6
        if i % 7 == 0:
            dur = rng.choice([7.5, 10, 15])
            lo = op * 0.25
            A(f'<circle cx="{f2(x)}" cy="{f2(y)}" r="{r}" opacity="{f2(op)}">'
              f'<animate attributeName="opacity" values="{f2(op)};{f2(lo)};{f2(op)}" '
              f'dur="{dur}s" begin="-{f2(rng.uniform(0, dur))}s" repeatCount="indefinite" '
              f'calcMode="spline" keySplines=".45 0 .55 1;.45 0 .55 1"/></circle>')
        else:
            A(f'<circle cx="{f2(x)}" cy="{f2(y)}" r="{r}" opacity="{f2(op)}"/>')
    A("</g>")

    # ---- frame: hairline rule with fine ruler ticks top and bottom (instrument scale)
    fo = t["frame_op"]
    A(f'<g stroke="{t["line"]}" stroke-width="0.6" opacity="{fo}">')
    x0, x1 = 40, W - 40
    for yb, sgn in ((22, 1), (H - 22, -1)):
        A(f'<path d="M{x0} {yb}H{x1}"/>')
        d = []
        for k, x in enumerate(range(x0, x1 + 1, 10)):
            L = 6 if (x - CX) % 100 == 0 else (3.5 if (x - CX) % 50 == 0 else 2)
            d.append(f"M{x} {yb}v{f2(sgn * L)}")
        A(f'<path d="{"".join(d)}"/>')
    A("</g>")

    # ---- axes through the centre of mass (faint), with crosshair
    A(f'<g stroke="{t["line"]}" stroke-width="0.5" opacity="{f2(fo * 0.7)}">')
    A(f'<path d="M60 {CY}H{CX-12}M{CX+12} {CY}H{W-60}" stroke-dasharray="1 5"/>')
    A(f'<path d="M{CX} 34V{CY-12}M{CX} {CY+12}V{H-34}" stroke-dasharray="1 5"/>')
    A("</g>")

    # ---- the orbit itself, hairline
    A(f'<use href="#orbit" xlink:href="#orbit" stroke="{t["line"]}" stroke-width="0.75" '
      f'opacity="{t["orbit_op"]}"/>')

    # ---- isochrone ticks: equal time intervals along the orbit (24 per period)
    ticks = []
    n_ticks = 24
    for k in range(n_ticks):
        i = k * len(pts) // n_ticks
        p = pts[i]
        tan = pts[(i + 1) % len(pts)] - pts[i - 1]
        tan /= np.linalg.norm(tan)
        nrm = np.array([-tan[1], tan[0]])
        L = 5.5 if k % 8 == 0 else 3.5
        a, b = p - nrm * L, p + nrm * L
        ticks.append(f"M{f2(a[0])} {f2(a[1])}L{f2(b[0])} {f2(b[1])}")
    A(f'<path d="{"".join(ticks)}" stroke="{t["line"]}" stroke-width="0.7" '
      f'opacity="{t["tick_op"]}"/>')

    # ---- keyframes
    idx = [int(round(k * len(pts) / N_KEYS)) for k in range(N_KEYS)]
    key_times = [k / N_KEYS for k in range(N_KEYS)] + [1.0]
    s_vals = [cum[i] for i in idx] + [P]
    kt = ";".join(f4(v) for v in key_times)
    kp = ";".join(f4(min(s / P, 1.0)) for s in s_vals)
    dash_vals = ";".join(f2(-s) for s in s_vals)
    phases = [0.0, PERIOD_S / 3, 2 * PERIOD_S / 3]

    def begin(ph):
        return "0s" if ph == 0 else f"-{f4(ph)}s"

    # ---- triangle joining the three bodies (path d morph)
    tri = []
    for k in range(N_KEYS + 1):
        i = idx[k % N_KEYS]
        n = len(pts)
        a, b, c = pts[i], pts[(i + n // 3) % n], pts[(i + 2 * n // 3) % n]
        tri.append(f"M{f1(a[0])} {f1(a[1])}L{f1(b[0])} {f1(b[1])}L{f1(c[0])} {f1(c[1])}Z")
    A(f'<path d="{tri[0]}" stroke="{t["line"]}" stroke-width="0.6" opacity="{t["tri_op"]}" '
      f'stroke-dasharray="2 3">'
      f'<animate attributeName="d" values="{";".join(tri)}" keyTimes="{kt}" dur="{PERIOD_S}s" '
      f'repeatCount="indefinite"/></path>')

    # ---- comet trails: stacked dashes ending exactly at the body
    n_layers = 14
    trail_len = 0.26 * P
    layer_ops = trail_layers(n_layers, t["trail_peak"])
    for ph in phases:
        A('<g stroke-linecap="round" stroke-dashoffset="0">')
        A(f'<animate attributeName="stroke-dashoffset" values="{dash_vals}" keyTimes="{kt}" '
          f'dur="{PERIOD_S}s" begin="{begin(ph)}" repeatCount="indefinite"/>')
        for j in range(n_layers):
            ln = trail_len * (j + 1) / n_layers
            wdt = 2.4 - 1.6 * j / (n_layers - 1)
            col = mix(t["ink"], t["steel"], (j / (n_layers - 1)) ** 0.8)
            A(f'<use href="#orbit" xlink:href="#orbit" stroke="{col}" stroke-width="{f2(wdt)}" '
              f'opacity="{f4(layer_ops[j])}" stroke-dasharray="0 {f2(P - ln)} {f2(ln)} 0"/>')
        A("</g>")

    # ---- bodies: soft halo, solid dot, hairline ring, velocity vector (body-local coords)
    n = len(pts)
    vel = np.array([(pts[(i + 1) % n] - pts[i - 1]) for i in range(n)])
    speed = np.linalg.norm(vel, axis=1)
    vmax = speed.max()
    arrows = []
    for k in range(N_KEYS + 1):
        i = idx[k % N_KEYS]
        u = vel[i] / speed[i]
        nv = np.array([-u[1], u[0]])
        L = t["arrow_len"] * speed[i] / vmax
        a = u * 11
        b = u * (11 + L)
        h1 = b - u * 4.5 + nv * 2.2
        h2 = b - u * 4.5 - nv * 2.2
        arrows.append(f"M{f1(a[0])} {f1(a[1])}L{f1(b[0])} {f1(b[1])}"
                      f"M{f1(h1[0])} {f1(h1[1])}L{f1(b[0])} {f1(b[1])}L{f1(h2[0])} {f1(h2[1])}")
    arrow_vals = ";".join(arrows)
    print("speed ratio min/max:", speed.min() / vmax)
    for ph in phases:
        A("<g>")
        A(f'<animateMotion dur="{PERIOD_S}s" begin="{begin(ph)}" repeatCount="indefinite" '
          f'calcMode="linear" keyPoints="{kp}" keyTimes="{kt}">'
          f'<mpath href="#orbit" xlink:href="#orbit"/></animateMotion>')
        if t["halo"]:
            A('<circle r="22" fill="url(#halo)"/>')
        A(f'<path d="{arrows[0]}" stroke="{t["accent"]}" stroke-width="0.8" stroke-linecap="round" '
          f'stroke-linejoin="round" opacity="{t["arrow_op"]}">'
          f'<animate attributeName="d" values="{arrow_vals}" keyTimes="{kt}" dur="{PERIOD_S}s" '
          f'begin="{begin(ph)}" repeatCount="indefinite"/></path>')
        A(f'<circle r="7.5" stroke="{t["ink"]}" stroke-width="0.6" opacity="0.45"/>')
        A(f'<circle r="3.4" fill="{t["ink"]}"/>')
        A("</g>")

    # ---- centre of mass marker (accent)
    # a faint gold ring breathes out of the centre each time a body passes through it
    # (an Euler collinear configuration: happens every T/6)
    i0 = int(np.argmin(np.hypot(pts[:, 0] - CX, pts[:, 1] - CY)))
    six = PERIOD_S / 6
    tc = (i0 / len(pts) * PERIOD_S) % six
    A(f'<circle cx="{CX}" cy="{CY}" r="4" stroke="{t["accent"]}" stroke-width="0.7" opacity="0">'
      f'<animate attributeName="r" values="4;26;26" keyTimes="0;.6;1" dur="{f4(six)}s" '
      f'begin="-{f4(six - tc)}s" repeatCount="indefinite" calcMode="spline" keySplines=".2 .6 .4 1;0 0 1 1"/>'
      f'<animate attributeName="opacity" values="{t["pulse_op"]};0;0" keyTimes="0;.6;1" dur="{f4(six)}s" '
      f'begin="-{f4(six - tc)}s" repeatCount="indefinite" calcMode="spline" keySplines=".3 .5 .5 1;0 0 1 1"/>'
      f'</circle>')
    A(f'<g stroke="{t["accent"]}" stroke-width="0.8">')
    A(f'<circle cx="{CX}" cy="{CY}" r="4" opacity="0.9"/>')
    A(f'<path d="M{CX-9} {CY}h5M{CX+4} {CY}h5M{CX} {CY-9}v5M{CX} {CY+4}v5" opacity="0.9"/>')
    A("</g>")

    # ---- fine print, like the small type on a dial
    cap = []
    yb = H - 34
    cap.append(text_path("m₁ = m₂ = m₃", 44, yb, 9.5, tracking=0.04))
    cap.append(text_path("T = 6.32591398", W - 44, yb, 9.5, anchor="end", tracking=0.04))
    cap.append(text_path("x₁(0) = −x₂(0) = (0.97000436, −0.24308753)", 44, 42, 9.5, tracking=0.04))
    cap.append(text_path("ẋ₃(0) = −2ẋ₁(0) = (−0.93240737, −0.86473146)", W - 44, 42, 9.5,
                         anchor="end", tracking=0.04))
    A(f'<path d="{"".join(cap)}" fill="{t["line"]}" opacity="{t["caption_op"]}"/>')

    A("</g>")  # clip
    A(f'<rect x="0.5" y="0.5" width="{W-1}" height="{H-1}" rx="16" stroke="{t["border"]}"/>')
    A("</svg>")
    return "\n".join(out)


def main():
    pts = rounded(to_screen(simulate()))
    for name in THEMES:
        svg = build(name, pts)
        path = os.path.join(HERE, f"banner-{name}.svg")
        with open(path, "w") as fh:
            fh.write(svg)
        print(path, len(svg.encode()) // 1024, "KB")


if __name__ == "__main__":
    main()
