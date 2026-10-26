#!/usr/bin/env python3
"""Dot-matrix interference banner ("ripple tank").

Two coherent point sources emit the same slow wave. Any single-frequency field factors into a
STATIC amplitude and a STATIC phase per dot:

    cos(k d1 - wt) + cos(k d2 - wt) = 2 cos(k (d1 - d2) / 2) * cos(k (d1 + d2) / 2 - wt)

so every dot runs the same periodic pulse, only scaled by its amplitude A(x, y) and shifted by its
phase. That is what makes the piece cheap:

  * dots are zero-length sub-paths with round caps, so a dot's diameter is the path's stroke-width;
  * a static layer draws the resting grid once;
  * the "lit" overlay is split into (phase bucket x amplitude level) paths.  Each phase bucket is
    ONE <g> whose stroke-width carries the shared pulse; amplitude is applied with a scale()
    transform on the child path (stroke-width scales with the transform), so the whole field is
    driven by N_PHASE animations;
  * the animation is calcMode="discrete" on a common clock (STEPS frames per loop, every bucket
    offset is a whole number of frames), so the browser repaints only on clock ticks, like an LED
    panel refreshing, instead of every display frame.

Run:  python generate.py   -> banner-dark.svg, banner-light.svg
"""
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------------------- layout
W, H = 1200, 320
PITCH = 10.0
MARGIN_X, MARGIN_Y = 36.0, 30.0
PERIOD = 6.0           # seconds per wave cycle (= loop length)
STEPS = 120            # display frames per loop (discrete clock -> STEPS / PERIOD fps)
N_PHASE = 40           # phase buckets (must divide STEPS)
N_AMP = 8              # amplitude levels
D_REST = 1.5           # resting dot diameter
D_MAX = 7.0            # crest dot diameter at full amplitude
REST_PATTERN = True
A_CUT = 0.06           # amplitudes below this stay at rest

SRC = [(470.0, 130.0), (730.0, 190.0)]
WAVELEN = 96.0
DECAY_R = 420.0
DECAY_Y = 1.15
PERSP = None          # e.g. dict(hy=-40, znear=900, ybot=290, fog=3000)
SPREAD = 80.0          # per-source 1/sqrt(1 + r/SPREAD) falloff
PULSE_POW = 3.0
AMP_GAMMA = 0.8

THEMES = {
    "dark": dict(bg="#0d1117", border="#30363d", dot="#e6edf3", rest_op=0.20, lit_lo=0.55,
                 accent="#ff4f37"),
    "light": dict(bg="#ffffff", border="#d0d7de", dot="#1f2328", rest_op=0.16, lit_lo=0.55,
                  accent="#e5372a"),
}


def grid():
    nx = int((W - 2 * MARGIN_X) // PITCH) + 1
    ny = int((H - 2 * MARGIN_Y) // PITCH) + 1
    x0 = (W - (nx - 1) * PITCH) / 2
    y0 = (H - (ny - 1) * PITCH) / 2
    X, Y = np.meshgrid(x0 + PITCH * np.arange(nx), y0 + PITCH * np.arange(ny))
    return X.ravel(), Y.ravel()


def field(X, Y):
    """Complex superposition of coherent point sources -> (amplitude, phase) per dot."""
    k = 2 * math.pi / WAVELEN
    if PERSP:
        # screen -> ground plane: the dot panel shows a tilted ripple tank seen in perspective
        hy, znear, ybot = PERSP["hy"], PERSP["znear"], PERSP["ybot"]
        z = znear * (ybot - hy) / (Y - hy)
        U = (X - W / 2) * z / znear
        V = z
    else:
        U, V = X, Y
    S = np.zeros_like(X, dtype=complex)
    for sx, sy in SRC:
        d = np.hypot(U - sx, V - sy)
        S += np.exp(1j * k * d) / np.sqrt(1.0 + d / SPREAD)
    amp = np.abs(S) / len(SRC)
    phase = np.angle(S)
    cx = np.mean([s[0] for s in SRC])
    cy = np.mean([s[1] for s in SRC])
    r = np.hypot(U - cx, (V - cy) * DECAY_Y)
    decay = 1.0 / (1.0 + (r / DECAY_R) ** 2)
    if PERSP:
        decay = decay * np.exp(-((V / PERSP["fog"]) ** 2))
    # soft edge vignette so the field dissolves into the resting grid
    ex = np.clip(np.minimum(X - MARGIN_X, W - MARGIN_X - X) / 150.0, 0, 1)
    ey = np.clip(np.minimum(Y - MARGIN_Y, H - MARGIN_Y - Y) / 70.0, 0, 1)
    vign = np.sqrt(ex * (2 - ex) * ey * (2 - ey))
    return amp * decay * vign, phase


def pulse(theta):
    """Wave profile: a narrow crest and a long flat trough."""
    return ((1 + np.cos(theta)) / 2) ** PULSE_POW


def fmt(v, nd=2):
    s = f"{v:.{nd}f}".rstrip("0").rstrip(".")
    return s if s not in ("", "-0") else "0"


def dots_path(X, Y):
    return "".join(f"M{fmt(x)} {fmt(y)}h0" for x, y in zip(X, Y))


def build(theme):
    t = THEMES[theme]
    X, Y = grid()
    A, PH = field(X, Y)
    A = (A / A.max()) ** AMP_GAMMA
    lvl = np.clip(np.ceil((A - A_CUT) / (1 - A_CUT) * N_AMP), 0, N_AMP).astype(int)
    bucket = np.round((PH % (2 * math.pi)) / (2 * math.pi) * N_PHASE).astype(int) % N_PHASE

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
        f'fill="none" stroke-linecap="round">',
        f'<rect x=".5" y=".5" width="{W-1}" height="{H-1}" rx="16" fill="{t["bg"]}" '
        f'stroke="{t["border"]}"/>',
    ]
    if REST_PATTERN:
        # resting grid as a tiled pattern: one tile rasterised, then blitted (cheap to repaint)
        gx, gy = X.min() - PITCH / 2, Y.min() - PITCH / 2
        gw, gh = X.max() - X.min() + PITCH, Y.max() - Y.min() + PITCH
        out.append(
            f'<defs><pattern id="g" x="{fmt(gx)}" y="{fmt(gy)}" width="{fmt(PITCH)}" '
            f'height="{fmt(PITCH)}" patternUnits="userSpaceOnUse"><circle cx="{fmt(PITCH/2)}" '
            f'cy="{fmt(PITCH/2)}" r="{fmt(D_REST/2)}" fill="{t["dot"]}" fill-opacity="{t["rest_op"]}"/>'
            f'</pattern></defs><rect x="{fmt(gx)}" y="{fmt(gy)}" width="{fmt(gw)}" height="{fmt(gh)}" '
            f'fill="url(#g)"/>'
        )
    else:
        out.append(
            f'<path stroke="{t["dot"]}" stroke-width="{D_REST}" stroke-opacity="{t["rest_op"]}" '
            f'd="{dots_path(X, Y)}"/>'
        )

    # one shared pulse, sampled on the discrete clock
    th = 2 * math.pi * np.arange(STEPS) / STEPS
    widths = D_MAX * pulse(th)
    wv = ";".join(fmt(v) for v in widths)
    step = PERIOD / STEPS
    per_bucket = STEPS // N_PHASE
    out.append(f'<g stroke="{t["dot"]}">')
    for b in range(N_PHASE):
        # dot with phase p shows pulse(p - w t): start the cycle b/N_PHASE of the way through
        k = (-b * per_bucket) % STEPS
        begin = -k * step
        kids = []
        for L in range(1, N_AMP + 1):
            m = (lvl == L) & (bucket == b)
            if not m.any():
                continue
            a = L / N_AMP
            op = t["lit_lo"] + (1 - t["lit_lo"]) * a
            kids.append(
                f'<path transform="scale({fmt(a, 3)})" stroke-opacity="{fmt(op)}" '
                f'd="{dots_path(X[m] / a, Y[m] / a)}"/>'
            )
        if not kids:
            continue
        out.append(
            f'<g stroke-width="0"><animate attributeName="stroke-width" calcMode="discrete" '
            f'values="{wv}" dur="{PERIOD}s" begin="{fmt(begin, 3)}s" repeatCount="indefinite"/>'
            + "".join(kids) + "</g>"
        )
    out.append("</g>")

    for sx, sy in SRC:
        out.append(f'<circle cx="{fmt(sx)}" cy="{fmt(sy)}" r="2.6" fill="{t["accent"]}"/>')
    out.append("</svg>")
    return "".join(out)


def main():
    for theme in THEMES:
        svg = build(theme)
        p = os.path.join(HERE, f"banner-{theme}.svg")
        with open(p, "w") as fh:
            fh.write(svg)
        print(p, f"{len(svg)/1024:.1f} KB")


if __name__ == "__main__":
    main()
