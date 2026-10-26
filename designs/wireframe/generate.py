#!/usr/bin/env python3
"""Hidden-line wireframe of a two-source interference surface.

Two coherent point sources on a plane emit circular ripples; their sum is
rendered as a perspective wireframe with hidden-line removal (painter's
algorithm: every row is drawn far -> near together with an opaque band in the
background colour that hides whatever lies behind it).  Every frame is
computed here and baked into SMIL `d` keyframe morphs, so the browser only
interpolates numbers.  The wave phase completes exactly one cycle per loop,
so the animation is seamless.

The scene is mirror-symmetric about the centre line, so only the left half is
stored; each row group is re-used through a mirrored <use>.

    python generate.py                 -> banner-dark.svg, banner-light.svg
    python generate.py OUTDIR KEY=VAL  -> override parameters (for experiments)
"""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = HERE
ARGS = {}
for a in sys.argv[1:]:
    if "=" in a:
        k, v = a.split("=", 1)
        ARGS[k] = v
    else:
        OUT = a


def param(key, default):
    return type(default)(ARGS.get(key, default))


# ---------------------------------------------------------------- canvas
W, H = 1200, 340
SC = 2                         # path coordinates in 1/SC px integer units
NF = param("NF", 12)           # keyframes per loop
DUR = param("DUR", 16.0)       # seconds per loop

# ---------------------------------------------------------------- camera
CAM_Y = param("CAM_Y", 4.5)    # camera height above the mean plane
PITCH = math.radians(param("PITCH", 16.0))
FOCAL = 1000.0
CX, CY = W / 2, H * 0.42
COS, SIN = math.cos(PITCH), math.sin(PITCH)


def project(x, y, z):
    """World -> screen px.  Camera at (0, CAM_Y, 0) looking +z, pitched down."""
    yc = y - CAM_Y
    zc = z * COS - yc * SIN
    yv = z * SIN + yc * COS
    return CX + FOCAL * x / zc, CY - FOCAL * yv / zc


def plane_z(v):
    """Depth z where screen row v meets the plane y = 0."""
    dy = -(v - CY) / FOCAL
    return CAM_Y / -(dy * COS - SIN) * (dy * SIN + COS)


def depth(z):
    return z * COS + CAM_Y * SIN


def xmax(z):
    """World half-width of the view (plus margin) at depth z."""
    return (W / 2 + 30) / FOCAL * depth(z)


# ---------------------------------------------------------------- surface
GRID = param("GRID", 0.5)
Z_MIN = plane_z(H + 30)
Z_MAX = plane_z(param("TOPY", -10.0))
SEP = param("SEP", 4.0)
Z_SRC = plane_z(H * param("SRCY", 0.4))
SOURCES = [(-SEP, Z_SRC), (SEP, Z_SRC)]
K = 2 * math.pi / param("WL", 2.2)
AMP = param("AMP", 0.6)


def height(x, z, t):
    """Surface height at world (x, z) for loop phase t in [0, 1)."""
    x = np.asarray(x, float)
    z = np.asarray(z, float)
    h = np.zeros(np.broadcast(x, z).shape)
    for sx, sz in SOURCES:
        r = np.hypot(x - sx, z - sz)
        h += (1.0 + 0.6 * r) ** -0.7 * np.sin(K * r - 2 * math.pi * t)
    return AMP * h


# ---------------------------------------------------------------- light falloff
# Elliptical radial gradients (screen space) fade the mesh into the background.
GX, GY = W / 2, H * param("GLOWY", 0.62)
ROW_R = (param("GRX", 640.0), param("GRY", 260.0))
COL_R = (param("CRX", 600.0), param("CRY", 200.0))
STOPS = [(0, 1), (.35, .85), (.65, .35), (.88, .08), (1, 0)]


def glow_r(px, py, rad):
    return np.hypot((np.asarray(px) - GX) / rad[0], (np.asarray(py) - GY) / rad[1])


def glow_half_width(py, rad):
    v = 1 - ((min(py, GY) - GY) / rad[1]) ** 2
    return rad[0] * math.sqrt(max(v, 0.0))


# ---------------------------------------------------------------- geometry
SEG_PX = param("SEG", 26.0)    # screen spacing of curve vertices along rows
CSEG_PX = param("CSEG", 12.0)  # screen spacing of column polyline vertices


def row_pts(z, t, xm):
    """Left half of a row, from -xm to x = 0 (screen centre)."""
    n = max(int(W / 2 / SEG_PX * xm / (xmax(z) + GRID)), 4)
    x = np.linspace(-xm, 0.0, n + 1)
    return project(x, height(x, z, t), np.full_like(x, z))


def col_pts(x, z0, z1, t, n):
    z = np.linspace(z0, z1, n + 1)
    return project(np.full_like(z, x), height(x, z, t), z)


def q(v):
    return np.round(np.asarray(v) * SC).astype(int)


def nums(vals):
    out = []
    for i, p in enumerate(vals):
        s = str(int(p))
        out.append(" " + s if i and p >= 0 else s)
    return "".join(out)


def interleave(a, b):
    return np.stack([a, b], -1).ravel()


def row_d(sx, sy, bottom):
    """Row as a Catmull-Rom spline (c + s commands), closed below.

    The closure runs just past the centre line (hidden by the clip that
    removes the right half) and back along `bottom`, where the next nearer
    band covers it.
    """
    P = np.stack([sx, sy], -1)
    # end tangent mirrors across the centre line, so the row is smooth there
    Pe = np.vstack([2 * P[0] - P[1], P, [2 * CX - P[-2, 0], P[-2, 1]]])
    T = (Pe[2:] - Pe[:-2]) / 6.0
    A = np.round(P * SC).astype(int)
    c1 = np.round((P[:-1] + T[:-1]) * SC).astype(int)
    c2 = np.round((P[1:] - T[1:]) * SC).astype(int)
    d = f"M{A[0,0]} {A[0,1]}c" + nums(np.concatenate([c1[0] - A[0], c2[0] - A[0], A[1] - A[0]]))
    rel = []
    for k in range(1, len(A) - 1):
        rel += list(c2[k] - A[k]) + list(A[k + 1] - A[k])
    if rel:
        d += "s" + nums(rel)
    return d + f"h{2 * SC}V{bottom}H{A[0,0]}Z"


def segs_d(segs):
    """Many short polylines in one path, chained with relative moves."""
    parts = []
    cx = cy = 0
    for k, (sx, sy) in enumerate(segs):
        ax, ay = q(sx), q(sy)
        parts.append(f"M{ax[0]} {ay[0]}" if k == 0 else "m" + nums([ax[0] - cx, ay[0] - cy]))
        parts.append("l" + nums(interleave(np.diff(ax), np.diff(ay))))
        cx, cy = ax[-1], ay[-1]
    return "".join(parts)


# ---------------------------------------------------------------- themes
THEMES = {
    "dark": dict(bg="#0d1117", border="#30363d", near="#a5d0ff", far="#8a7dff",
                 row_op=0.95, col_op=0.5),
    "light": dict(bg="#ffffff", border="#d0d7de", near="#0a3069", far="#6e56cf",
                  row_op=0.9, col_op=0.42),
}


def scene():
    """Theme-independent geometry: list of (stroke width, row ds, col ds)."""
    ts = [i / NF for i in range(NF)]
    rows = []
    for z in np.arange(Z_MIN, Z_MAX + 1e-9, GRID)[::-1]:          # far -> near
        x = np.linspace(-xmax(z), 0, 200)
        ymax = max(project(x, height(x, z, t), z)[1].max() for t in ts)
        hp = glow_half_width(ymax, ROW_R) + 24
        if hp > 40:
            rows.append((z, min(xmax(z) + GRID, hp / FOCAL * depth(z))))
    pts = [[row_pts(z, t, xm) for t in ts] for z, xm in rows]
    ymaxs = [max(p[1].max() for p in fr) for fr in pts]

    out = []
    for i, (z, xm) in enumerate(rows):
        # closing line of this band: just below the next row's lowest point
        if i + 1 < len(rows):
            b = max(ymaxs[: i + 2]) + 3
        else:
            b = H + 20
        bottom = int(round(min(b, H + 20) * SC))
        cols = None
        if i > 0:
            zf = rows[i - 1][0]
            n = int(xmax(zf) / GRID + 0.5)
            xs = -(np.arange(n) + 0.5) * GRID
            px, py = project(xs, 0 * xs, np.full_like(xs, z))
            xs = xs[glow_r(px, py, COL_R) < 1.0]
            if len(xs):
                a = project(0, 0, z)[1] - project(0, 0, zf)[1]
                nsub = max(1, int(a / CSEG_PX + 0.5))
                cols = [segs_d([col_pts(x, zf, z, t, nsub) for x in xs]) for t in ts]
        sw = float(np.clip(1.4 * Z_MIN / z, 0.55, 1.4)) * SC
        ds = [row_d(*p, bottom) for p in pts[i]]
        out.append((sw, ds, cols))
    return out


def gradient(gid, th, rad):
    gy = GY * SC
    rx, ry = rad[0] * SC, rad[1] * SC
    k = ry / rx
    st = "".join(f'<stop offset="{a:g}" stop-color="{th["near"] if a < .5 else th["far"]}" '
                 f'stop-opacity="{b:g}"/>' for a, b in STOPS)
    return (f'<radialGradient id="{gid}" gradientUnits="userSpaceOnUse" cx="{W*SC/2:g}" '
            f'cy="{gy:g}" r="{rx:g}" gradientTransform="matrix(1 0 0 {k:.3f} 0 {gy*(1-k):.1f})">'
            f'{st}</radialGradient>')


def build(th, geo):
    def anim(ds):
        return (f'<animate attributeName="d" dur="{DUR:g}s" repeatCount="indefinite" '
                f'values="{";".join(ds + [ds[0]])}"/>')

    o = [f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
         f'viewBox="0 0 {W*SC} {H*SC}" width="{W}" height="{H}" fill="none">', "<defs>",
         gradient("f", th, ROW_R), gradient("k", th, COL_R),
         f'<clipPath id="c"><rect x="1" y="1" width="{W*SC-2}" height="{H*SC-2}" rx="{16*SC}"/></clipPath>',
         f'<clipPath id="L"><rect x="-100" y="-100" width="{W*SC//2+100}" height="{H*SC+300}"/></clipPath>',
         "</defs>",
         f'<rect x="1" y="1" width="{W*SC-2}" height="{H*SC-2}" rx="{16*SC}" fill="{th["bg"]}" '
         f'stroke="{th["border"]}" stroke-width="2"/>',
         f'<g clip-path="url(#c)" stroke="url(#f)" stroke-linejoin="round">']
    mir = f"matrix(-1 0 0 1 {W*SC} 0)"
    for i, (sw, ds, cols) in enumerate(geo):
        o.append(f'<g id="g{i}">')
        if cols:
            o.append(f'<path stroke="url(#k)" stroke-width="{0.9*SC:g}" stroke-opacity="{th["col_op"]}" '
                     f'd="{cols[0]}">{anim(cols)}</path>')
        o.append(f'<use xlink:href="#r{i}" fill="{th["bg"]}" stroke="none"/>')
        o.append(f'<path id="r{i}" clip-path="url(#L)" stroke-width="{sw:.1f}" '
                 f'stroke-opacity="{th["row_op"]}" d="{ds[0]}">{anim(ds)}</path>')
        o.append(f'</g><use xlink:href="#g{i}" transform="{mir}"/>')
    o.append("</g></svg>")
    return "\n".join(o)


def main():
    geo = scene()
    for name, th in THEMES.items():
        svg = build(th, geo)
        p = os.path.join(OUT, f"banner-{name}.svg")
        with open(p, "w") as fh:
            fh.write(svg)
        print(p, f"{len(svg.encode())/1024:.1f} KB, {len(geo)} rows")


if __name__ == "__main__":
    main()
