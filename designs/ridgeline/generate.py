#!/usr/bin/env python3
"""Ridgeline banner: a pulsar-style waterfall of stacked ridgelines.

Approach
--------
M ridge profiles are generated from a field that is smooth and periodic in the
stack direction (line index j = 0..M-1 wraps), plus per-line fine texture.
Each ridge is one open <path> whose two ends sit on the baseline, off-card.
It is stroked (the line) and filled with the card colour: an open path is
filled as if closed by the straight chord between its ends, which is the
baseline, so the fill occludes exactly the ridges behind it while the chord
itself is never stroked.  No extra geometry, no dashes, no masks.

Paint order is back to front (top to bottom).  The M ridges are drawn twice,
the second copy a <use> shifted down by one period (M * GAP), and the whole
stack is translated down by exactly one period per loop.  At the end of the
loop the picture is identical to the start, and paint order is always correct,
so the motion is one seamless, constant drift toward the viewer: new pulses
emerge out of the haze at the top and slide forward under the bottom edge.
One ridge per period carries a warm accent.  The flank dimming and the haze
at the top are static card-coloured overlays, so no moving element carries a
gradient (except the single accent ridge).

Usage:  python generate.py       -> banner-dark.svg, banner-light.svg
"""
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------- parameters
W, H = 1200, 340
SEED = int(os.environ.get('SEED', 7))
GAP = float(os.environ.get('GAP', 8.5))                # vertical spacing between ridges (px)
M = int(os.environ.get('M', 56))                  # distinct ridges per period (period = M * GAP px)
DUR = float(os.environ.get('DUR', 96))   # seconds per loop (one period of drift)
X0, X1 = -24, W + 24      # profile extent (beyond the card, hidden by clip)
DX = 2.0                  # dense sampling step before simplification
TOL = 0.12                # polyline simplification tolerance (px)
AMP = float(os.environ.get('AMP', 95.0))               # peak height scale (px)
TOP_FADE = 110            # haze height at the top (px)
ACCENT_RANK = int(os.environ.get('ARANK', 2))  # accent = ridge with the n-th tallest pulse
SW = float(os.environ.get('SW', 1.35))   # line weight (px)
JITTER = float(os.environ.get('JITTER', 0.3))
NSIG = float(os.environ.get('NSIG', 1.4))   # texture correlation (samples)
NFLOOR = float(os.environ.get('NFLOOR', 0.002))
SWELL = float(os.environ.get('SWELL', 0.55))  # px
NGAIN = float(os.environ.get('NGAIN', 0.013))


# ------------------------------------------------------------------- profiles
def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def build_profiles(seed):
    rng = np.random.default_rng(seed)
    xs = np.arange(X0, X1 + 1e-9, DX)
    theta = 2 * np.pi * np.arange(M) / M

    def periodic(n_harm=3, amp=1.0):
        """Smooth random periodic function of the line index."""
        out = np.zeros(M)
        for h in range(1, n_harm + 1):
            out += rng.normal(0, 1) / h * np.cos(h * theta + rng.uniform(0, 2 * np.pi))
        return amp * out / max(1e-9, np.abs(out).max())

    # broad activity envelope: calm flanks, a wide active middle
    env = smoothstep(150, 470, xs) * (1 - smoothstep(760, 1060, xs))
    env = env ** 1.3

    # pulse components (centre, width, amplitude): a main cluster slightly
    # right of centre, a weaker leading shoulder and a faint trailing echo
    spec = [(500, 40, 0.22), (566, 16, 0.50), (604, 9, 0.80), (628, 11, 1.10),
            (652, 8, 0.85), (680, 13, 0.60), (718, 22, 0.40), (790, 36, 0.22),
            (400, 40, 0.14), (900, 48, 0.12)]
    comps = []
    for c, w, a in spec:
        comps.append(dict(c=c, w=w, a=a,
                          drift=periodic(4, rng.uniform(4, 12)),
                          gain=periodic(5, 1.0)))
    swing = periodic(2, 28.0)                     # whole cluster meanders

    profiles = []
    for j in range(M):
        y = np.zeros_like(xs)
        for cp in comps:
            g = np.clip(0.5 + 0.5 * cp['gain'][j], 0, None) ** 1.5
            g *= rng.lognormal(0, JITTER)            # pulse-to-pulse jitter
            c = cp['c'] + swing[j] + cp['drift'][j] + rng.normal(0, 2.5)
            y += cp['a'] * g * np.exp(-0.5 * ((xs - c) / cp['w']) ** 2)
        # fine texture: band-limited noise, strongest where the signal lives
        n = rng.normal(0, 1, xs.size)
        k = np.exp(-0.5 * (np.arange(-10, 11) / NSIG) ** 2)
        n = np.convolve(n, k / k.sum(), mode='same') * math.sqrt(NSIG * 3.5)
        y = y * (0.15 + 0.85 * env) * env + n * (NFLOOR + NGAIN * env)
        # slow swell across the full width (px): keeps the calm flanks from
        # being ruler-straight, which would shimmer as they drift sub-pixel
        sw = rng.normal(0, 1, xs.size + 160)
        kk = np.exp(-0.5 * (np.arange(-60, 61) / 16.0) ** 2)
        sw = np.convolve(sw, kk / kk.sum(), mode='same')[80:-80]
        sw = sw / (sw.std() + 1e-9) * SWELL
        profiles.append(AMP * y + sw)
    return xs, profiles


# ------------------------------------------------------------ simplification
def rdp(pts, tol):
    keep = np.zeros(len(pts), bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        p, q = pts[a], pts[b]
        d = q - p
        seg = pts[a + 1:b] - p
        dist = np.abs(seg[:, 0] * d[1] - seg[:, 1] * d[0]) / (np.hypot(*d) + 1e-12)
        i = int(np.argmax(dist))
        if dist[i] > tol:
            m = a + 1 + i
            keep[m] = True
            stack += [(a, m), (m, b)]
    return pts[keep]


def fmt(v):
    s = f"{v:.1f}".rstrip('0').rstrip('.')
    return '0' if s in ('-0', '') else s


def ridge_d(xs, prof, base):
    pts = np.column_stack([xs, base - prof])
    pts[0, 1] = pts[-1, 1] = base        # pin the off-card ends to the baseline
    pts = rdp(pts, TOL)
    pts = np.round(pts * 10) / 10
    out = [f"M{fmt(pts[0, 0])} {fmt(pts[0, 1])}"]
    prev = pts[0]
    segs = []
    for p in pts[1:]:
        dx, dy = p - prev
        segs.append(f"{fmt(dx)} {fmt(dy)}")
        prev = p
    # relative lineto chain; negative numbers need no separator
    chain = 'l' + ' '.join(segs)
    chain = chain.replace(' -', '-')
    return out[0] + chain


# ------------------------------------------------------------------- output
THEMES = {
    'dark': dict(bg='#0d1117', border='#30363d', line='#c3ccd8', op_hi=0.9,
                 op_lo=0.42, accent='#f0a868', haze='#0d1117'),
    'light': dict(bg='#ffffff', border='#d0d7de', line='#1f2630', op_hi=0.8,
                  op_lo=0.38, accent='#cf6a1f', haze='#ffffff'),
}


def accent_line(profiles):
    """The ridge with the ACCENT_RANK-th tallest pulse."""
    peaks = [p.max() for p in profiles]
    order = np.argsort(peaks)[::-1]
    return int(order[ACCENT_RANK])


def line_opacity(th, o):
    """Horizontal vignette: lines are brightest across the middle."""
    t = min(1.0, abs(o - 0.53) / 0.47)
    t = max(0.0, (t - 0.25) / 0.75)
    t = t * t * (3 - 2 * t)
    return th['op_hi'] + (th['op_lo'] - th['op_hi']) * t


def accent_stops(th):
    """Accent stroke: warm across the pulse, blending back to the line colour."""
    out = []
    for o in np.linspace(0, 1, 11):
        a = max(0.0, 1 - abs(o - 0.53) / 0.3)
        a = a * a * (3 - 2 * a)
        col = mix(th['line'], th['accent'], a)
        op = th['op_hi'] + (1 - th['op_hi']) * a
        out.append(f'<stop offset="{o:.1f}" stop-color="{col}" stop-opacity="{op:.2f}"/>')
    return '\n'.join(out)


def vignette_stops(th):
    """Static card-coloured overlay that dims the lines toward the flanks
    (cheaper than a gradient stroke on every moving ridge)."""
    out = []
    for o in np.linspace(0, 1, 11):
        a = 1 - line_opacity(th, o) / th['op_hi']
        out.append(f'<stop offset="{o:.1f}" stop-color="{th['bg']}" stop-opacity="{a:.2f}"/>')
    return '\n'.join(out)


def mix(c1, c2, t):
    a = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
    return '#' + ''.join(f'{round(x + (y - x) * t):02x}' for x, y in zip(a, b))


def build_svg(theme, xs, profiles):
    th = THEMES[theme]
    period = M * GAP
    y_start = -period - 10          # two copies cover the card for every offset
    paths = []
    accent = accent_line(profiles)
    for j, prof in enumerate(profiles):
        base = y_start + j * GAP
        cls = (f' stroke="url(#acc)" stroke-width="{SW * 1.3:.2f}"'
               if j == accent else '')
        paths.append(f'<path{cls} d="{ridge_d(xs, prof, base)}"/>')
    body = '\n'.join(paths)
    return f'''<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 {W} {H}" width="{W}" height="{H}">
<defs>
<clipPath id="clip"><rect width="{W}" height="{H}" rx="16"/></clipPath>
<linearGradient id="haze" x1="0" y1="0" x2="0" y2="1">
<stop offset="0" stop-color="{th['haze']}" stop-opacity="1"/>
<stop offset=".35" stop-color="{th['haze']}" stop-opacity=".8"/>
<stop offset="1" stop-color="{th['haze']}" stop-opacity="0"/>
</linearGradient>
<linearGradient id="vig">
{vignette_stops(th)}
</linearGradient>
<linearGradient id="acc" gradientUnits="userSpaceOnUse" x1="0" y1="0" x2="{W}" y2="0">
{accent_stops(th)}
</linearGradient>
</defs>
<rect width="{W}" height="{H}" rx="16" fill="{th['bg']}"/>
<g clip-path="url(#clip)">
<g>
<animateTransform attributeName="transform" type="translate" from="0 0" to="0 {period:g}" dur="{DUR:g}s" repeatCount="indefinite"/>
<g id="r" fill="{th['bg']}" stroke="{th['line']}" stroke-opacity="{th['op_hi']}" stroke-width="{SW}" stroke-linejoin="round">
{body}
</g>
<use xlink:href="#r" href="#r" y="{period:g}"/>
</g>
<rect width="{W}" height="{H}" fill="url(#vig)"/>
<rect width="{W}" height="{TOP_FADE}" fill="url(#haze)"/>
</g>
<rect x=".5" y=".5" width="{W - 1}" height="{H - 1}" rx="15.5" fill="none" stroke="{th['border']}"/>
</svg>
'''


def main():
    xs, profiles = build_profiles(SEED)
    dips = min(p.min() for p in profiles)
    assert -dips < GAP - 2, dips
    print('deepest dip below baseline: %.1f px' % -dips)
    for theme in THEMES:
        svg = build_svg(theme, xs, profiles)
        path = os.path.join(HERE, f'banner-{theme}.svg')
        with open(path, 'w') as f:
            f.write(svg)
        print(path, len(svg.encode()) // 1024, 'KB')


if __name__ == '__main__':
    main()
