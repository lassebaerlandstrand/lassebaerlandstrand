"""Geometry: smooth cubic strands through the crossings, split into pieces."""
import math

from knot import direction


def _turn(d0, d1, sign):
    a0 = math.atan2(d0[1], d0[0])
    a1 = math.atan2(d1[1], d1[0])
    t = (a1 - a0) * sign
    t %= 2 * math.pi
    if t > 2 * math.pi - 1e-6:
        t = 0.0
    return t


def _ratio(theta):
    if theta < 1e-4:
        return 1 / 3
    return (4 / 3) * math.tan(theta / 4) / (2 * math.sin(theta / 2))


def bez_point(P, t):
    u = 1 - t
    return tuple(u ** 3 * P[0][i] + 3 * u * u * t * P[1][i] + 3 * u * t * t * P[2][i] + t ** 3 * P[3][i]
                 for i in range(2))


def bez_split(P, t):
    lerp = lambda p, q: (p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t)
    p01, p12, p23 = lerp(P[0], P[1]), lerp(P[1], P[2]), lerp(P[2], P[3])
    p012, p123 = lerp(p01, p12), lerp(p12, p23)
    m = lerp(p012, p123)
    return (P[0], p01, p012, m), (m, p123, p23, P[3])


def bez_sub(P, t0, t1):
    _, right = bez_split(P, t0)
    if t1 >= 1:
        return right
    left, _ = bez_split(right, (t1 - t0) / (1 - t0))
    return left


def bez_len(P, n=48):
    L = 0.0
    prev = P[0]
    for k in range(1, n + 1):
        q = bez_point(P, k / n)
        L += math.hypot(q[0] - prev[0], q[1] - prev[1])
        prev = q
    return L


def strand_curves(strand, xf, params):
    """Cubic segments (list of 4-point tuples, already in output space) for one strand.

    xf: affine map (x, y) -> (X, Y) applied to lattice space.
    params: dict with 'k_straight', 'k_bounce', 'k_corner' multipliers.
    """
    n = len(strand)
    segs = []
    for k in range(n):
        a, b, s = strand[k]
        a2, b2, s2 = strand[(k + 1) % n]
        p0 = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
        p3 = ((a2[0] + b2[0]) / 2, (a2[1] + b2[1]) / 2)
        d0 = direction(a, b, s)
        d1 = direction(a2, b2, s2)
        sign = -1 if s == 'L' else 1
        th = _turn(d0, d1, sign)
        chord = math.hypot(p3[0] - p0[0], p3[1] - p0[1])
        if th < 1e-4:
            h = chord * params['k_straight']
        elif th < math.pi * 0.75:
            h = chord * _ratio(th) * params['k_bounce']
        else:
            h = chord * _ratio(th) * params['k_corner']
        P = (p0, (p0[0] + d0[0] * h, p0[1] + d0[1] * h),
             (p3[0] - d1[0] * h, p3[1] - d1[1] * h), p3)
        segs.append(tuple(xf(*p) for p in P))
    return segs


def pieces(segs, eps=0.04):
    """Split a closed chain of segments into pieces, one per crossing.

    Piece k = second half of segment k-1 + first half of segment k (crossing k
    sits at the joint).  Returns list of (cubics, start_arclength, length)."""
    n = len(segs)
    lens = [bez_len(P) for P in segs]
    total = sum(lens)
    cum = [0.0]
    for L in lens:
        cum.append(cum[-1] + L)
    out = []
    for k in range(n):
        prev = segs[(k - 1) % n]
        Lp = lens[(k - 1) % n]
        c1 = bez_sub(prev, 0.5 - eps, 1.0)
        c2 = bez_sub(segs[k], 0.0, 0.5 + eps)
        l1, l2 = bez_len(c1), bez_len(c2)
        start = cum[(k - 1) % n] + (Lp - l1) if k > 0 else cum[n - 1] + (Lp - l1) - total
        out.append(([c1, c2], start, l1 + l2))
    return out, total


def fmt(v):
    s = f'{v:.1f}'
    if s.endswith('.0'):
        s = s[:-2]
    if s == '-0':
        s = '0'
    return s


def path_d(cubics):
    P = cubics[0]
    d = f'M{fmt(P[0][0])} {fmt(P[0][1])}'
    for P in cubics:
        d += 'C' + ' '.join(f'{fmt(p[0])} {fmt(p[1])}' for p in P[1:])
    return d


def _deriv(P, t):
    u = 1 - t
    return tuple(3 * u * u * (P[1][i] - P[0][i]) + 6 * u * t * (P[2][i] - P[1][i]) + 3 * t * t * (P[3][i] - P[2][i])
                 for i in range(2))


def _deriv2(P, t):
    u = 1 - t
    return tuple(6 * u * (P[2][i] - 2 * P[1][i] + P[0][i]) + 6 * t * (P[3][i] - 2 * P[2][i] + P[1][i])
                 for i in range(2))


def _normal_curv(P, t):
    d = _deriv(P, t)
    dd = _deriv2(P, t)
    sp = math.hypot(*d)
    n = (-d[1] / sp, d[0] / sp)
    k = (d[0] * dd[1] - d[1] * dd[0]) / sp ** 3
    return n, k


def offset_cubic(P, dist, split=2):
    """Approximate offset of a cubic by `dist` along its left normal."""
    out = []
    parts = [P]
    for _ in range(split - 1):
        nxt = []
        for Q in parts:
            nxt.extend(bez_split(Q, 0.5))
        parts = nxt
    for Q in parts:
        n0, k0 = _normal_curv(Q, 0.0)
        n1, k1 = _normal_curv(Q, 1.0)
        A = (Q[0][0] + n0[0] * dist, Q[0][1] + n0[1] * dist)
        D = (Q[3][0] + n1[0] * dist, Q[3][1] + n1[1] * dist)
        s0 = 1 - dist * k0
        s1 = 1 - dist * k1
        B = (A[0] + (Q[1][0] - Q[0][0]) * s0, A[1] + (Q[1][1] - Q[0][1]) * s0)
        C = (D[0] + (Q[2][0] - Q[3][0]) * s1, D[1] + (Q[2][1] - Q[3][1]) * s1)
        out.append((A, B, C, D))
    return out
