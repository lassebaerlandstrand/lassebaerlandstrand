"""Knot construction: an interlace as the medial graph of a planar grid graph.

The base graph G is a rectangular lattice of points with some edges removed
("breaks").  Every remaining edge of G becomes one crossing of the knot (at the
edge midpoint); strands run straight through a crossing and then turn around
the vertex of G they are heading for, to the next edge of the same face.  This
is the classical construction of plaitwork with breaks, and it gives an
alternating interlace whose over/under labelling is always consistent.
"""
import math


def build_graph(cols, rows, removed, mask=None):
    """Lattice cols x rows, minus edges in `removed` (set of frozenset pairs)."""
    nbrs = {}
    for i in range(cols):
        for j in range(rows):
            if mask is None or mask(i, j):
                nbrs[(i, j)] = []
    for i in range(cols):
        for j in range(rows):
            for di, dj in ((1, 0), (0, 1)):
                a, b = (i, j), (i + di, j + dj)
                if a in nbrs and b in nbrs and frozenset((a, b)) not in removed:
                    nbrs[a].append(b)
                    nbrs[b].append(a)
    # angular order (y up)
    for v, ns in nbrs.items():
        ns.sort(key=lambda w: math.atan2(w[1] - v[1], w[0] - v[0]))
    nbrs = {v: ns for v, ns in nbrs.items() if ns}
    bad = [v for v, ns in nbrs.items() if len(ns) == 1]
    assert not bad, f'dangling vertices {bad}'
    return nbrs


def _next_around(nbrs, b, a, cw):
    ns = nbrs[b]
    k = ns.index(a)
    return ns[(k - 1) % len(ns)] if cw else ns[(k + 1) % len(ns)]


def trace(nbrs):
    """Return list of strands; each strand is a list of states (a, b, side).

    State (a, b, s): standing on the midpoint of edge {a,b}, leaving towards
    vertex b on side s ('L' or 'R') of the directed edge a->b.
    """
    seen = set()
    strands = []
    states = []
    for a, ns in nbrs.items():
        for b in ns:
            for s in 'LR':
                states.append((a, b, s))
    for st in sorted(states):
        if st in seen:
            continue
        strand = []
        cur = st
        while cur not in seen:
            seen.add(cur)
            a, b, s = cur
            seen.add((b, a, s))          # same strand, opposite direction
            strand.append(cur)
            c = _next_around(nbrs, b, a, cw=(s == 'L'))
            cur = (b, c, 'R' if s == 'L' else 'L')
        strands.append(strand)
    return strands


def direction(a, b, s):
    tx, ty = b[0] - a[0], b[1] - a[1]
    n = math.hypot(tx, ty)
    tx, ty = tx / n, ty / n
    nx, ny = (-ty, tx) if s == 'L' else (ty, -tx)
    dx, dy = tx + nx, ty + ny
    m = math.hypot(dx, dy)
    return dx / m, dy / m


def over_under(strands):
    """Alternate over/under along every strand, consistently at each crossing.

    Returns per strand a list of booleans (True = this passage is 'over')."""
    # crossing id = frozenset edge
    label = {}   # (edge, strand_idx, pos) -> bool
    owner = {}   # edge -> list of (strand_idx, pos)
    for si, st in enumerate(strands):
        for k, (a, b, s) in enumerate(st):
            owner.setdefault(frozenset((a, b)), []).append((si, k))
    res = [None] * len(strands)
    # propagate parity via BFS over strands
    parity = {}
    order = [0]
    parity[0] = 0
    done = set()
    while order:
        si = order.pop()
        if si in done:
            continue
        done.add(si)
        st = strands[si]
        res[si] = [((k + parity[si]) % 2 == 0) for k in range(len(st))]
        for k, (a, b, s) in enumerate(st):
            for sj, kj in owner[frozenset((a, b))]:
                if (sj, kj) == (si, k):
                    continue
                want_over = not res[si][k]
                pj = 0 if ((kj % 2 == 0) == want_over) else 1
                if sj in parity:
                    assert parity[sj] == pj, 'inconsistent alternation'
                else:
                    parity[sj] = pj
                    order.append(sj)
        if not order:
            for sj in range(len(strands)):
                if sj not in parity:
                    parity[sj] = 0
                    order.append(sj)
                    break
    return res
