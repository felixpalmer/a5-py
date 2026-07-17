# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

# Compiles an L-system grammar into flat numeric tables, once at module init,
# so the descents in the lsystem package are pure scalar arithmetic: no string
# expansion and no object allocation per call.
#
# Every grammar compiled here keeps every turn inside a rule at 180 deg
# (`+++`/`---`), so a child is only ever placed un-flipped or flipped (rotated
# 180 deg) relative to its parent -- never at 60 deg. compile_grammar enforces
# that invariant and records the orientation as a single `flip` bit (180 deg =
# negate on this lattice); the whole descent then tracks orientation as one bool.
# For the A5 grammar the invariant is also what keeps every parallelogram cell
# on-axis; the compat W/Z grammar is the original two-motif curve re-gauged
# into this form (see compat.py).
#
# Motif tokens are indexed by their position in the motif list (uppercase
# motifs first, then their lowercase reverses); a descent state is
# (motif index, flip bit). All hot-path lookups are flat list reads indexed
# by that state.

from typing import Dict, List, NamedTuple

from .grammar import reverse_motif, expand_once
from .turtle import AB, host_corners, host_sum, net_of, walk


class CurveTables(NamedTuple):
    """Flat numeric tables for one grammar, consumed by the descents in the lsystem package."""
    motif_idx: Dict[str, int]
    # children: entry ci = motif * 4 + digit
    child_token: List[int]
    child_flip: List[int]
    child_off_a: List[float]
    child_off_b: List[float]
    # footprint hulls per (motif, flip): edge list [3*c0.a, 3*c0.b, d.a, d.b]*E
    fp_edges: List[List[float]]
    # leaf tables per (motif, flip): 4 host cells as corner sums, point-in-cell
    # triangle edges, and pentagon flavors
    leaf_sum: List[float]
    leaf_flavor: List[int]
    # Branchless child classifier per state k = motif*2+flip: 3 separating lines
    # (class_sep[k*9 ..] = [nx0,ny0,c0, nx1,ny1,c1, nx2,ny2,c2]) evaluated against
    # the normalised target give a 3-bit pattern; class_lut[k*8 + pat] is the
    # child digit. Replaces the exact-path 4-hull scan with 3 dot products + a LUT.
    class_sep: List[float]
    class_lut: List[int]


BSP_EPS = 1e-6

# The pentagon FLAVOR (0-3) of the cell a draw symbol hosts: which of the four
# pentagon orientations of the Cairo-like metatile it gets. The pentagon is a
# 1:1 function of the cell's jigsaw piece and reduces to the closed-form rule
#   flavor = BASE[symbol] XOR isLowercase XOR (heading & 1)
# with BASE = {S:0, D:1, E:2, T:3}; bit 0 is a 180 deg rotation, bit 1 a Y
# reflection of the base pentagon (see core/tiling.py). Derived and verified
# exhaustively against the pentagon geometry.
_FLAVOR_BASE = {'S': 0, 'D': 1, 'E': 2, 'T': 3}


class _Child(NamedTuple):
    token: str
    off_unit: AB  # offset from the parent origin, in net(.,1) units
    flip: bool


def _to_draws(motif: str, level: int, rules: Dict[str, str], draws: Dict[str, str]) -> str:
    """
    Expand a motif to a pure draw string: `level` rule passes, then one draws
    pass (turning every remaining motif into its leaf terminal).
    """
    s = motif
    for _ in range(level):
        s = expand_once(s, rules)
    return expand_once(s, draws)


def _motif_net(motif: str, rules: Dict[str, str], draws: Dict[str, str]) -> AB:
    return net_of(_to_draws(motif, 1, rules, draws))[0]


def _child_table(rule: str, rules: Dict[str, str], draws: Dict[str, str]) -> List[_Child]:
    pos = AB(0, 0)
    h = 0
    children: List[_Child] = []
    for ch in rule:
        if ch == '+':
            h = (h + 1) % 6
            continue
        if ch == '-':
            h = (h + 5) % 6
            continue
        if ch.upper() not in rules:
            continue
        if h != 0 and h != 3:
            raise ValueError(f'lsystem: non-180 deg turn ({60 * h} deg) before a child in rule "{rule}"')
        flip = h == 3
        children.append(_Child(token=ch, off_unit=pos, flip=flip))
        n = _motif_net(ch, rules, draws)
        pos = AB(pos.a - n.a, pos.b - n.b) if flip else AB(pos.a + n.a, pos.b + n.b)
    if len(children) != 4:
        raise ValueError(f'lsystem: rule "{rule}" must have 4 children')
    return children


def _convex_hull(pts: List[AB]) -> List[AB]:
    p = sorted(set(pts), key=lambda q: (q.a, q.b))
    if len(p) < 3:
        return p

    def cross(o: AB, a: AB, b: AB) -> int:
        return (a.a - o.a) * (b.b - o.b) - (a.b - o.b) * (b.a - o.a)

    lower: List[AB] = []
    for q in p:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], q) <= 0:
            lower.pop()
        lower.append(q)
    upper: List[AB] = []
    for q in reversed(p):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], q) <= 0:
            upper.pop()
        upper.append(q)
    return lower[:-1] + upper[:-1]


def compile_grammar(rules: Dict[str, str], draws: Dict[str, str]) -> CurveTables:
    """
    Compile a grammar (motif rules + leaf draws) into flat descent tables.
    Lowercase motifs are the uppercase rules reversed, generated automatically.
    """
    # Deterministic motif order (indices are internal, so a stable sort suffices).
    motifs = sorted(rules.keys())
    all_motifs = motifs + [m.lower() for m in motifs]
    motif_count = len(all_motifs)
    motif_idx = {m: i for i, m in enumerate(all_motifs)}

    # ---------- child tables: 4 children per motif ----------
    children_of: Dict[str, List[_Child]] = {}
    for m in motifs:
        children_of[m] = _child_table(rules[m], rules, draws)
    for m in motifs:
        children_of[m.lower()] = _child_table(reverse_motif(rules[m]), rules, draws)

    child_token = [0] * (motif_count * 4)
    child_flip = [0] * (motif_count * 4)
    child_off_a = [0.0] * (motif_count * 4)
    child_off_b = [0.0] * (motif_count * 4)
    for m in all_motifs:
        cs = children_of[m]
        for d in range(4):
            ci = motif_idx[m] * 4 + d
            child_token[ci] = motif_idx[cs[d].token]
            child_flip[ci] = 1 if cs[d].flip else 0
            child_off_a[ci] = float(cs[d].off_unit.a)
            child_off_b[ci] = float(cs[d].off_unit.b)

    # ---------- footprint hulls (convex hull of leaf host corners) ----------
    # per (motif, flip): edge list [3*c0.a, 3*c0.b, d.a, d.b]*E.
    # The corner is pre-tripled (the descent works in the corner-sum frame, = 3x
    # the (a,b) point frame); the edge direction stays UNIT so the containment
    # cross products stay ~O(2^R) instead of O(2^2R) -- exact integer at every
    # resolution. The flipped variant is the hull negated (180 deg = negate,
    # winding-preserving).
    fp_edges: List[List[float]] = [[] for _ in range(motif_count * 2)]
    for m in all_motifs:
        corners: List[AB] = []
        walk(
            _to_draws(m, 1, rules, draws), AB(0, 0), 0,
            lambda sym, frm, h: corners.extend(host_corners(sym, frm, h)),
        )
        hull = _convex_hull(corners)
        for flip in range(2):
            sign = -1.0 if flip == 1 else 1.0
            edges = [0.0] * (len(hull) * 4)
            for i in range(len(hull)):
                c0 = hull[i]
                c1 = hull[(i + 1) % len(hull)]
                edges[i * 4] = 3.0 * sign * c0.a
                edges[i * 4 + 1] = 3.0 * sign * c0.b
                edges[i * 4 + 2] = sign * (c1.a - c0.a)
                edges[i * 4 + 3] = sign * (c1.b - c0.b)
            fp_edges[motif_idx[m] * 2 + flip] = edges

    # ---------- leaf tables: per (motif, flip = heading 0|3) the 4 level-1 host cells ----------
    leaf_sum = [0.0] * (motif_count * 2 * 8)
    leaf_flavor = [0] * (motif_count * 2 * 4)
    for m in all_motifs:
        draw_str = _to_draws(m, 1, rules, draws)
        for flip in range(2):
            base = motif_idx[m] * 2 + flip
            counter = [0]  # boxed so the closure can mutate it

            def on_draw(sym: str, frm: AB, hh: int, base=base) -> None:
                d = counter[0]
                sm = host_sum(sym, frm, hh)
                leaf_sum[base * 8 + d * 2] = float(sm.a)
                leaf_sum[base * 8 + d * 2 + 1] = float(sm.b)
                upper = sym.upper()
                if upper not in _FLAVOR_BASE:
                    raise ValueError(f'lsystem: no pentagon flavor for draw symbol {sym}')
                is_lower = 0 if sym == upper else 1
                leaf_flavor[base * 4 + d] = _FLAVOR_BASE[upper] ^ is_lower ^ (hh & 1)
                counter[0] += 1

            walk(draw_str, AB(0, 0), 3 if flip == 1 else 0, on_draw)

    # ---------- branchless child classifier (3 line tests + LUT per state) ----------
    class_sep = [0.0] * (motif_count * 2 * 9)
    class_lut = [0] * (motif_count * 2 * 8)
    for m in range(motif_count):
        for f in range(2):
            k = m * 2 + f
            tree = _build_bsp(_child_polys(m, f, child_token, child_flip, child_off_a, child_off_b, fp_edges))
            seps: List[tuple] = []
            _collect_seps(tree, seps)
            for i, s in enumerate(seps):
                class_sep[k * 9 + i * 3] = s[0]
                class_sep[k * 9 + i * 3 + 1] = s[1]
                class_sep[k * 9 + i * 3 + 2] = s[2]
            for p in range(8):
                class_lut[k * 8 + p] = _walk_bsp(tree, p, seps)

    return CurveTables(
        motif_idx=motif_idx,
        child_token=child_token,
        child_flip=child_flip,
        child_off_a=child_off_a,
        child_off_b=child_off_b,
        fp_edges=fp_edges,
        leaf_sum=leaf_sum,
        leaf_flavor=leaf_flavor,
        class_sep=class_sep,
        class_lut=class_lut,
    )


def _child_polys(motif, pflip, child_token, child_flip, child_off_a, child_off_b, fp_edges):
    """The 4 child footprint polygons for state (motif, pflip), in the normalised
    target-relative-to-cursor frame (scale-invariant), as (digit, verts)."""
    psign = -1 if pflip else 1
    out = []
    for d in range(4):
        ci = motif * 4 + d
        tok = child_token[ci]
        cfl = child_flip[ci]
        oa = child_off_a[ci]
        ob = child_off_b[ci]
        edges = fp_edges[tok * 2 + (pflip ^ cfl)]
        verts = [(3 * oa * psign + edges[e], 3 * ob * psign + edges[e + 1])
                 for e in range(0, len(edges), 4)]
        out.append((d, verts))
    return out


def _build_bsp(children):
    """Build a child-selection BSP; children tile convexly, so a polygon edge
    cleanly splits them into two groups. Returns ('leaf', digit) | ('node', (nx,ny,c), pos, neg)."""
    if len(children) == 1:
        return ('leaf', children[0][0])
    for _, poly in children:
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            nx = y2 - y1
            ny = -(x2 - x1)
            c = -(nx * x1 + ny * y1)
            pos, neg, ok = [], [], True
            for d, cp in children:
                vals = [nx * x + ny * y + c for x, y in cp]
                if min(vals) >= -BSP_EPS:
                    pos.append((d, cp))
                elif max(vals) <= BSP_EPS:
                    neg.append((d, cp))
                else:
                    ok = False
                    break
            if ok and pos and neg:
                return ('node', (nx, ny, c), _build_bsp(pos), _build_bsp(neg))
    raise ValueError("lsystem: no clean BSP split for child set")


def _collect_seps(tree, seps):
    if tree[0] == 'leaf':
        return
    if tree[1] not in seps:
        seps.append(tree[1])
    _collect_seps(tree[2], seps)
    _collect_seps(tree[3], seps)


def _walk_bsp(tree, p, seps):
    """Walk the tree for a fixed sign pattern p (bit i = which side of separator i)."""
    if tree[0] == 'leaf':
        return tree[1]
    idx = seps.index(tree[1])
    return _walk_bsp(tree[2] if (p >> idx) & 1 else tree[3], p, seps)


# powers of 2 used by the descents (index by level), the child-offset scale
POW2 = [2.0 ** i for i in range(32)]
