# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

# The A5 space-filling curve, a turtle L-system on the triangular lattice.
#
# This replaces the earlier `shift_digits` Hilbert construction. shift_digits was
# an approximation of this curve: they agree exactly through resolution 4, but
# shift_digits self-intersects from resolution 5 on, whereas this curve never
# crosses itself at any resolution while tiling the exact same cells with the same
# metacell hierarchy. (The old curve remains available bit-for-bit via compat.py,
# which runs the original two-motif grammar + the shift_digits digit recode through
# the same descents below.)
#
# The curve is a vertex-to-vertex turtle L-system on the triangular lattice: 7
# self-referential motifs (A B C M P Q R), each a clean A5 unit (2 parallelograms
# + 2 triangles). The symbolic grammar lives in grammar.py and is compiled to flat
# tables in tables.py; this module evaluates it as an O(resolution) digit
# transducer:
#   forward  s -> cell   : descend the quaternary digits, accumulating a turtle
#            position + heading, then map (a,b) -> A5 triple via a fixed
#            similarity; the leaf state also yields the cell's pentagon flavor.
#   inverse  triple -> s : descend picking, at each level, the child whose convex
#            footprint (triforce / parallelogram) contains the target cell.
#
# Every turn in every rule is 180 deg (see tables.py), so the descent tracks
# orientation as a single flip bit; for the A5 grammar that invariant is also
# what keeps every parallelogram cell on-axis.

import math
from typing import NamedTuple, Tuple

from ..types import Orientation, Triple
from .grammar import RULES, DRAWS
from .tables import compile_grammar, CurveTables, POW2, POW4
from .turtle import AB

# The compiled A5 grammar.
A5 = compile_grammar(RULES, DRAWS)


class Cell(NamedTuple):
    """A cell as the descent identifies it: its triple + its pentagon flavor."""
    triple: Triple
    flavor: int


class LeafCell(NamedTuple):
    """Result of the forward leaf descent: host cell corner sum + pentagon flavor."""
    a: float
    b: float
    flavor: int


# ---------- exact (a,b) corner-sum <-> A5 triple ----------
# The turtle (a,b) lattice and A5's triple frame are two views of the same
# triangular grid. Composing them, the sqrt3 from each basis cancels, leaving an
# exact rational map: from a cell's corner sum (= 3*centroid),
#   y - z      = (2*sum.a + sum.b - 12) / 12
#   2x - y - z = (sum.b + 4) / 4
# and the parity x+y+z in {0,1} pins x, y, z. No floating point.
def ab_to_triple(sum_a: float, sum_b: float) -> Triple:
    sa = int(round(sum_a))
    sb = int(round(sum_b))
    if (2 * sa + sb) % 12 != 0 or sb % 4 != 0:
        raise ValueError(f'ab_to_triple: off-lattice corner sum ({sum_a},{sum_b})')
    yz = (2 * sa + sb - 12) // 12  # y - z
    e = (sb + 4) // 4  # 2x - y - z
    for parity in (0, 1):
        if (e + parity) % 3 != 0:
            continue
        x = (e + parity) // 3
        r = parity - x  # = y + z
        if (r + yz) % 2 != 0:
            continue
        return Triple(x, (r + yz) // 2, (r - yz) // 2)
    raise ValueError(f'ab_to_triple: no integer triple for ({sum_a},{sum_b})')


def triple_to_ab(t: Triple) -> Tuple[float, float]:
    x, y, z = t
    b = 4 * (2 * x - y - z) - 4
    a = (12 * (y - z) + 12 - b) // 2
    return float(a), float(b)


# ---------- forward: s -> leaf host cell (corner sum + flavor) ----------
# A child placed at (parent-relative) off_unit under a `flip` frame has its
# offset negated when flipped (180 deg); the child's own frame is
# `flip XOR child.flip`. Internal; also used by compat.py.
def axiom_leaf_cell(t: CurveTables, s: int, R: int, axiom: int) -> LeafCell:
    motif = axiom
    flip = 0
    pos_a = 0.0
    pos_b = 0.0
    for level in range(R, 1, -1):
        idx = level - 1
        d = (s >> (idx * 2)) & 3
        ci = motif * 4 + d
        scale = -POW2[level - 2] if flip else POW2[level - 2]
        pos_a += t.child_off_a[ci] * scale
        pos_b += t.child_off_b[ci] * scale
        flip ^= t.child_flip[ci]
        motif = t.child_token[ci]
    # level 1: leaf walk (from heading 0 or 3), take the d0-th host cell
    d0 = (s & 3) if R >= 1 else 0
    base = motif * 2 + flip
    return LeafCell(
        a=3.0 * pos_a + t.leaf_sum[base * 8 + d0 * 2],
        b=3.0 * pos_b + t.leaf_sum[base * 8 + d0 * 2 + 1],
        flavor=t.leaf_flavor[base * 4 + d0],
    )


# ---------- inverse: descend by which child's convex footprint contains the target ----------
def _inside_score(t, motif, flip, lvl, pos_a, pos_b, ta, tb, best):
    scale = POW2[lvl - 1]
    edges = t.fp_edges[motif * 2 + flip]
    min_cross = math.inf
    e = 0
    while e < len(edges):
        dta = ta - (3.0 * pos_a + edges[e] * scale)
        dtb = tb - (3.0 * pos_b + edges[e + 1] * scale)
        cross = edges[e + 2] * dtb - edges[e + 3] * dta
        if cross < min_cross:
            min_cross = cross
            if min_cross <= 0.0 and min_cross <= best:
                return min_cross
        e += 4
    return min_cross


# Shared descent for both leaf modes. `exact` targets are corner sums of real
# cells (leaf resolved by exact sum match); fractional targets resolve the leaf
# by point-in-cell over the 4 level-1 triangles. Internal; also used by compat.py.
def axiom_target_to_s(t: CurveTables, ta: float, tb: float, R: int, axiom: int, exact: bool) -> int:
    motif = axiom
    flip = 0
    pos_a = 0.0
    pos_b = 0.0
    s_val = 0
    for level in range(R, 1, -1):
        scale = POW2[level - 2]
        sign = -scale if flip else scale
        best_d = 0
        best_score = -math.inf
        for d in range(4):
            ci = motif * 4 + d
            score = _inside_score(
                t, t.child_token[ci], flip ^ t.child_flip[ci], level - 1,
                pos_a + t.child_off_a[ci] * sign, pos_b + t.child_off_b[ci] * sign,
                ta, tb, best_score,
            )
            if score > best_score:
                best_score = score
                best_d = d
                if score > 0.0:
                    break  # strictly inside: the unique containing child
        ci = motif * 4 + best_d
        pos_a += t.child_off_a[ci] * sign
        pos_b += t.child_off_b[ci] * sign
        flip ^= t.child_flip[ci]
        motif = t.child_token[ci]
        idx = level - 1
        s_val += best_d << (2 * idx)
    # level 1: pick the leaf cell, by exact corner-sum match or point-in-cell
    base = motif * 2 + flip
    d0 = 0
    if exact:
        rel_a = ta - 3.0 * pos_a
        rel_b = tb - 3.0 * pos_b
        found = False
        for d in range(4):
            if t.leaf_sum[base * 8 + d * 2] == rel_a and t.leaf_sum[base * 8 + d * 2 + 1] == rel_b:
                d0 = d
                found = True
                break
        if not found:
            raise ValueError(f'lsystem inverse: no leaf match for corner sum ({ta},{tb})')
    else:
        best_score = -math.inf
        for d in range(4):
            min_cross = math.inf
            for e in range(3):
                o = base * 48 + d * 12 + e * 4
                dta = ta - (3.0 * pos_a + t.leaf_tri[o])
                dtb = tb - (3.0 * pos_b + t.leaf_tri[o + 1])
                cross = t.leaf_tri[o + 2] * dtb - t.leaf_tri[o + 3] * dta
                if cross < min_cross:
                    min_cross = cross
            if min_cross > best_score:
                best_score = min_cross
                d0 = d
                if min_cross > 0.0:
                    break
    return s_val + d0


# ---------- orientation = which triforce motif is the axiom ----------
# Each orientation is one of the three triforce motifs used as the axiom
# (uv->A, uw->C, wv->B), with the reverse orientations (vu, wu, vw) walking the
# same curve backward (s -> N-1-s).
_ORIENT = {
    'uv': ('A', False, False),
    'vu': ('A', True, False),
    'uw': ('C', False, False),
    'wu': ('C', True, False),
    'vw': ('B', True, True),
    'wv': ('B', False, True),
}


def s_to_cell(s: int, resolution: int, orientation: Orientation = 'uv') -> Cell:
    """
    The A5 curve position `s` -> cell (triple coordinate + pentagon flavor), for
    a given resolution and orientation. The triple is bijective with
    `triple_to_s_lattice`.
    """
    n = 1 << (2 * resolution)
    axiom_char, reverse, is_b = _ORIENT[orientation]
    axiom = A5.motif_idx[axiom_char]
    s_axiom = (n - 1 - s) if reverse else s
    cell = axiom_leaf_cell(A5, s_axiom, resolution, axiom)
    base = ab_to_triple(cell.a, cell.b)
    if not is_b:
        return Cell(triple=base, flavor=cell.flavor)
    p = int(POW2[resolution])
    return Cell(triple=Triple(base.x - p, base.y + p, base.z), flavor=cell.flavor)


def s_to_triple(s: int, resolution: int, orientation: Orientation = 'uv') -> Triple:
    """The A5 curve position `s` -> triple coordinate. Bijective with `triple_to_s_lattice`."""
    return s_to_cell(s, resolution, orientation).triple


def triple_to_s_lattice(triple: Triple, resolution: int, orientation: Orientation = 'uv') -> int:
    """Triple coordinate -> the A5 curve position `s`. Inverse of `s_to_triple`."""
    n = 1 << (2 * resolution)
    axiom_char, reverse, is_b = _ORIENT[orientation]
    axiom = A5.motif_idx[axiom_char]
    ab_a, ab_b = triple_to_ab(triple)
    tau_sum = 12.0 * POW2[resolution] if is_b else 0.0
    s_axiom = axiom_target_to_s(A5, ab_a - tau_sum, ab_b + tau_sum, resolution, axiom, True)
    return (n - 1 - s_axiom) if reverse else s_axiom


def sum_point_to_s(ta: float, tb: float, resolution: int, orientation: Orientation = 'uv') -> int:
    """
    Fractional point -> the curve position `s` of the containing cell, by direct
    descent. The target is given in the corner-sum frame (= 3x the L-system (a,b)
    point frame); callers map their coordinate system into it (for the IJ plane
    the exact affine map is target = (12*(i+j), -12*j), see curve.py).
    """
    n = 1 << (2 * resolution)
    axiom_char, reverse, is_b = _ORIENT[orientation]
    axiom = A5.motif_idx[axiom_char]
    tau_sum = 12.0 * POW2[resolution] if is_b else 0.0
    s_axiom = axiom_target_to_s(A5, ta - tau_sum, tb + tau_sum, resolution, axiom, False)
    return (n - 1 - s_axiom) if reverse else s_axiom
