# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

# The ORIGINAL A5 curve (the shift_digits construction), expressed on top of the
# L-system machinery -- bit-for-bit compatible with the pre-L-system library.
#
# The old construction is two layers, and both are preserved here exactly:
#
# 1. The base ordering is a simple two-motif L-system (the "original quaternary
#    curve" the A5 curve grew out of), verified index-for-index equal to the
#    old raw descent. Its native form uses +/-60 deg turns
#    (X: X-Y+X++Y--  Y: Y+X-Y--X++, draws X -> E, Y -> -e+), but it re-gauges
#    into the 180-deg-only form the table compiler requires -- the +/-60 deg is
#    absorbed into Z's leaf gauge (a walk-identical re-gauging):
#      W: W+++Z---WZ   Z: Z+++W---ZW    (draws W -> E, Z -> +e-)
# 2. On top of it, the shift_digits digit recode (ported verbatim below), which
#    rearranges children so they overlap their parent cells -- the "hierarchy
#    fix" that introduced the self-intersections the new curve removes.
#
# Orientations follow the old engine exactly: reverse remaps s -> N-1-s,
# flipIJ (uw/wu) selects the flipped pattern and mirrors the raw cell
# (x <-> z in triple space), invertJ (vw/wv) flips the quintant vertically
# ((x,y,z) -> (y-(n-1), x+(n-1), z), n = 2^res). Both maps are self-inverse.

from typing import List, Optional

from ..core.coordinate_systems import IJ
from .lsystem.tables import compile_grammar, POW2
from .lsystem import (
    Cell, a5_triple_to_flavor, ab_to_triple, axiom_leaf_cell, axiom_target_to_s,
    triple_to_ab,
)
from .types import Orientation, Triple

# The compiled two-motif grammar of the original curve (W/Z gauge).
ORIGINAL = compile_grammar({'W': 'W+++Z---WZ', 'Z': 'Z+++W---ZW'}, {'W': 'E', 'Z': '+e-'})
_AXIOM_W = ORIGINAL.motif_idx['W']


# ---------- shift_digits (ported verbatim from the original construction) ----------
# Patterns used to rearrange the cells when shifting. This adjusts the layout
# so that children always overlap with their parent cells.
def _reverse_pattern(pattern: List[int]) -> List[int]:
    return [pattern.index(i) for i in range(len(pattern))]


PATTERN = [0, 1, 3, 4, 5, 6, 7, 2]
PATTERN_FLIPPED = [0, 1, 2, 7, 3, 4, 5, 6]
PATTERN_REVERSED = _reverse_pattern(PATTERN)
PATTERN_FLIPPED_REVERSED = _reverse_pattern(PATTERN_FLIPPED)


def _shift_digits(digits: List[int], i: int, flips: List[int], invert_j: bool, pattern: List[int]) -> None:
    if i <= 0:
        return

    parent_k = digits[i]
    child_k = digits[i - 1]
    F = flips[0] + flips[1]

    # Detect when cells need to be shifted. The rule is flipped depending on the
    # orientation, specifically on the value of invert_j.
    if invert_j != (F == 0):
        needs_shift = parent_k == 1 or parent_k == 2  # Second & third pentagons only
        first = parent_k == 1  # Second pentagon is first
    else:
        needs_shift = parent_k < 2  # First two pentagons only
        first = parent_k == 0  # First pentagon is first
    if not needs_shift:
        return

    # Apply the pattern by setting the digits based on the value provided
    src = child_k if first else child_k + 4
    dst = pattern[src]
    digits[i - 1] = dst % 4
    digits[i] = (parent_k + 4 + dst // 4 - src // 4) % 4


# the flips product accumulates per digit exactly as quaternary_to_flips did:
# digit 1 flips the second component, digit 3 the first
def _apply_digit_flips(flips: List[int], d: int) -> None:
    if d == 1:
        flips[1] = -flips[1]
    elif d == 3:
        flips[0] = -flips[0]


def _forward_shift(digits: List[int], invert_j: bool, flip_ij: bool) -> None:
    """old s digits -> geometric (X/Y curve) digits, in place. LSB-first list."""
    pattern = PATTERN_FLIPPED if flip_ij else PATTERN
    flips = [1, 1]
    for i in range(len(digits) - 1, -1, -1):
        _shift_digits(digits, i, flips, invert_j, pattern)
        _apply_digit_flips(flips, digits[i])


def _inverse_shift(digits: List[int], invert_j: bool, flip_ij: bool) -> None:
    """
    geometric (X/Y curve) digits -> old s digits, in place. LSB-first list.
    The flips state starts as the product over ALL digits and each iteration
    cancels digit i's contribution -- so at step i it holds the product of the
    digits ABOVE i, matching the forward pass's state at the same level.
    """
    pattern = PATTERN_FLIPPED_REVERSED if flip_ij else PATTERN_REVERSED
    flips = [1, 1]
    for d in digits:
        _apply_digit_flips(flips, d)
    for i in range(len(digits)):
        _apply_digit_flips(flips, digits[i])
        _shift_digits(digits, i, flips, invert_j, pattern)


def _digits_of(s: int, resolution: int) -> List[int]:
    digits: List[int] = []
    v = s
    while v > 0 or len(digits) < resolution:
        digits.append(v & 3)
        v >>= 2
    return digits


def _pack_digits(digits: List[int]) -> int:
    s = 0
    for i in range(len(digits) - 1, -1, -1):
        s = (s << 2) | digits[i]
    return s


# ---------- orientations (as in the old engine) ----------
# (reverse, invert_j, flip_ij)
_COMPAT_ORIENT = {
    'uv': (False, False, False),
    'vu': (True, False, False),
    'uw': (False, False, True),
    'wu': (True, False, True),
    'vw': (True, True, False),
    'wv': (False, True, False),
}


def compat_s_to_triple(s: int, resolution: int, orientation: Orientation = 'uv') -> Triple:
    """Old-curve position `s` -> triple coordinate, via the ORIGINAL (W/Z) forward
    descent + shiftDigits recode. No flavor (that needs a second, A5, descent)."""
    reverse, invert_j, flip_ij = _COMPAT_ORIENT[orientation]
    v = ((1 << (2 * resolution)) - 1 - s) if reverse else s
    digits = _digits_of(v, resolution)
    _forward_shift(digits, invert_j, flip_ij)
    raw = axiom_leaf_cell(ORIGINAL, _pack_digits(digits), resolution, _AXIOM_W)
    triple = ab_to_triple(raw.a, raw.b)
    if flip_ij:
        triple = Triple(triple.z, triple.y, triple.x)
    if invert_j:
        n1 = int(POW2[resolution]) - 1
        triple = Triple(triple.y - n1, triple.x + n1, triple.z)
    return triple


def compat_s_to_cell(s: int, resolution: int, orientation: Orientation = 'uv') -> Cell:
    """Old-curve position `s` -> cell (triple + pentagon flavor)."""
    triple = compat_s_to_triple(s, resolution, orientation)
    # The X/Y walk hosts every cell via a diagonal (E/e) segment, so its leaf
    # state cannot distinguish all four pentagon flavors -- that missing bit is
    # exactly why the original engine carried its fractal flips field. The
    # flavor is a per-cell geometric property, so read it off the A5 descent.
    return Cell(triple=triple, flavor=a5_triple_to_flavor(triple, resolution))


def compat_triple_to_s(t: Triple, resolution: int, orientation: Orientation = 'uv') -> Optional[int]:
    """Triple -> old-curve position `s`, or None if the triple has invalid parity."""
    total = t.x + t.y + t.z
    if total != 0 and total != 1:
        return None
    reverse, invert_j, flip_ij = _COMPAT_ORIENT[orientation]
    n = 1 << (2 * resolution)
    raw = t
    if invert_j:
        n1 = int(POW2[resolution]) - 1
        raw = Triple(raw.y - n1, raw.x + n1, raw.z)
    if flip_ij:
        raw = Triple(raw.z, raw.y, raw.x)
    ab_a, ab_b = triple_to_ab(raw)
    s_geo = axiom_target_to_s(ORIGINAL, ab_a, ab_b, resolution, _AXIOM_W, True)[0]
    digits = _digits_of(s_geo, resolution)
    _inverse_shift(digits, invert_j, flip_ij)
    v = _pack_digits(digits)
    return (n - 1 - v) if reverse else v


def compat_ij_to_s(ij: IJ, resolution: int, orientation: Orientation = 'uv') -> int:
    """Fractional IJ point -> old-curve position `s` of the containing cell."""
    reverse, invert_j, flip_ij = _COMPAT_ORIENT[orientation]
    n = 1 << (2 * resolution)
    i, j = ij[0], ij[1]
    if flip_ij:
        i, j = j, i
    if invert_j:
        j = POW2[resolution] - (i + j)
    s_geo = axiom_target_to_s(ORIGINAL, 12.0 * (i + j), -12.0 * j, resolution, _AXIOM_W, False)[0]
    digits = _digits_of(s_geo, resolution)
    _inverse_shift(digits, invert_j, flip_ij)
    v = _pack_digits(digits)
    return (n - 1 - v) if reverse else v
