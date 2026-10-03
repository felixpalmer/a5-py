# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

from typing import Optional

from .lsystem import triple_to_s_lattice
from .types import Orientation, Triple


def triple_parity(t: Triple) -> int:
    """The parity of a triple (0 or 1), equal to x + y + z."""
    return t.x + t.y + t.z


# The pentagon flavor is a CLOSED FORM of the triple. The A5 tiling is gyro
# applied to the square grid R left when every lattice edge parallel to the
# quintant's dodecahedron edge is deleted (A5 = g o^r D in Conway notation).
# Bit 0 is the triangle's parity (which half of its rhombus it is); bit 1 is
# the colour of its apex in the 2-colouring of R, coloured from a dodecahedron
# vertex. The apex of either triangle in unit square (m, n) has colour
# (m + n + max_row + 1) & 1, and x + z = -(m + n). For max_row + 1 even (every
# resolution but 0) face centres and vertices share a colour, so bit 1 is just
# (x + z) & 1; at resolution 0 they differ, which gives the corner cell flavor 2.
# Verified against the descent's flavor over all cells (tests/lattice/test_curve.py).


def triple_flavor(t: Triple, max_row: int) -> int:
    """The pentagon flavor (0-3) of a triple's cell -- orientation-independent."""
    return (t.x + t.y + t.z) | (((max_row + 1 + t.x + t.z) & 1) << 1)


def triple_in_bounds(t: Triple, max_row: int) -> bool:
    """Check if a triple is within valid quintant bounds."""
    s = t.x + t.y + t.z
    if s != 0 and s != 1:
        return False
    limit = t.y - s
    return t.x <= 0 and t.z <= 0 and t.y >= 0 and t.y <= max_row and t.x >= -limit and t.z >= -limit


def triple_to_s(t: Triple, resolution: int, orientation: Orientation = 'uv') -> Optional[int]:
    """
    Convert triple coordinates to an s-value on the A5 (L-system) curve.
    The engine's `a5.lattice.triple_to_s` is currently the compat alias; this is
    the pure-curve form it swaps to at the canonical cutover (mirrors the other
    ports' triple modules).

    Returns s-value, or None if the triple has invalid parity.
    """
    s = t.x + t.y + t.z
    if s != 0 and s != 1:
        return None
    return triple_to_s_lattice(t, resolution, orientation)
