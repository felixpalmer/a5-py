# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

from typing import Optional

from .lsystem import triple_to_s_lattice
from .types import Orientation, Triple


def triple_parity(t: Triple) -> int:
    """The parity of a triple (0 or 1), equal to x + y + z."""
    return t.x + t.y + t.z


# The pentagon flavor is a CLOSED FORM of the triple: it depends only on the
# parity and y mod 2 (the Cairo-like tiling repeats its four orientations with
# period 2). Verified exhaustively against the descent's flavor over all cells
# (see tests/lattice/test_curve.py); the descent's leaf flavor agrees because
# both describe the same fixed tiling.
_FLAVOR_LUT = (0, 2, 3, 1)  # index = parity << 1 | (y & 1)


def triple_flavor(t: Triple) -> int:
    """The pentagon flavor (0-3) of a triple's cell -- orientation-independent."""
    return _FLAVOR_LUT[((t.x + t.y + t.z) << 1) | (t.y & 1)]


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
