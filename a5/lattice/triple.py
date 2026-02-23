# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

import math
from typing import Tuple, Optional, NamedTuple
from ..core.coordinate_systems import IJ
from .types import Anchor, Orientation, YES, NO
from .anchor import offset_flips_to_anchor
from .hilbert import s_to_anchor, anchor_to_s, ij_to_s, ij_to_flips


class Triple(NamedTuple):
    x: int
    y: int
    z: int


def triple_parity(t: Triple) -> int:
    """The parity of a triple (0 or 1), equal to x + y + z."""
    return t.x + t.y + t.z


def triple_in_bounds(t: Triple, max_row: int) -> bool:
    """Check if a triple is within valid quintant bounds."""
    s = t.x + t.y + t.z
    if s != 0 and s != 1:
        return False
    limit = t.y - s
    return t.x <= 0 and t.z <= 0 and t.y >= 0 and t.y <= max_row and t.x >= -limit and t.z >= -limit


def triple_to_s(t: Triple, resolution: int, orientation: Orientation = 'uv') -> Optional[int]:
    """
    Convert triple coordinates to an s-value (Hilbert index).
    Returns s-value, or None if the triple has invalid parity.
    """
    anchor = triple_to_anchor(t, resolution, orientation)
    if anchor is None:
        return None
    return anchor_to_s(anchor, resolution, orientation)


def anchor_to_triple(anchor: Anchor) -> Triple:
    """
    Compute triple coordinates from an anchor.

    Maps the pentagonal A5 grid to a triangular grid coordinate system where
    neighbors differ by +/-1 in exactly one coordinate while the other two stay constant.
    """
    # Start with shift in IJ space
    shift_i = 0.25
    shift_j = 0.25
    flip0, flip1 = anchor.flips

    # First check for [1, -1] rotation
    if flip0 == NO and flip1 == YES:
        # Rotate 180 degrees
        shift_i = -shift_i
        shift_j = -shift_j

    # Then apply additional adjustments
    if flip0 == YES and flip1 == YES:
        # Rotate 180 degrees
        shift_i = -shift_i
        shift_j = -shift_j
    elif flip0 == YES:
        # Shift left (subtract w = [0, 1])
        shift_j -= 1
    elif flip1 == YES:
        # Shift right (add w = [0, 1])
        shift_j += 1

    # Compute center
    i = anchor.offset[0] + shift_i
    j = anchor.offset[1] + shift_j

    # Compute row and column in triangular grid
    r = (i + j) - 0.5
    c = (i - j) + r

    # Compute triple coordinates (all integers for valid anchors)
    x = int(math.floor((c + 1) / 2 - r))
    y = int(r)
    z = int(math.floor((1 - c) / 2))

    return Triple(x, y, z)


def triple_to_anchor(t: Triple, resolution: int, orientation: Orientation = 'uv') -> Optional[Anchor]:
    """
    Convert triple coordinates to an Anchor.

    This is the inverse of anchor_to_triple().
    """
    x, y, z = t

    # Verify parity constraint
    s = x + y + z
    if s != 0 and s != 1:
        return None

    # Compute r and c from triple coordinates
    r = y
    c_min = max(2 * x + 2 * r - 1, -2 * z - 1 + 0.0001)
    c_max = min(2 * x + 2 * r + 1 - 0.0001, 1 - 2 * z)
    c = round((c_min + c_max) / 2)

    # Compute center IJ coordinates from r and c
    center_i = (c + 0.5) / 2
    center_j = r - c / 2 + 0.25

    # Fast path for uv/vu: use ij_to_flips directly (works in raw IJ space)
    if orientation == 'uv' or orientation == 'vu':
        flips = ij_to_flips((center_i, center_j), resolution)

        # Compute shift from flips (inverse of anchor_to_triple logic)
        shift_i = 0.25
        shift_j = 0.25
        if flips[0] == NO and flips[1] == YES:
            shift_i = -shift_i
            shift_j = -shift_j
        if flips[0] == YES and flips[1] == YES:
            shift_i = -shift_i
            shift_j = -shift_j
        elif flips[0] == YES:
            shift_j -= 1
        elif flips[1] == YES:
            shift_j += 1

        offset = (round(center_i - shift_i), round(center_j - shift_j))
        return offset_flips_to_anchor(offset, flips, orientation)

    # General path: ij_to_s -> s_to_anchor (handles all orientation transforms)
    s_val = ij_to_s((center_i, center_j), resolution, orientation)
    return s_to_anchor(s_val, resolution, orientation)
