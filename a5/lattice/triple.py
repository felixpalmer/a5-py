# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

from typing import Optional

from .compat import compat_triple_to_s
from .types import Orientation, Triple


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
    Convert triple coordinates to an s-value (curve index).
    Returns s-value, or None if the triple has invalid parity.
    """
    s = t.x + t.y + t.z
    if s != 0 and s != 1:
        return None
    return compat_triple_to_s(t, resolution, orientation)
