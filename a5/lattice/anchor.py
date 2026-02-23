# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

from typing import Tuple
from ..core.coordinate_systems import IJ
from .types import Anchor, Quaternary, Flip, Orientation, YES, NO


def _is_group2_orientation(orientation: Orientation) -> bool:
    """Group 2: uw, wu."""
    return orientation == 'uw' or orientation == 'wu'


def compute_q(offset: IJ, flips: Tuple[Flip, Flip], orientation: Orientation = 'uv') -> Quaternary:
    """
    Deduce the quaternary value q from offset and flip values.

    Uses the discovered invariant that q can be deterministically computed
    from offset parity and flip values.
    """
    i, j = offset
    flip0, flip1 = flips

    imod2 = int(i) & 1
    jmod2 = int(j) & 1
    f0idx = (flip0 + 1) >> 1  # Map: YES (-1) -> 0, NO (1) -> 1
    f1idx = (flip1 + 1) >> 1

    if _is_group2_orientation(orientation):
        group2_lookup = [
            [[[0, 3], [3, 0]], [[3, 2], [2, 3]]],
            [[[2, 1], [1, 2]], [[1, 0], [0, 1]]]
        ]
        return group2_lookup[imod2][jmod2][f0idx][f1idx]
    else:
        if imod2 == 0:
            return 0 if jmod2 == 0 else 2
        odd_i_lookup = [
            [[3, 1], [1, 3]],
            [[1, 3], [3, 1]]
        ]
        return odd_i_lookup[jmod2][f0idx][f1idx]


def offset_flips_to_anchor(offset: IJ, flips: Tuple[Flip, Flip], orientation: Orientation = 'uv') -> Anchor:
    """
    Create a complete Anchor by deducing q from offset and flips.
    """
    q = compute_q(offset, flips, orientation)
    return Anchor(q, offset, flips)
