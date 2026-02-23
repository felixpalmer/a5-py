# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

from typing import Tuple
from ..core.coordinate_systems import KJ
from .types import Quaternary, Flip, YES, NO

# Using KJ allows simplification of definitions
k_pos = (1.0, 0.0)  # k
j_pos = (0.0, 1.0)  # j
k_neg = (-k_pos[0], -k_pos[1])
j_neg = (-j_pos[0], -j_pos[1])
ZERO = (0.0, 0.0)


def quaternary_to_kj(n: Quaternary, flips: Tuple[Flip, Flip]) -> KJ:
    """Indirection to allow for flips."""
    flip_x, flip_y = flips
    p = ZERO
    q = ZERO

    if flip_x == NO and flip_y == NO:
        p = k_pos
        q = j_pos
    elif flip_x == YES and flip_y == NO:
        # Swap and negate
        p = j_neg
        q = k_neg
    elif flip_x == NO and flip_y == YES:
        # Swap only
        p = j_pos
        q = k_pos
    elif flip_x == YES and flip_y == YES:
        # Negate only
        p = k_neg
        q = j_neg

    if n == 0:
        return ZERO
    elif n == 1:
        return p
    elif n == 2:
        return (p[0] + q[0], p[1] + q[1])
    elif n == 3:
        return (q[0] + 2 * p[0], q[1] + 2 * p[1])
    else:
        raise ValueError(f"Invalid Quaternary value: {n}")


def quaternary_to_flips(n: Quaternary) -> Tuple[Flip, Flip]:
    """Convert quaternary number to flip configuration."""
    flips = [(NO, NO), (NO, YES), (NO, NO), (YES, NO)]
    return flips[n]


def ij_to_quaternary(ij: Tuple[float, float], flips: Tuple[Flip, Flip]) -> Quaternary:
    """Convert IJ coordinates to quaternary number with flips."""
    u, v = ij
    digit = 0

    # Boundaries to compare against
    a = -(u + v) if flips[0] == YES else u + v
    b = -u if flips[1] == YES else u
    c = -v if flips[0] == YES else v

    # Only one flip
    if flips[0] + flips[1] == 0:
        if c < 1:
            digit = 0
        elif b > 1:
            digit = 3
        elif a > 1:
            digit = 2
        else:
            digit = 1
    # No flips or both
    else:
        if a < 1:
            digit = 0
        elif b > 1:
            digit = 3
        elif c > 1:
            digit = 2
        else:
            digit = 1

    return digit
