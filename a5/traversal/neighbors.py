# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

from typing import List, Tuple
from ..lattice import Anchor, Flip
from ..core.tiling import get_pentagon_flavor, PentagonFlavor

# [di, dj, flip0, flip1]
NeighborPattern = Tuple[int, int, Flip, Flip]

NEIGHBORS: dict = {
    0: [
        (0, -2, -1, 1), (0, -2, -1, -1),
        (0, -1, 1, -1), (0, -1, -1, -1), (0, -1, 1, 1),
        (1, -2, -1, -1),
        (1, -1, -1, 1), (1, -1, 1, -1),
        (1, 0, 1, -1),
        (2, -1, 1, -1),
        (2, -2, -1, -1),
    ],
    1: [
        (-1, -1, -1, 1),
        (0, -2, -1, -1),
        (0, -1, -1, -1), (0, -1, 1, -1),
        (0, 0, -1, 1), (0, 0, -1, -1),
        (0, 1, 1, -1), (0, 1, 1, 1),
        (1, -2, -1, -1),
        (1, -1, 1, -1), (1, -1, -1, -1),
        (1, 0, 1, -1),
    ],
    2: [
        (-2, 2, -1, -1),
        (-2, 1, 1, -1),
        (-1, 0, 1, -1),
        (-1, 1, 1, -1), (-1, 1, -1, 1),
        (-1, 2, -1, -1),
        (0, 1, -1, -1), (0, 1, 1, -1), (0, 1, 1, 1),
        (0, 2, -1, -1), (0, 2, -1, 1),
    ],
    3: [
        (-1, 0, 1, -1),
        (-1, 1, 1, -1), (-1, 1, -1, -1),
        (-1, 2, -1, -1),
        (0, -1, 1, -1), (0, -1, 1, 1),
        (0, 0, -1, -1), (0, 0, -1, 1),
        (0, 1, -1, -1), (0, 1, 1, -1),
        (0, 2, -1, -1),
        (1, 1, -1, 1),
    ],
    4: [
        (0, -1, 1, -1), (0, -1, 1, 1),
        (0, 0, -1, -1), (0, 0, -1, 1),
        (0, 1, -1, -1),
        (1, 0, -1, -1), (1, 0, 1, -1),
        (1, -1, 1, -1), (1, 1, -1, 1),
        (2, -1, 1, -1), (2, 0, -1, -1),
    ],
    5: [
        (-1, 1, -1, 1),
        (0, -1, 1, -1),
        (0, 0, -1, -1),
        (0, 1, -1, -1), (0, 1, 1, -1), (0, 1, 1, 1),
        (0, 2, -1, -1), (0, 2, -1, 1),
        (1, -1, 1, -1),
        (1, 0, -1, -1), (1, 0, 1, -1),
        (1, 1, -1, -1),
    ],
    6: [
        (-2, 0, -1, -1),
        (-2, 1, 1, -1),
        (-1, -1, -1, 1),
        (-1, 0, -1, -1), (-1, 0, 1, -1),
        (-1, 1, 1, -1),
        (0, -1, -1, -1),
        (0, 0, -1, -1), (0, 0, -1, 1),
        (0, 1, 1, -1), (0, 1, 1, 1),
    ],
    7: [
        (-1, -1, -1, -1),
        (-1, 0, -1, -1), (-1, 0, 1, -1),
        (-1, 1, 1, -1),
        (0, -2, -1, -1), (0, -2, -1, 1),
        (0, -1, -1, -1), (0, -1, 1, -1), (0, -1, 1, 1),
        (0, 0, -1, -1),
        (0, 1, 1, -1),
        (1, -1, -1, 1),
    ],
}


def is_neighbor(origin: Anchor, candidate: Anchor) -> bool:
    """Check if two anchors are neighbors in uv/raw IJ space."""
    origin_flavor = get_pentagon_flavor(origin)
    candidate_flavor = get_pentagon_flavor(candidate)
    if origin_flavor == candidate_flavor:
        return False
    neighbors = NEIGHBORS[origin_flavor]
    relative = (
        candidate.offset[0] - origin.offset[0],
        candidate.offset[1] - origin.offset[1],
        candidate.flips[0] * origin.flips[0],
        candidate.flips[1] * origin.flips[1],
    )

    for pattern in neighbors:
        if (relative[0] == pattern[0] and
            relative[1] == pattern[1] and
            relative[2] == pattern[2] and
            relative[3] == pattern[3]):
            return True

    return False
