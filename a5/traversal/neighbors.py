# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

# The neighbors of a cell, in triple space, are a fixed function of its
# pentagon flavor: pentagons tile edge-to-edge, so the arrangement around a
# pentagon is forced. Every cell has exactly 5 edge-sharing and 2 vertex-only
# neighbors, at the triple deltas below. No validation is needed -- each delta
# IS a neighbor (bounds permitting).
#
# Derived geometrically (shared pentagon vertices) and verified exhaustively
# over all interior cells at res 4-5, all orientations, zero conflicts. The
# flavor-1/3 lists are the flavor-0/2 lists negated (they are the 180-deg-rotated
# shapes).

from typing import List, NamedTuple
from ..lattice import Triple


class NeighborDeltas(NamedTuple):
    edge: List[Triple]    # 5 edge-sharing neighbors
    vertex: List[Triple]  # 2 vertex-only neighbors
    all: List[Triple]     # edge ++ vertex, spelled out so this stays a pure data table


def _d(x: int, y: int, z: int) -> Triple:
    return Triple(x, y, z)


NEIGHBOR_DELTAS: List[NeighborDeltas] = [
    NeighborDeltas(  # flavor 0
        edge=[_d(0, 0, 1), _d(0, 1, -1), _d(0, 1, 0), _d(1, -1, 0), _d(1, 0, 0)],
        vertex=[_d(1, -1, 1), _d(1, 1, -1)],
        all=[_d(0, 0, 1), _d(0, 1, -1), _d(0, 1, 0), _d(1, -1, 0), _d(1, 0, 0), _d(1, -1, 1), _d(1, 1, -1)],
    ),
    NeighborDeltas(  # flavor 1 (= flavor 0 rotated 180 deg: deltas negated)
        edge=[_d(0, 0, -1), _d(0, -1, 1), _d(0, -1, 0), _d(-1, 1, 0), _d(-1, 0, 0)],
        vertex=[_d(-1, 1, -1), _d(-1, -1, 1)],
        all=[_d(0, 0, -1), _d(0, -1, 1), _d(0, -1, 0), _d(-1, 1, 0), _d(-1, 0, 0), _d(-1, 1, -1), _d(-1, -1, 1)],
    ),
    NeighborDeltas(  # flavor 2
        edge=[_d(-1, 1, 0), _d(0, -1, 1), _d(0, 0, 1), _d(0, 1, 0), _d(1, 0, 0)],
        vertex=[_d(-1, 1, 1), _d(1, -1, 1)],
        all=[_d(-1, 1, 0), _d(0, -1, 1), _d(0, 0, 1), _d(0, 1, 0), _d(1, 0, 0), _d(-1, 1, 1), _d(1, -1, 1)],
    ),
    NeighborDeltas(  # flavor 3 (= flavor 2 rotated 180 deg: deltas negated)
        edge=[_d(1, -1, 0), _d(0, 1, -1), _d(0, 0, -1), _d(0, -1, 0), _d(-1, 0, 0)],
        vertex=[_d(1, -1, -1), _d(-1, 1, -1)],
        all=[_d(1, -1, 0), _d(0, 1, -1), _d(0, 0, -1), _d(0, -1, 0), _d(-1, 0, 0), _d(1, -1, -1), _d(-1, 1, -1)],
    ),
]
