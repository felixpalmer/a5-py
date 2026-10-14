# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

from typing import List
from ..lattice import (
    Orientation, Triple,
    s_to_cell, triple_to_s, triple_in_bounds,
)
from .neighbors import NEIGHBOR_DELTAS


def get_cell_neighbors(
    s: int,
    resolution: int,
    orientation: Orientation = 'uv',
    edge_only: bool = False
) -> List[int]:
    """
    Neighbor finding via triple coordinates and pentagon flavor.

    Triple coordinates are orientation-independent -- the same geometric cell
    always has the same triple coords regardless of curve orientation. Only the
    s-value changes between orientations, so neighbors are found in triple space
    and converted back to the requested orientation.
    """
    cell = s_to_cell(s, resolution, orientation)
    max_row = (1 << resolution) - 1
    deltas = NEIGHBOR_DELTAS[cell.flavor].edge if edge_only else NEIGHBOR_DELTAS[cell.flavor].all
    neighbors: List[int] = []
    for d in deltas:
        neighbor = Triple(cell.triple.x + d.x, cell.triple.y + d.y, cell.triple.z + d.z)
        if triple_in_bounds(neighbor, max_row):
            neighbors.append(triple_to_s(neighbor, resolution, orientation))
    return sorted(neighbors)
