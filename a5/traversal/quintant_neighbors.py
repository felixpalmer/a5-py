# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

from typing import List
from ..lattice import (
    Orientation, Triple,
    s_to_cell, triple_to_s, triple_in_bounds,
)
from .neighbors import NEIGHBOR_DELTAS


def find_quintant_neighbor_s(
    source_triple: Triple,
    source_flavor: int,
    source_s: int,
    resolution: int,
    orientation: Orientation,
    edge_only: bool
) -> List[int]:
    """
    Find within-quintant neighbors via the cell's pentagon flavor.

    A cell's neighbors sit at fixed triple deltas determined by its flavor
    (NEIGHBOR_DELTAS -- 5 edge-sharing + 2 vertex-only), so no per-candidate
    validation is needed: each in-bounds delta is a neighbor.

    Args:
        source_triple: Triple coordinates of the source cell
        source_flavor: Pentagon flavor of the source cell (0-3)
        source_s: Source s-value to exclude from results
        resolution: Resolution level
        orientation: Curve orientation
        edge_only: If True, only the 5 edge-sharing neighbors
    """
    max_s = 4 ** resolution
    max_row = (1 << resolution) - 1
    deltas = NEIGHBOR_DELTAS[source_flavor]
    neighbors: List[int] = []

    lst = deltas.edge if edge_only else deltas.all
    for d in lst:
        neighbor_triple = Triple(
            source_triple.x + d.x,
            source_triple.y + d.y,
            source_triple.z + d.z,
        )
        if not triple_in_bounds(neighbor_triple, max_row):
            continue
        neighbor_s = triple_to_s(neighbor_triple, resolution, orientation)
        if neighbor_s is not None and 0 <= neighbor_s < max_s and neighbor_s != source_s:
            neighbors.append(neighbor_s)

    return neighbors


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

    result = find_quintant_neighbor_s(
        cell.triple, cell.flavor, s, resolution, orientation, edge_only
    )
    result.sort()
    return result
