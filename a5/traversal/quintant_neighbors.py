# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

from typing import List, Optional
from ..lattice import (
    Anchor, Orientation, Triple,
    s_to_anchor, anchor_to_triple, triple_to_anchor, triple_to_s, triple_in_bounds
)
from .neighbors import is_neighbor


def find_quintant_neighbor_s(
    source_triple: Triple,
    uv_source_anchor: Optional[Anchor],
    source_s: int,
    resolution: int,
    orientation: Orientation,
    edge_only: bool
) -> List[int]:
    """
    Find within-quintant neighbors via triple coordinate search.

    Generates +/-1 candidate triples, validates with is_neighbor() in uv space,
    and converts validated triples to s-values in the requested orientation.
    """
    max_s = 4 ** resolution
    max_row = (1 << resolution) - 1
    neighbors: List[int] = []

    for dx in range(-1, 2):
        for dy in range(-1, 2):
            for dz in range(-1, 2):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                if abs(dx) + abs(dy) + abs(dz) > 3:
                    continue
                if edge_only and abs(dx) + abs(dy) + abs(dz) > 2:
                    continue

                neighbor_triple = Triple(
                    source_triple.x + dx,
                    source_triple.y + dy,
                    source_triple.z + dz
                )
                if not triple_in_bounds(neighbor_triple, max_row):
                    continue

                # Validate in uv space where is_neighbor is known to work
                uv_neighbor_anchor = triple_to_anchor(neighbor_triple, resolution, 'uv')
                if uv_neighbor_anchor is None or uv_source_anchor is None:
                    continue
                if not is_neighbor(uv_source_anchor, uv_neighbor_anchor):
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
    Fast neighbor finding using triple coordinates.

    Strategy:
    1. Convert cell to triple coordinates (x, y, z) -- orientation-independent
    2. Generate neighbor triples (Manhattan distance <= 3) -- ~12 candidates
    3. Validate with is_neighbor() in 'uv' space
    4. Convert validated triples to s-values in the requested orientation
    """
    anchor = s_to_anchor(s, resolution, orientation)
    triple = anchor_to_triple(anchor)
    uv_source_anchor = triple_to_anchor(triple, resolution, 'uv')

    result = find_quintant_neighbor_s(
        triple, uv_source_anchor, s, resolution, orientation, edge_only
    )
    result.sort()
    return result
