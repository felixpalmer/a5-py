# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

from typing import List, Set

from ..core.serialization import deserialize, get_resolution, serialize, FIRST_CURVE_RESOLUTION
from ..core.origin import origins
from ..core.face_adjacency import FACE_ADJACENCY
from .triple_cells import cell_ids_to_triples, for_each_triple_neighbor, triple_cell_to_id


def get_global_cell_neighbors(cell_id: int, edge_only: bool = False) -> List[int]:
    """
    Get all neighbors of a cell across quintant and face boundaries: within its
    quintant the fixed per-flavor triple deltas, and across a quintant edge the
    boundary delta tables (see `for_each_triple_neighbor`).

    Args:
        edge_only: If True, return only edge-sharing neighbors (5 per cell).
            Default False returns all neighbors including vertex-only neighbors (6-8 per cell).
    """
    resolution = get_resolution(cell_id)
    neighbors: Set[int] = set()
    if resolution == 0:
        # The cells are the 12 dodecahedron faces, adjacent across their edges
        for face, _ in FACE_ADJACENCY[deserialize(cell_id)['origin'].id]:
            neighbors.add(serialize({'origin': origins[face], 'segment': 0, 'S': 0, 'resolution': 0}))
    else:
        curve_level = resolution - FIRST_CURVE_RESOLUTION + 1

        def visit(o: int, q: int, x: int, y: int, z: int) -> None:
            neighbors.add(triple_cell_to_id(o, q, x, y, z, curve_level, resolution))

        for_each_triple_neighbor(*cell_ids_to_triples([cell_id]), (1 << curve_level) - 1, edge_only, visit)
    return sorted(neighbors)
