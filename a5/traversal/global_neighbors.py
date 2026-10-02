# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

from typing import List, Set
from ..lattice import s_to_cell, triple_parity
from ..core.utils import Origin
from ..core.serialization import deserialize, serialize, FIRST_HILBERT_RESOLUTION
from ..core.origin import segment_to_quintant, origins
from ..core.face_adjacency import FACE_ADJACENCY
from .quintant_neighbors import find_quintant_neighbor_s
from .lattice_boundary import BoundaryContext, get_boundary_neighbors


def _get_res0_neighbors(origin: Origin) -> List[int]:
    """
    Get neighbors of a resolution 0 cell (dodecahedron face).
    """
    neighbor_set: Set[int] = set()
    for q in range(5):
        adjacent_face_id, _ = FACE_ADJACENCY[origin.id][q]
        neighbor_set.add(serialize({
            'origin': origins[adjacent_face_id], 'segment': 0,
            'S': 0, 'resolution': 0
        }))
    return sorted(neighbor_set)


def get_global_cell_neighbors(cell_id: int, edge_only: bool = False) -> List[int]:
    """
    Get all neighbors of a cell across quintant and face boundaries.

    Within-quintant neighbors come from the fixed per-flavor triple deltas
    (via find_quintant_neighbor_s). Cross-quintant, cross-face, apex, and
    corner neighbors are emitted by the shared get_boundary_neighbors helper
    using fixed delta tables -- see lattice_boundary.py.
    """
    cell = deserialize(cell_id)
    origin, segment, S, resolution = cell['origin'], cell['segment'], cell['S'], cell['resolution']
    if resolution == 0:
        return _get_res0_neighbors(origin)

    hilbert_res = resolution - FIRST_HILBERT_RESOLUTION + 1
    source_quintant, source_orientation = segment_to_quintant(segment, origin)

    # Triple coordinates are orientation-independent
    source_cell = s_to_cell(S, hilbert_res, source_orientation)
    triple = source_cell.triple

    neighbor_set: Set[int] = set()

    # --- Within-quintant: fixed per-flavor triple deltas ---
    for neighbor_s in find_quintant_neighbor_s(triple, source_cell.flavor, S, hilbert_res, source_orientation, edge_only):
        neighbor_set.add(serialize({
            'origin': origin, 'segment': segment,
            'S': neighbor_s, 'resolution': resolution
        }))

    # --- Cross-quintant / cross-face / apex / corner: shared lattice-boundary helper ---
    ctx = BoundaryContext(
        triple=triple,
        parity=triple_parity(triple),
        source_quintant=source_quintant,
        origin=origin,
        hilbert_res=hilbert_res,
        max_s=4 ** hilbert_res,
        max_row=(1 << hilbert_res) - 1,
        resolution=resolution,
    )
    for cid in get_boundary_neighbors(ctx, edge_only):
        neighbor_set.add(cid)

    return sorted(neighbor_set)
