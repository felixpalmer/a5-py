# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

from typing import List, Set
from ..lattice import s_to_cell, triple_parity
from ..core.utils import Origin
from ..core.serialization import deserialize, serialize, FIRST_HILBERT_RESOLUTION
from ..core.origin import origins, QUINTANT_TO_SEGMENT, SEGMENT_TO_ORIENTATION, SEGMENT_TO_QUINTANT
from ..core.face_adjacency import FACE_ADJACENCY
from .quintant_neighbors import find_quintant_neighbor_s
from .lattice_boundary import BoundaryContext, get_boundary_neighbors


def _serialize_res1(origin: Origin, quintant: int) -> int:
    """Serialize a res 1 cell from origin and quintant."""
    segment = QUINTANT_TO_SEGMENT[origin.id * 5 + quintant]
    return serialize({'origin': origin, 'segment': segment, 'S': 0, 'resolution': 1})


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


def _get_res1_neighbors(origin: Origin, segment: int, edge_only: bool) -> List[int]:
    """
    Get neighbors of a resolution 1 cell (quintant).
    """
    quintant = SEGMENT_TO_QUINTANT[origin.id * 5 + segment]
    neighbor_set: Set[int] = set()

    # Left and right quintant on the same face (A, B)
    left_q = (quintant - 1 + 5) % 5
    right_q = (quintant + 1) % 5
    neighbor_set.add(_serialize_res1(origin, left_q))
    neighbor_set.add(_serialize_res1(origin, right_q))

    # Adjacent quintant on adjacent face (C)
    adjacent_face_id, adjacent_quintant = FACE_ADJACENCY[origin.id][quintant]
    adjacent_origin = origins[adjacent_face_id]
    neighbor_set.add(_serialize_res1(adjacent_origin, adjacent_quintant))

    if edge_only:
        return sorted(neighbor_set)

    # Remaining neighbors on face
    neighbor_set.add(_serialize_res1(origin, (quintant - 2 + 5) % 5))
    neighbor_set.add(_serialize_res1(origin, (quintant + 2) % 5))

    # Left & right quintant neighbors of C
    neighbor_set.add(_serialize_res1(adjacent_origin, (adjacent_quintant - 1 + 5) % 5))
    neighbor_set.add(_serialize_res1(adjacent_origin, (adjacent_quintant + 1) % 5))

    # Two neighbors each from adjacent faces of A & B
    left_adjacent_face_id, left_adjacent_quintant = FACE_ADJACENCY[origin.id][left_q]
    left_adjacent_origin = origins[left_adjacent_face_id]
    neighbor_set.add(_serialize_res1(left_adjacent_origin, left_adjacent_quintant))
    neighbor_set.add(_serialize_res1(left_adjacent_origin, (left_adjacent_quintant - 1 + 5) % 5))

    right_adjacent_face_id, right_adjacent_quintant = FACE_ADJACENCY[origin.id][right_q]
    right_adjacent_origin = origins[right_adjacent_face_id]
    neighbor_set.add(_serialize_res1(right_adjacent_origin, right_adjacent_quintant))
    neighbor_set.add(_serialize_res1(right_adjacent_origin, (right_adjacent_quintant + 1) % 5))

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
    if resolution == 1:
        return _get_res1_neighbors(origin, segment, edge_only)

    hilbert_res = resolution - FIRST_HILBERT_RESOLUTION + 1
    global_quintant = origin.id * 5 + segment
    source_quintant = SEGMENT_TO_QUINTANT[global_quintant]
    source_orientation = SEGMENT_TO_ORIENTATION[global_quintant]

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
