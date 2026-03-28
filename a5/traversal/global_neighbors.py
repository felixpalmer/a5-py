# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

from typing import List, Tuple, Optional, Set
from ..lattice import (
    Orientation, Triple,
    s_to_anchor, anchor_to_triple, triple_to_anchor, triple_to_s, triple_parity, triple_in_bounds
)
from ..core.utils import Origin
from ..core.serialization import deserialize, serialize, FIRST_HILBERT_RESOLUTION
from ..core.origin import segment_to_quintant, quintant_to_segment, origins
from ..core.face_adjacency import FACE_ADJACENCY
from .quintant_neighbors import find_quintant_neighbor_s

# Neighbor delta: (dx, dy, dz, is_edge_sharing)
NeighborDelta = Tuple[int, int, int, bool]

# Cross-quintant left-edge deltas (source z=0), indexed by parity * 2 + (y_odd ? 1 : 0)
LEFT_EDGE_DELTAS: List[List[NeighborDelta]] = [
    # parity=0, yEven
    [(0, 0, 0, True), (0, 0, 1, False)],
    # parity=0, yOdd
    [(0, 0, 0, True), (0, 1, 0, True), (0, -1, 1, False), (0, 1, -1, False)],
    # parity=1, yEven
    [],
    # parity=1, yOdd
    [(0, -1, 0, True), (0, 0, -1, False)],
]

# Cross-quintant right-edge deltas (source x=0), indexed by parity * 2 + (y_odd ? 1 : 0)
RIGHT_EDGE_DELTAS: List[List[NeighborDelta]] = [
    # parity=0, yEven
    [(0, 0, 0, True), (0, 1, 0, True), (-1, 1, 0, False), (1, -1, 0, False)],
    # parity=0, yOdd
    [(0, 0, 0, True), (1, 0, 0, False)],
    # parity=1, yEven
    [(0, -1, 0, True), (-1, 0, 0, False)],
    # parity=1, yOdd
    [],
]

# Cross-face base-edge deltas (source y=maxRow), indexed by parity
CROSS_FACE_DELTAS: List[List[NeighborDelta]] = [
    # parity=0
    [(0, 0, 0, True), (1, 0, 0, True), (1, 0, -1, False)],
    # parity=1
    [(0, 0, -1, True), (0, 0, 0, False)],
]


class _NeighborContext:
    """Shared state for neighbor search helpers."""
    def __init__(self, hilbert_res: int, resolution: int, edge_only: bool):
        self.hilbert_res = hilbert_res
        self.resolution = resolution
        self.max_s = 4 ** hilbert_res
        self.max_row = (1 << hilbert_res) - 1
        self.edge_only = edge_only
        self.neighbor_set: Set[int] = set()


def _add_neighbor(
    ctx: _NeighborContext,
    neighbor_triple: Triple, orientation: Orientation,
    neighbor_origin: Origin, neighbor_segment: int
) -> None:
    """Try to convert a triple to a cell ID and add it to the neighbor set."""
    s = triple_to_s(neighbor_triple, ctx.hilbert_res, orientation)
    if s is None or s < 0 or s >= ctx.max_s:
        return
    ctx.neighbor_set.add(serialize({
        'origin': neighbor_origin, 'segment': neighbor_segment,
        'S': s, 'resolution': ctx.resolution
    }))


def _add_delta_neighbors(
    ctx: _NeighborContext,
    base: Triple, deltas: List[NeighborDelta],
    orientation: Orientation, neighbor_origin: Origin, neighbor_segment: int
) -> None:
    """Apply a delta table to a base triple and add valid neighbors."""
    for dx, dy, dz, is_edge in deltas:
        if ctx.edge_only and not is_edge:
            continue
        neighbor_triple = Triple(base.x + dx, base.y + dy, base.z + dz)
        if not triple_in_bounds(neighbor_triple, ctx.max_row):
            continue
        _add_neighbor(ctx, neighbor_triple, orientation, neighbor_origin, neighbor_segment)


def _serialize_res1(origin: Origin, quintant: int) -> int:
    """Serialize a res 1 cell from origin and quintant."""
    segment, _ = quintant_to_segment(quintant, origin)
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
    quintant, _ = segment_to_quintant(segment, origin)
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

    Uses three strategies:
    - Within-quintant: Standard triple coordinate approach
    - Cross-quintant: Lateral edge handling via coordinate transformation
    - Cross-face: Base edge handling via x<->z swap
    """
    cell = deserialize(cell_id)
    origin, segment, S, resolution = cell['origin'], cell['segment'], cell['S'], cell['resolution']
    if resolution == 0:
        return _get_res0_neighbors(origin)
    if resolution == 1:
        return _get_res1_neighbors(origin, segment, edge_only)

    hilbert_res = resolution - FIRST_HILBERT_RESOLUTION + 1
    source_quintant, source_orientation = segment_to_quintant(segment, origin)
    anchor = s_to_anchor(S, hilbert_res, source_orientation)

    # Triple coordinates are orientation-independent
    triple = anchor_to_triple(anchor)

    # Get uv anchor for is_neighbor validation (within-quintant)
    uv_source_anchor = triple_to_anchor(triple, hilbert_res, 'uv')

    ctx = _NeighborContext(hilbert_res, resolution, edge_only)

    # --- Within-quintant neighbors ---
    for neighbor_s in find_quintant_neighbor_s(triple, uv_source_anchor, S, hilbert_res, source_orientation, edge_only):
        ctx.neighbor_set.add(serialize({
            'origin': origin, 'segment': segment,
            'S': neighbor_s, 'resolution': resolution
        }))

    # --- Cross-quintant neighbors ---
    parity = triple_parity(triple)  # 0 or 1
    y_odd = triple.y % 2 != 0
    delta_index = parity * 2 + (1 if y_odd else 0)

    # Left edge (z=0): neighbor in previous quintant at swapped [0, y, x]
    if triple.z == 0:
        target_quintant = (source_quintant - 1 + 5) % 5
        target_segment, target_orientation = quintant_to_segment(target_quintant, origin)
        swapped_base = Triple(0, triple.y, triple.x)
        _add_delta_neighbors(ctx, swapped_base, LEFT_EDGE_DELTAS[delta_index], target_orientation, origin, target_segment)

    # Right edge (x=0): neighbor in next quintant at swapped [z, y, 0]
    if triple.x == 0:
        target_quintant = (source_quintant + 1) % 5
        target_segment, target_orientation = quintant_to_segment(target_quintant, origin)
        swapped_base = Triple(triple.z, triple.y, 0)
        _add_delta_neighbors(ctx, swapped_base, RIGHT_EDGE_DELTAS[delta_index], target_orientation, origin, target_segment)

    # --- Cross-face neighbors ---
    if triple.y == ctx.max_row:
        adj_face_id, adj_quintant = FACE_ADJACENCY[origin.id][source_quintant]
        adj_origin = origins[adj_face_id]
        adj_segment, adj_orientation = quintant_to_segment(adj_quintant, adj_origin)
        mirrored_base = Triple(triple.z, ctx.max_row, triple.x)
        _add_delta_neighbors(ctx, mirrored_base, CROSS_FACE_DELTAS[parity], adj_orientation, adj_origin, adj_segment)

    # Apex: [0,0,0] cells from all 5 quintants meet at the face center
    if triple.x == 0 and triple.y == 0 and triple.z == 0:
        for q in range(5):
            if q == source_quintant:
                continue
            # Adjacent quintants (distance=1) share an edge; non-adjacent (distance=2) share only a vertex
            distance = min(
                (q - source_quintant + 5) % 5,
                (source_quintant - q + 5) % 5
            )
            if edge_only and distance != 1:
                continue
            target_segment, target_orientation = quintant_to_segment(q, origin)
            _add_neighbor(ctx, triple, target_orientation, origin, target_segment)

    # Special case: base-left corner cells
    if triple.x == -ctx.max_row and triple.y == ctx.max_row and triple.z == 0:
        # Vertex neighbor 1: across the previous quintant's base edge
        prev_quintant = (source_quintant - 1 + 5) % 5
        prev_adj_face_id, prev_adj_quintant = FACE_ADJACENCY[origin.id][prev_quintant]
        prev_adj_origin = origins[prev_adj_face_id]
        prev_adj_segment, prev_adj_orientation = quintant_to_segment(prev_adj_quintant, prev_adj_origin)
        _add_neighbor(ctx, triple, prev_adj_orientation, prev_adj_origin, prev_adj_segment)

        # Vertex neighbor 2: adjacent quintant on the primary cross-face
        cross_face_id, cross_quintant = FACE_ADJACENCY[origin.id][source_quintant]
        cross_origin = origins[cross_face_id]
        next_cross_quintant = (cross_quintant + 1) % 5
        cross_segment, cross_orientation = quintant_to_segment(next_cross_quintant, cross_origin)
        _add_neighbor(ctx, triple, cross_orientation, cross_origin, cross_segment)

    result = sorted(ctx.neighbor_set)
    return result
