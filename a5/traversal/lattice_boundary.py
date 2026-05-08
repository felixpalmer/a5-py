# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

from typing import List, Tuple
from dataclasses import dataclass

from ..lattice import Orientation, Triple, triple_to_s, triple_in_bounds
from ..core.utils import Origin
from ..core.serialization import serialize
from ..core.origin import quintant_to_segment, origins
from ..core.face_adjacency import FACE_ADJACENCY

# Neighbor delta: (dx, dy, dz, is_edge_sharing)
NeighborDelta = Tuple[int, int, int, bool]

# Cross-quintant left-edge deltas (source z=0), indexed by parity * 2 + (y_odd ? 1 : 0).
# Applied to the swapped base triple [0, y, x] in the previous quintant.
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

# Cross-quintant right-edge deltas (source x=0), indexed by parity * 2 + (y_odd ? 1 : 0).
# Applied to the swapped base triple [z, y, 0] in the next quintant.
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

# Cross-face base-edge deltas (source y=maxRow), indexed by parity.
# Applied to the mirrored position [z, maxRow, x] on the adjacent face.
CROSS_FACE_DELTAS: List[List[NeighborDelta]] = [
    # parity=0
    [(0, 0, 0, True), (1, 0, 0, True), (1, 0, -1, False)],
    # parity=1
    [(0, 0, -1, True), (0, 0, 0, False)],
]


@dataclass
class BoundaryContext:
    """Source-cell context shared by all boundary-neighbor cases."""
    triple: Triple
    parity: int
    source_quintant: int
    origin: Origin
    hilbert_res: int
    max_s: int
    max_row: int
    resolution: int


def _push_triple(
    out: List[int], triple: Triple, orientation: Orientation,
    origin: Origin, segment: int, ctx: BoundaryContext,
) -> None:
    """If the triple maps to a valid cell, append its cell ID to out."""
    if not triple_in_bounds(triple, ctx.max_row):
        return
    s = triple_to_s(triple, ctx.hilbert_res, orientation)
    if s is None or s < 0 or s >= ctx.max_s:
        return
    out.append(serialize({
        'origin': origin, 'segment': segment,
        'S': s, 'resolution': ctx.resolution,
    }))


def _push_deltas(
    out: List[int], base: Triple, deltas: List[NeighborDelta], edge_only: bool,
    orientation: Orientation, origin: Origin, segment: int, ctx: BoundaryContext,
) -> None:
    """Apply a delta table to a base triple, appending each valid cell."""
    for dx, dy, dz, is_edge in deltas:
        if edge_only and not is_edge:
            continue
        _push_triple(out, Triple(base.x + dx, base.y + dy, base.z + dz),
                     orientation, origin, segment, ctx)


def get_boundary_neighbors(
    ctx: BoundaryContext,
    edge_only: bool,
) -> List[int]:
    """
    Return every neighbor that lies outside the source cell's quintant: cross-quintant
    lateral edges, cross-face base edge, apex (face center), and the [-maxRow, maxRow, 0]
    vertex corner. The within-quintant +/-1 candidates are NOT covered here -- callers
    generate those directly.

    The result may contain duplicates and the order is not stable; callers
    deduplicate (via set) or accept duplicates if their downstream pipeline tolerates them.
    """
    out: List[int] = []
    triple = ctx.triple
    parity = ctx.parity
    source_quintant = ctx.source_quintant
    origin = ctx.origin
    max_row = ctx.max_row
    y_odd = triple.y % 2 != 0
    delta_index = parity * 2 + (1 if y_odd else 0)

    # Left edge (z=0): neighbor in previous quintant at swapped [0, y, x]
    if triple.z == 0:
        target_quintant = (source_quintant - 1 + 5) % 5
        segment, orientation = quintant_to_segment(target_quintant, origin)
        _push_deltas(out, Triple(0, triple.y, triple.x), LEFT_EDGE_DELTAS[delta_index], edge_only,
                     orientation, origin, segment, ctx)

    # Right edge (x=0): neighbor in next quintant at swapped [z, y, 0]
    if triple.x == 0:
        target_quintant = (source_quintant + 1) % 5
        segment, orientation = quintant_to_segment(target_quintant, origin)
        _push_deltas(out, Triple(triple.z, triple.y, 0), RIGHT_EDGE_DELTAS[delta_index], edge_only,
                     orientation, origin, segment, ctx)

    # Base edge (y=maxRow): neighbor on adjacent face at mirrored [z, maxRow, x]
    if triple.y == max_row:
        adj_face_id, adj_quintant = FACE_ADJACENCY[origin.id][source_quintant]
        adj_origin = origins[adj_face_id]
        segment, orientation = quintant_to_segment(adj_quintant, adj_origin)
        _push_deltas(out, Triple(triple.z, max_row, triple.x), CROSS_FACE_DELTAS[parity], edge_only,
                     orientation, adj_origin, segment, ctx)

    # Apex [0,0,0]: cells from all 5 quintants meet at the face center
    if triple.x == 0 and triple.y == 0 and triple.z == 0:
        for q in range(5):
            if q == source_quintant:
                continue
            distance = min((q - source_quintant + 5) % 5, (source_quintant - q + 5) % 5)
            if edge_only and distance != 1:
                continue
            segment, orientation = quintant_to_segment(q, origin)
            _push_triple(out, triple, orientation, origin, segment, ctx)

    # Base-left corner [-maxRow, maxRow, 0]: 3 dodecahedron faces meet at this vertex.
    # The symmetric base-right corner is implicitly covered: its cross-quintant and
    # cross-face paths land on the [-maxRow, maxRow, 0] cell of neighboring quintants.
    if triple.x == -max_row and triple.y == max_row and triple.z == 0:
        # Vertex neighbor 1: across the previous quintant's base edge
        prev_quintant = (source_quintant - 1 + 5) % 5
        prev_adj_face_id, prev_adj_quintant = FACE_ADJACENCY[origin.id][prev_quintant]
        prev_adj_origin = origins[prev_adj_face_id]
        prev_adj_segment, prev_adj_orientation = quintant_to_segment(prev_adj_quintant, prev_adj_origin)
        _push_triple(out, triple, prev_adj_orientation, prev_adj_origin, prev_adj_segment, ctx)

        # Vertex neighbor 2: adjacent quintant on the primary cross-face
        cross_face_id, cross_quintant = FACE_ADJACENCY[origin.id][source_quintant]
        cross_origin = origins[cross_face_id]
        next_cross_quintant = (cross_quintant + 1) % 5
        cross_segment, cross_orientation = quintant_to_segment(next_cross_quintant, cross_origin)
        _push_triple(out, triple, cross_orientation, cross_origin, cross_segment, ctx)

    return out
