# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

from typing import List, Tuple

from ..lattice import Triple, triple_flavor, triple_in_bounds
from ..core.utils import Origin
from ..core.face_adjacency import FACE_ADJACENCY, seam_triple
from .neighbors import NEIGHBOR_DELTAS

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

def _push_triple(
    out: List[int], x: int, y: int, z: int, origin_id: int, quintant: int, max_row: int,
) -> None:
    """If the triple is a valid cell, append it to out as (origin_id, quintant, x, y, z)."""
    if not triple_in_bounds(Triple(x, y, z), max_row):
        return
    out.extend((origin_id, quintant, x, y, z))


def _push_deltas(
    out: List[int], base: Triple, deltas: List[NeighborDelta], edge_only: bool,
    origin_id: int, quintant: int, max_row: int,
) -> None:
    """Apply a delta table to a base triple, appending each valid cell."""
    for dx, dy, dz, is_edge in deltas:
        if edge_only and not is_edge:
            continue
        _push_triple(out, base.x + dx, base.y + dy, base.z + dz, origin_id, quintant, max_row)


def get_boundary_neighbor_triples(
    triple: Triple,
    parity: int,
    source_quintant: int,
    origin: Origin,
    max_row: int,
    edge_only: bool,
    skip_corners: bool,
    out: List[int],
) -> None:
    """
    Every neighbor that lies outside the source cell's quintant, appended to
    `out` as flat (origin_id, quintant, x, y, z) quintuples: cross-quintant
    lateral edges, cross-face base edge, apex (face center), and (when not
    `skip_corners`) the [-maxRow, maxRow, 0] vertex corner. The within-quintant
    +/-1 candidates are NOT covered here -- callers generate those directly.

    Only cells on a quintant edge (x = 0, z = 0 or y = max_row) have any. The
    result may contain duplicates; callers deduplicate.

    Args:
        edge_only: drop apex non-adjacent quintants and other vertex-only neighbors
        skip_corners: drop the [-maxRow, maxRow, 0] corner -- used when the caller's
                      connectivity (e.g. lattice +/-1 moves) doesn't traverse that vertex
    """
    y_odd = triple.y % 2 != 0
    delta_index = parity * 2 + (1 if y_odd else 0)

    # Left edge (z=0): neighbor in previous quintant at swapped [0, y, x]
    if triple.z == 0:
        target_quintant = (source_quintant - 1 + 5) % 5
        _push_deltas(out, Triple(0, triple.y, triple.x), LEFT_EDGE_DELTAS[delta_index], edge_only,
                     origin.id, target_quintant, max_row)

    # Right edge (x=0): neighbor in next quintant at swapped [z, y, 0]
    if triple.x == 0:
        target_quintant = (source_quintant + 1) % 5
        _push_deltas(out, Triple(triple.z, triple.y, 0), RIGHT_EDGE_DELTAS[delta_index], edge_only,
                     origin.id, target_quintant, max_row)

    # Base edge (y=maxRow): across the face seam the lattice continues, so the
    # neighbors on the adjacent face are those of the cell's image there
    if triple.y == max_row:
        adj_face_id, adj_quintant = FACE_ADJACENCY[origin.id][source_quintant]
        image = seam_triple(triple, max_row)
        # The image's pentagon is the cell's turned half-way round: its flavor's parity bit flipped
        deltas = NEIGHBOR_DELTAS[triple_flavor(triple, max_row) ^ 1]
        delta_list = deltas.edge if edge_only else deltas.all
        for i in range(len(delta_list)):
            d = delta_list[i]
            # Only steps back towards the seam (dy < 0) can land inside the neighbor quintant
            if d.y < 0:
                _push_triple(out, image.x + d.x, image.y + d.y, image.z + d.z, adj_face_id, adj_quintant, max_row)

    # Apex [0,0,0]: cells from all 5 quintants meet at the face center
    if triple.x == 0 and triple.y == 0 and triple.z == 0:
        for q in range(5):
            if q == source_quintant:
                continue
            distance = min((q - source_quintant + 5) % 5, (source_quintant - q + 5) % 5)
            if edge_only and distance != 1:
                continue
            _push_triple(out, 0, 0, 0, origin.id, q, max_row)

    # Base-left corner [-maxRow, maxRow, 0]: 3 dodecahedron faces meet at this vertex.
    # The symmetric base-right corner is implicitly covered: its cross-quintant and
    # cross-face paths land on the [-maxRow, maxRow, 0] cell of neighboring quintants.
    if not skip_corners and triple.x == -max_row and triple.y == max_row and triple.z == 0:
        # Vertex neighbor 1: across the previous quintant's base edge
        prev_quintant = (source_quintant - 1 + 5) % 5
        prev_adj_face_id, prev_adj_quintant = FACE_ADJACENCY[origin.id][prev_quintant]
        _push_triple(out, triple.x, triple.y, triple.z, prev_adj_face_id, prev_adj_quintant, max_row)

        # Vertex neighbor 2: adjacent quintant on the primary cross-face
        cross_face_id, cross_quintant = FACE_ADJACENCY[origin.id][source_quintant]
        _push_triple(out, triple.x, triple.y, triple.z, cross_face_id, (cross_quintant + 1) % 5, max_row)
