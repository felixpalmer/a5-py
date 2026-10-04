# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

from typing import List, Set

from ..lattice import Triple, s_to_triple, triple_flavor, triple_in_bounds, triple_to_s
from ..core.compact import compact
from ..core.serialization import deserialize, serialize, FIRST_HILBERT_RESOLUTION
from ..core.origin import origins, quintant_to_segment, segment_to_quintant
from ..core.face_adjacency import FACE_ADJACENCY
from .lattice_boundary import get_boundary_neighbor_triples
from .neighbors import NEIGHBOR_DELTAS

# Cells are deduplicated by one integer key: the quintant (origin.id * 5 +
# quintant, < 60), parity, and the low _KEY_BITS bits of -x and -z (y follows).
# Up to Hilbert resolution 21 the coordinates fit whole; above it, two cells of
# one quintant share a key only if they are 2^22 rows apart, and a disk holding
# both would need k ~ 2^21 (~10^12 cells) -- far past what fits in memory.
_KEY_BITS = 22
_KEY_MASK = (1 << _KEY_BITS) - 1
_KEY_SIDE = 1 << _KEY_BITS

# Segment and curve orientation of each of the 60 quintants, by origin.id * 5 + quintant
_QUINTANT_SEGMENTS = [quintant_to_segment(q, origin) for origin in origins for q in range(5)]


class _Ring:
    """One BFS ring: its dedup keys, and its cells as flat (origin_id, quintant, x, y, z)."""

    __slots__ = ('keys', 'cells')

    def __init__(self) -> None:
        self.keys: Set[int] = set()
        self.cells: List[int] = []


def _add_cell(nxt: _Ring, prev: _Ring, current: _Ring,
              origin_id: int, quintant: int, x: int, y: int, z: int) -> None:
    """Add a cell to `nxt` unless it is already in one of the three live rings."""
    key = ((((-x) & _KEY_MASK) * _KEY_SIDE + ((-z) & _KEY_MASK)) * 2 + x + y + z
           + (origin_id * 5 + quintant) * 2 * _KEY_SIDE * _KEY_SIDE)
    if key in prev.keys or key in current.keys or key in nxt.keys:
        return
    nxt.keys.add(key)
    nxt.cells.extend((origin_id, quintant, x, y, z))


def _push_cell_ids(out: List[int], cells: List[int], hilbert_res: int, resolution: int) -> None:
    """Encode a ring's cells as cell IDs, appending them to `out`."""
    for c in range(0, len(cells), 5):
        segment, orientation = _QUINTANT_SEGMENTS[cells[c] * 5 + cells[c + 1]]
        s = triple_to_s(Triple(cells[c + 2], cells[c + 3], cells[c + 4]), hilbert_res, orientation)
        out.append(serialize({'origin': origins[cells[c]], 'segment': segment, 'S': s, 'resolution': resolution}))


def _grid_disk_faces(origin_id: int, k: int) -> List[int]:
    """Resolution 0: the cells are the 12 dodecahedron faces, adjacent across their edges."""
    disk = {origin_id}
    ring = 0
    while ring < k and len(disk) < 12:
        for face_id in list(disk):
            for q in range(5):
                disk.add(FACE_ADJACENCY[face_id][q][0])
        ring += 1
    return compact([serialize({'origin': origins[i], 'segment': 0, 'S': 0, 'resolution': 0}) for i in disk])


def _grid_disk(cell_id: int, k: int, edge_only: bool) -> List[int]:
    """
    BFS grid disk in triple space, with progressive compaction.

    Neighbors come from the per-flavor triple deltas, plus the boundary delta
    tables for cells on a quintant edge, so no cell is decoded and each is
    encoded exactly once, when it leaves the window.

    Uses a sliding-window dedup approach: only the previous and current frontier
    rings are kept in memory for deduplication (BFS guarantees cells >=2 rings
    behind the frontier can never be re-discovered). Evicted interior cells are
    periodically compacted to reduce memory pressure.
    """
    if k == 0:
        return [cell_id]
    cell = deserialize(cell_id)
    origin = cell['origin']
    resolution = cell['resolution']
    if resolution == 0:
        return _grid_disk_faces(origin.id, k)
    hilbert_res = resolution - FIRST_HILBERT_RESOLUTION + 1
    max_row = (1 << hilbert_res) - 1
    quintant, orientation = segment_to_quintant(cell['segment'], origin)
    seed = s_to_triple(cell['S'], hilbert_res, orientation)

    # The seed is `cell_id` already, so it goes straight to the output
    interior: List[int] = [cell_id]
    prev_frontier = _Ring()
    frontier = _Ring()
    _add_cell(frontier, prev_frontier, prev_frontier, origin.id, quintant, seed.x, seed.y, seed.z)
    boundary: List[int] = []

    for ring in range(1, k + 1):
        next_frontier = _Ring()
        cells = frontier.cells
        for c in range(0, len(cells), 5):
            origin_id = cells[c]
            q = cells[c + 1]
            x = cells[c + 2]
            y = cells[c + 3]
            z = cells[c + 4]
            triple = Triple(x, y, z)

            # Within the quintant: the fixed per-flavor deltas
            flavor = triple_flavor(triple, max_row)
            deltas = NEIGHBOR_DELTAS[flavor].edge if edge_only else NEIGHBOR_DELTAS[flavor].all
            for d in deltas:
                neighbor = Triple(x + d.x, y + d.y, z + d.z)
                if not triple_in_bounds(neighbor, max_row):
                    continue
                _add_cell(next_frontier, prev_frontier, frontier, origin_id, q, neighbor.x, neighbor.y, neighbor.z)

            # Across a quintant edge: the boundary delta tables
            if x == 0 or z == 0 or y == max_row:
                boundary.clear()
                get_boundary_neighbor_triples(triple, x + y + z, q, origins[origin_id], max_row,
                                              edge_only, False, boundary)
                for i in range(0, len(boundary), 5):
                    _add_cell(next_frontier, prev_frontier, frontier, boundary[i], boundary[i + 1],
                              boundary[i + 2], boundary[i + 3], boundary[i + 4])

        # The seed ring is expanded; drop its cell so it isn't encoded again (its key stays)
        if ring == 1:
            frontier.cells.clear()

        # Evict prev_frontier -- these cells are >=2 rings behind the new frontier
        # and can never be re-discovered by BFS
        _push_cell_ids(interior, prev_frontier.cells, hilbert_res, resolution)

        # Progressively compact interior to reduce memory pressure
        if len(interior) > 100:
            interior = list(compact(interior))

        prev_frontier = frontier
        frontier = next_frontier

    # Merge remaining boundary rings with compacted interior
    _push_cell_ids(interior, prev_frontier.cells, hilbert_res, resolution)
    _push_cell_ids(interior, frontier.cells, hilbert_res, resolution)

    return compact(interior)


def grid_disk(cell_id: int, k: int) -> List[int]:
    """
    Compute the grid disk of edge-sharing neighbors within k hops.
    Returns a sorted, compacted list of cell IDs including the center cell.

    This matches H3's gridDisk semantics -- only edge-sharing neighbors are
    followed. For A5 pentagons, each cell has exactly 5 edge neighbors.
    """
    return _grid_disk(cell_id, k, True)


def grid_disk_vertex(cell_id: int, k: int) -> List[int]:
    """
    Compute the grid disk of all neighbors (edge + vertex sharing) within k hops.
    Returns a sorted, compacted list of cell IDs including the center cell.

    This is an A5 extension -- pentagons have both edge-sharing (5) and
    vertex-only-sharing neighbors (1-3), giving 6-8 total neighbors per cell.
    """
    return _grid_disk(cell_id, k, False)
