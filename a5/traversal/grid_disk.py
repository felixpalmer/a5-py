# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

from typing import List, Set

from ..collections.slot_runs import compact_cells, to_covering
from ..core.serialization import deserialize, serialize, FIRST_HILBERT_RESOLUTION
from ..core.origin import origins
from ..core.face_adjacency import walk_faces
from .triple_cells import cell_ids_to_triples, for_each_triple_neighbor, triple_cell_key, triple_cell_to_id


class _Ring:
    """One BFS ring: its dedup keys, and its cells as flat (origin_id, quintant, x, y, z)."""

    __slots__ = ('keys', 'cells')

    def __init__(self) -> None:
        self.keys: Set[int] = set()
        self.cells: List[int] = []


def _add_cell(nxt: _Ring, prev: _Ring, current: _Ring,
              origin_id: int, quintant: int, x: int, y: int, z: int) -> None:
    """Add a cell to `nxt` unless it is already in one of the three live rings."""
    key = triple_cell_key(origin_id, quintant, x, y, z)
    if key in prev.keys or key in current.keys or key in nxt.keys:
        return
    nxt.keys.add(key)
    nxt.cells.extend((origin_id, quintant, x, y, z))


def _push_cell_ids(out: List[int], cells: List[int], hilbert_res: int, resolution: int) -> None:
    """Encode a ring's cells as cell IDs, appending them to `out`."""
    for c in range(0, len(cells), 5):
        out.append(triple_cell_to_id(cells[c], cells[c + 1], cells[c + 2], cells[c + 3], cells[c + 4],
                                     hilbert_res, resolution))


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
    cell = deserialize(cell_id)
    origin = cell['origin']
    resolution = cell['resolution']
    if k == 0:
        return to_covering([cell_id], resolution)
    if resolution == 0:
        # The cells are the 12 dodecahedron faces
        faces = walk_faces([origin.id], lambda face: True, k)
        return to_covering(
            [serialize({'origin': origins[face], 'segment': 0, 'S': 0, 'resolution': 0}) for face in faces], 0
        )
    hilbert_res = resolution - FIRST_HILBERT_RESOLUTION + 1
    max_row = (1 << hilbert_res) - 1
    seed = cell_ids_to_triples([cell_id])

    # The seed is `cell_id` already, so it goes straight to the output
    interior: List[int] = [cell_id]
    prev_frontier = _Ring()
    frontier = _Ring()
    _add_cell(frontier, prev_frontier, prev_frontier, *seed)

    for ring in range(1, k + 1):
        next_frontier = _Ring()

        def visit(origin_id: int, q: int, x: int, y: int, z: int) -> None:
            _add_cell(next_frontier, prev_frontier, frontier, origin_id, q, x, y, z)

        cells = frontier.cells
        for c in range(0, len(cells), 5):
            for_each_triple_neighbor(cells[c], cells[c + 1], cells[c + 2], cells[c + 3], cells[c + 4],
                                     max_row, edge_only, visit)

        # The seed ring is expanded; drop its cell so it isn't encoded again (its key stays)
        if ring == 1:
            frontier.cells.clear()

        # Evict prev_frontier -- these cells are >=2 rings behind the new frontier
        # and can never be re-discovered by BFS
        _push_cell_ids(interior, prev_frontier.cells, hilbert_res, resolution)

        # Progressively compact interior to reduce memory pressure
        if len(interior) > 100:
            interior = compact_cells(interior)

        prev_frontier = frontier
        frontier = next_frontier

    # Merge remaining boundary rings with compacted interior
    _push_cell_ids(interior, prev_frontier.cells, hilbert_res, resolution)
    _push_cell_ids(interior, frontier.cells, hilbert_res, resolution)

    return to_covering(interior, resolution)


def grid_disk(cell_id: int, k: int) -> List[int]:
    """
    Compute the grid disk of edge-sharing neighbors within k hops.
    Returns compacted cell IDs including the center cell, sorted in curve
    order, then a compaction marker recording the resolution.

    This matches H3's gridDisk semantics -- only edge-sharing neighbors are
    followed. For A5 pentagons, each cell has exactly 5 edge neighbors.
    """
    return _grid_disk(cell_id, k, True)


def grid_disk_vertex(cell_id: int, k: int) -> List[int]:
    """
    Compute the grid disk of all neighbors (edge + vertex sharing) within k hops.
    Returns compacted cell IDs including the center cell, sorted in curve
    order, then a compaction marker recording the resolution.

    This is an A5 extension -- pentagons have both edge-sharing (5) and
    vertex-only-sharing neighbors (1-3), giving 6-8 total neighbors per cell.
    """
    return _grid_disk(cell_id, k, False)
