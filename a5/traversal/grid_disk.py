# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

from typing import List
from .global_neighbors import get_global_cell_neighbors
from ..core.compact import compact


def _grid_disk_bfs(cell_id: int, k: int, edge_only: bool) -> List[int]:
    """
    BFS grid disk with progressive compaction.

    Uses a sliding-window dedup approach: only the previous and current frontier
    rings are kept in memory for deduplication.
    """
    if k == 0:
        return [cell_id]

    interior: List[int] = []
    prev_frontier: set = set()
    frontier: set = {cell_id}

    for ring in range(1, k + 1):
        next_frontier: set = set()
        for cid in frontier:
            for neighbor in get_global_cell_neighbors(cid, edge_only=edge_only):
                if neighbor not in prev_frontier and neighbor not in frontier and neighbor not in next_frontier:
                    next_frontier.add(neighbor)

        # Evict prevFrontier -- these cells are >=2 rings behind the new frontier
        for cid in prev_frontier:
            interior.append(cid)

        # Progressively compact interior to reduce memory pressure
        if len(interior) > 100:
            interior = list(compact(interior))

        prev_frontier = frontier
        frontier = next_frontier

    # Merge remaining boundary rings with compacted interior
    for cid in prev_frontier:
        interior.append(cid)
    for cid in frontier:
        interior.append(cid)

    return compact(interior)


def grid_disk(cell_id: int, k: int) -> List[int]:
    """
    Compute the grid disk of edge-sharing neighbors within k hops.
    Returns a sorted, compacted list of cell IDs including the center cell.

    This matches H3's gridDisk semantics -- only edge-sharing neighbors are
    followed. For A5 pentagons, each cell has exactly 5 edge neighbors.
    """
    return _grid_disk_bfs(cell_id, k, True)


def grid_disk_vertex(cell_id: int, k: int) -> List[int]:
    """
    Compute the grid disk of all neighbors (edge + vertex sharing) within k hops.
    Returns a sorted, compacted list of cell IDs including the center cell.

    This is an A5 extension -- pentagons have both edge-sharing (5) and
    vertex-only-sharing neighbors (1-3), giving 6-8 total neighbors per cell.
    """
    return _grid_disk_bfs(cell_id, k, False)
