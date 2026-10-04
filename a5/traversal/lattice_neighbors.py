# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

from typing import List

from ..core.serialization import get_resolution, FIRST_HILBERT_RESOLUTION
from .global_neighbors import get_global_cell_neighbors
from .triple_cells import cell_ids_to_triples, for_each_lattice_neighbor, triple_cell_to_id


def get_lattice_neighbors(cell_id: int) -> List[int]:
    """
    Fast lattice-based neighbor finding over triple-space deltas: the 3
    parity-valid moves -- strict triple-lattice edge connectivity, the
    connectivity `triple_space_flood_fill` uses -- plus the edge-sharing
    neighbors across a quintant edge (see `for_each_lattice_neighbor`). Falls
    back to get_global_cell_neighbors below res 2.
    """
    resolution = get_resolution(cell_id)
    if resolution < FIRST_HILBERT_RESOLUTION:
        return get_global_cell_neighbors(cell_id, True)

    hilbert_res = resolution - FIRST_HILBERT_RESOLUTION + 1
    origin_id, quintant, x, y, z = cell_ids_to_triples([cell_id])
    result: List[int] = []

    def visit(o: int, q: int, nx: int, ny: int, nz: int) -> None:
        result.append(triple_cell_to_id(o, q, nx, ny, nz, hilbert_res, resolution))

    for_each_lattice_neighbor(origin_id, quintant, x, y, z, (1 << hilbert_res) - 1, visit)
    return result
