# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

from typing import Dict, List, Optional, Set, Union

from ..lattice import Triple, triple_in_bounds
from ..core.serialization import FIRST_HILBERT_RESOLUTION
from .triple_cells import triple_cell_key

# Flood state, reusable across calls at one resolution: the keys of every cell visited so far
FloodState = Dict[str, Set[int]]


def triple_space_flood_fill(
    firewall: Union[List[int], Dict],
    seeds: List[int],
    resolution: int,
    max_layers: Optional[int] = None,
) -> Dict:
    """
    Triple-space flood fill over the 3 parity-valid lattice moves. Those never
    cross a quintant edge, so each quintant floods independently. All cells --
    firewall, seeds, the returned frontier -- are flat (origin_id, quintant, x,
    y, z), so nothing is decoded; discovered cells are encoded once, on output.

    Args:
        firewall: Cells the flood may not enter, or a reused {'state', 'delta'}
            from a previous call (state reused, 'delta' cells joining its firewall).
        seeds: BFS seeds. Always added to the frontier, even if already visited --
            reusing state with the same seeds restarts BFS.
        max_layers: Max BFS layers; None = run to convergence.

    Returns:
        {'interior', 'frontier', 'state'}: the cells discovered by this call
        (seeds excluded), the final frontier, and the state for a follow-up call.
    """
    hilbert_res = resolution - FIRST_HILBERT_RESOLUTION + 1
    max_row = (1 << hilbert_res) - 1

    if isinstance(firewall, list):
        state: FloodState = {'visited': set()}
        blocked = firewall
    else:
        state = firewall['state']
        blocked = firewall['delta']
    visited = state['visited']
    for c in range(0, len(blocked), 5):
        visited.add(triple_cell_key(blocked[c], blocked[c + 1], blocked[c + 2], blocked[c + 3], blocked[c + 4]))
    for c in range(0, len(seeds), 5):
        visited.add(triple_cell_key(seeds[c], seeds[c + 1], seeds[c + 2], seeds[c + 3], seeds[c + 4]))

    discovered: List[int] = []
    frontier = seeds
    layers = 0
    while frontier and (max_layers is None or layers < max_layers):
        next_frontier: List[int] = []

        def add(origin_id: int, quintant: int, x: int, y: int, z: int) -> None:
            if not triple_in_bounds(Triple(x, y, z), max_row):
                return
            key = triple_cell_key(origin_id, quintant, x, y, z)
            if key in visited:
                return
            visited.add(key)
            discovered.extend((origin_id, quintant, x, y, z))
            next_frontier.extend((origin_id, quintant, x, y, z))

        for c in range(0, len(frontier), 5):
            origin_id, quintant, x, y, z = frontier[c:c + 5]
            # +1 on one axis from a parity 0 triple, -1 from parity 1
            step = 1 if x + y + z == 0 else -1
            add(origin_id, quintant, x + step, y, z)
            add(origin_id, quintant, x, y + step, z)
            add(origin_id, quintant, x, y, z + step)
        frontier = next_frontier
        layers += 1

    return {'interior': discovered, 'frontier': frontier, 'state': state}
