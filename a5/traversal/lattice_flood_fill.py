# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Union

from ..lattice import (
    Orientation, s_to_triple, triple_to_s,
)
from ..core.utils import Origin
from ..core.serialization import deserialize, serialize, FIRST_HILBERT_RESOLUTION
from ..core.origin import segment_to_quintant


@dataclass
class _QuintantCtx:
    """Per-quintant context needed to convert triples back to cell IDs."""
    origin: Origin
    segment: int
    orientation: Orientation


@dataclass
class QuintantState:
    """Per-quintant packed BFS state. Reusable across phases at the same resolution."""
    ctx: _QuintantCtx
    visited: Set[int] = field(default_factory=set)
    frontier: List[int] = field(default_factory=list)


# Packed flood-fill state, indexed by quintant.
PackedFloodState = Dict[int, QuintantState]


def _pack_triple_key(x: int, y: int, parity: int, max_row: int, y_stride: int) -> int:
    """
    Pack a triple as a single number key for fast Set lookup.
    Encoding: (x + max_row) * y_stride + y * 2 + parity, where parity = (x+y+z) in {0,1}.
    """
    return (x + max_row) * y_stride + y * 2 + parity


def _unpack_triple_key(key: int, max_row: int, y_stride: int):
    """Inverse of _pack_triple_key -- recover (x, y, z, parity) from a packed key."""
    parity = key % 2
    y_part = (key - parity) % y_stride
    y = y_part // 2
    x = (key - y_part - parity) // y_stride - max_row
    z = parity - x - y
    return x, y, z, parity


def _packed_key_to_cell_id(
    key: int, ctx: _QuintantCtx, hilbert_res: int, max_row: int,
    y_stride: int, max_s: int, resolution: int,
) -> Optional[int]:
    """Convert a packed triple key back to a cell ID, or None if it doesn't map to a valid cell."""
    from ..lattice.triple import Triple
    x, y, z, _ = _unpack_triple_key(key, max_row, y_stride)
    s = triple_to_s(Triple(x, y, z), hilbert_res, ctx.orientation)
    if s is None or s < 0 or s >= max_s:
        return None
    return serialize({
        'origin': ctx.origin, 'segment': ctx.segment,
        'S': s, 'resolution': resolution,
    })


def _cell_to_quintant_key(cell_id: int, hilbert_res: int, max_row: int, y_stride: int):
    """Convert a cell ID into its quintant context and packed triple key."""
    cell = deserialize(cell_id)
    origin, segment, S = cell['origin'], cell['segment'], cell['S']
    quintant, orientation = segment_to_quintant(segment, origin)
    triple = s_to_triple(S, hilbert_res, orientation)
    parity = triple.x + triple.y + triple.z  # 0 or 1
    return (
        origin.id * 60 + segment,
        _pack_triple_key(triple.x, triple.y, parity, max_row, y_stride),
        _QuintantCtx(origin=origin, segment=segment, orientation=orientation),
    )


def triple_space_flood_fill(
    firewall: Union[Set[int], dict],
    seed_cell_ids: List[int],
    resolution: int,
    max_layers: Optional[int] = None,
) -> dict:
    """
    Triple-space flood fill in packed integer coordinates -- no per-step bigint ops.
    Uses the 3 parity-valid +/-1 moves; since those never cross quintant boundaries,
    each quintant is flooded independently.

    Args:
        firewall: A bigint firewall set (mutated to include discoveries) or
                  a reused {'state': PackedFloodState, 'delta': Iterable[int]}
                  from a previous call (state reused, only delta cells converted).
        seed_cell_ids: BFS seeds. Always added to the frontier, even if already
                       visited -- reusing state with the same seeds restarts BFS.
        max_layers: Max BFS layers; None = run to convergence.

    Returns:
        Dict with 'interior_cells', 'frontier_cell_ids', and 'state' keys.
    """
    hilbert_res = resolution - FIRST_HILBERT_RESOLUTION + 1
    max_row = (1 << hilbert_res) - 1
    y_stride = (max_row + 1) * 2
    max_s = 4 ** hilbert_res

    reusing = not isinstance(firewall, set)
    quintants: PackedFloodState

    # Discovered cells per quintant for THIS call (excludes prior-call discoveries)
    discovered_per_q: Dict[int, List[int]] = {}

    def get_or_create_q(quintant_idx: int, ctx: _QuintantCtx) -> QuintantState:
        q = quintants.get(quintant_idx)
        if q is None:
            q = QuintantState(ctx=ctx)
            quintants[quintant_idx] = q
        return q

    if isinstance(firewall, set):
        quintants = {}
        for cell_id in firewall:
            quintant_idx, key, ctx = _cell_to_quintant_key(cell_id, hilbert_res, max_row, y_stride)
            get_or_create_q(quintant_idx, ctx).visited.add(key)
    else:
        quintants = firewall['state']
        # Stale frontier from prior call -- clear so seeds drive this BFS
        for q in quintants.values():
            q.frontier = []
        for cell_id in firewall['delta']:
            quintant_idx, key, ctx = _cell_to_quintant_key(cell_id, hilbert_res, max_row, y_stride)
            get_or_create_q(quintant_idx, ctx).visited.add(key)

    # Seed the frontier
    for cell_id in seed_cell_ids:
        quintant_idx, key, ctx = _cell_to_quintant_key(cell_id, hilbert_res, max_row, y_stride)
        q = get_or_create_q(quintant_idx, ctx)
        q.visited.add(key)
        q.frontier.append(key)

    # 3 parity-valid moves and bounds checks inlined for the hot loop.
    layers = 0
    has_work = True
    while has_work and (max_layers is None or layers < max_layers):
        has_work = False
        for q_idx, q in quintants.items():
            if not q.frontier:
                continue
            discovered = discovered_per_q.get(q_idx)
            if discovered is None:
                discovered = []
                discovered_per_q[q_idx] = discovered
            next_frontier: List[int] = []
            for key in q.frontier:
                parity = key % 2
                y_part = (key - parity) % y_stride
                y = y_part // 2
                x = (key - y_part - parity) // y_stride - max_row
                step = 1 if parity == 0 else -1
                new_parity = 1 - parity
                y_limit = y - new_parity

                # Move in x: triple becomes (x+step, y, z); z = parity - x - y is unchanged
                nx = x + step
                nz_x = parity - x - y
                if nx <= 0 and nz_x <= 0 and nx >= -y_limit and nz_x >= -y_limit:
                    nk = (nx + max_row) * y_stride + y * 2 + new_parity
                    if nk not in q.visited:
                        q.visited.add(nk)
                        discovered.append(nk)
                        next_frontier.append(nk)

                # Move in y: triple becomes (x, y+step, z); z is unchanged
                ny = y + step
                nz_y = parity - x - y
                ny_limit = ny - new_parity
                if 0 <= ny <= max_row and nz_y <= 0 and x >= -ny_limit and nz_y >= -ny_limit:
                    nk = (x + max_row) * y_stride + ny * 2 + new_parity
                    if nk not in q.visited:
                        q.visited.add(nk)
                        discovered.append(nk)
                        next_frontier.append(nk)

                # Move in z: triple becomes (x, y, z+step); the packed key shape (x, y, parity)
                # is identical to the x and y moves' starting point apart from parity flip.
                z = parity - x - y
                nz = z + step
                if nz <= 0 and x >= -y_limit and nz >= -y_limit:
                    nk = (x + max_row) * y_stride + y * 2 + new_parity
                    if nk not in q.visited:
                        q.visited.add(nk)
                        discovered.append(nk)
                        next_frontier.append(nk)
            q.frontier = next_frontier
            if next_frontier:
                has_work = True
        layers += 1

    # Convert results back to cell IDs
    interior_cells: List[int] = []
    frontier_cell_ids: List[int] = []
    bigint_firewall = firewall if (not reusing and isinstance(firewall, set)) else None

    for q_idx, q in quintants.items():
        discovered = discovered_per_q.get(q_idx)
        if discovered:
            for key in discovered:
                cell_id = _packed_key_to_cell_id(key, q.ctx, hilbert_res, max_row, y_stride, max_s, resolution)
                if cell_id is not None:
                    interior_cells.append(cell_id)
                    if bigint_firewall is not None:
                        bigint_firewall.add(cell_id)
        for key in q.frontier:
            cell_id = _packed_key_to_cell_id(key, q.ctx, hilbert_res, max_row, y_stride, max_s, resolution)
            if cell_id is not None:
                frontier_cell_ids.append(cell_id)

    return {
        'interior_cells': interior_cells,
        'frontier_cell_ids': frontier_cell_ids,
        'state': quintants,
    }
