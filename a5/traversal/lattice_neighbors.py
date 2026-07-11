# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

from dataclasses import dataclass
from typing import List, Optional, Tuple

from ..lattice import (
    Orientation, Triple,
    s_to_triple, triple_to_s, triple_in_bounds, triple_parity,
)
from ..core.utils import Origin
from ..core.serialization import deserialize, serialize, FIRST_HILBERT_RESOLUTION
from ..core.origin import segment_to_quintant
from .global_neighbors import get_global_cell_neighbors
from .lattice_boundary import BoundaryContext, get_boundary_neighbors


@dataclass
class _LatticeSource:
    """Decoded source-cell state used by the lattice neighbor finder."""
    origin: Origin
    segment: int
    S: int
    resolution: int
    hilbert_res: int
    quintant: int
    orientation: Orientation
    triple: Triple
    max_s: int
    max_row: int


def _decode_source(cell_id: int) -> Optional[_LatticeSource]:
    """Deserialize and unpack into a _LatticeSource. Returns None below FIRST_HILBERT_RESOLUTION."""
    cell = deserialize(cell_id)
    origin, segment, S, resolution = cell['origin'], cell['segment'], cell['S'], cell['resolution']
    if resolution < FIRST_HILBERT_RESOLUTION:
        return None

    hilbert_res = resolution - FIRST_HILBERT_RESOLUTION + 1
    quintant, orientation = segment_to_quintant(segment, origin)
    triple = s_to_triple(S, hilbert_res, orientation)

    return _LatticeSource(
        origin=origin, segment=segment, S=S, resolution=resolution,
        hilbert_res=hilbert_res, quintant=quintant, orientation=orientation, triple=triple,
        max_s=4 ** hilbert_res,
        max_row=(1 << hilbert_res) - 1,
    )


def _boundary_context(src: _LatticeSource) -> BoundaryContext:
    """Build the BoundaryContext used by lattice-boundary helpers."""
    return BoundaryContext(
        triple=src.triple,
        parity=triple_parity(src.triple),
        source_quintant=src.quintant,
        origin=src.origin,
        hilbert_res=src.hilbert_res,
        max_s=src.max_s,
        max_row=src.max_row,
        resolution=src.resolution,
    )


# All 26 non-zero +/-1 moves in 3D -- vertex- and edge-sharing within-quintant candidates.
SUPERSET_DELTAS: List[Tuple[int, int, int]] = [
    (dx, dy, dz)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if not (dx == 0 and dy == 0 and dz == 0)
]

# The 3 parity-valid single-axis moves matching `triple_space_flood_fill`'s edge connectivity.
PARITY_EVEN_DELTAS: List[Tuple[int, int, int]] = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
PARITY_ODD_DELTAS: List[Tuple[int, int, int]] = [(-1, 0, 0), (0, -1, 0), (0, 0, -1)]


def get_lattice_neighbors(cell_id: int, edge_only: bool) -> List[int]:
    """
    Fast lattice-based neighbor finding. Skips is_neighbor() validation for
    within-quintant candidates; falls back to get_global_cell_neighbors below res 2.

    - edge_only=False: 26-cube +/-1 superset (may include vertex-only touchers).
      For BFS that re-validates candidates downstream (e.g. line tracing).
    - edge_only=True: 3 parity-valid moves matching `triple_space_flood_fill` --
      exact connectivity for shell-buffering the flood-fill firewall.
    """
    src = _decode_source(cell_id)
    if src is None:
        return get_global_cell_neighbors(cell_id, edge_only)

    origin = src.origin
    segment = src.segment
    S = src.S
    resolution = src.resolution
    hilbert_res = src.hilbert_res
    orientation = src.orientation
    triple = src.triple
    max_s = src.max_s
    max_row = src.max_row

    if edge_only:
        deltas = PARITY_EVEN_DELTAS if triple_parity(triple) == 0 else PARITY_ODD_DELTAS
    else:
        deltas = SUPERSET_DELTAS

    result: List[int] = []
    for dx, dy, dz in deltas:
        candidate = Triple(triple.x + dx, triple.y + dy, triple.z + dz)
        if not triple_in_bounds(candidate, max_row):
            continue
        candidate_s = triple_to_s(candidate, hilbert_res, orientation)
        if (candidate_s is not None and 0 <= candidate_s < max_s and candidate_s != S):
            result.append(serialize({
                'origin': origin, 'segment': segment,
                'S': candidate_s, 'resolution': resolution,
            }))

    # Strict lattice connectivity (edge_only) doesn't traverse the [-max_row, max_row, 0]
    # vertex corner, so we skip it there too -- keeping the firewall topology tight.
    for c in get_boundary_neighbors(_boundary_context(src), edge_only, edge_only):
        result.append(c)
    return result
