# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

from dataclasses import dataclass
from typing import List, Optional

from ..lattice import (
    Orientation, Triple,
    s_to_anchor, anchor_to_triple, triple_to_s, triple_in_bounds, triple_parity,
)
from ..core.utils import Origin
from ..core.serialization import deserialize, serialize, FIRST_HILBERT_RESOLUTION
from ..core.origin import segment_to_quintant
from .global_neighbors import get_global_cell_neighbors
from .lattice_boundary import BoundaryContext, get_boundary_neighbors


@dataclass
class _LatticeSource:
    """Source-cell state used by the lattice neighbor finder."""
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
    anchor = s_to_anchor(S, hilbert_res, orientation)
    triple = anchor_to_triple(anchor)

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


def get_lattice_neighbors(cell_id: int, edge_only: bool) -> List[int]:
    """
    Fast lattice-based neighbor finding for BFS in line tracing.

    Unlike get_global_cell_neighbors, this skips is_neighbor() validation for
    within-quintant candidates. The result is a SUPERSET of true neighbors --
    it may include a few extra cells that share only a vertex point (not an edge).

    This is safe for BFS contexts where candidates are validated by
    cell_intersects_segment -- false positives just fail that check.

    For res < 2, falls back to get_global_cell_neighbors (rare).

    Args:
        edge_only: If True, restrict to Manhattan distance <= 2 (edge-sharing candidates)
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

    result: List[int] = []

    # Within-quintant: enumerate the 26-cube of +/-1 deltas, skipping the source.
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                manhattan = abs(dx) + abs(dy) + abs(dz)
                if manhattan > 3:
                    continue
                if edge_only and manhattan > 2:
                    continue

                candidate = Triple(triple.x + dx, triple.y + dy, triple.z + dz)
                if not triple_in_bounds(candidate, max_row):
                    continue

                candidate_s = triple_to_s(candidate, hilbert_res, orientation)
                if (candidate_s is not None and 0 <= candidate_s < max_s and candidate_s != S):
                    result.append(serialize({
                        'origin': origin, 'segment': segment,
                        'S': candidate_s, 'resolution': resolution,
                    }))

    for c in get_boundary_neighbors(_boundary_context(src), edge_only):
        result.append(c)
    return result
