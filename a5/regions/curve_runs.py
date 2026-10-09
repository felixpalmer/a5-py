# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

# Polygon fill by curve runs. Within a quintant consecutive cells on the curve
# are neighbors, or at most a step over one or two cells. So the band of
# boundary cells plus one ring of their neighbors splits each quintant's
# stretch of the curve (a range of slots) into runs that lie wholly inside or
# wholly outside the polygon: a step over the boundary would have to land in
# the band. One probe classifies a run, and an inside run is emitted whole, as
# a slot run (see coverings/slot_runs), so the interior costs O(boundary), not
# O(area).

from typing import Dict, List, Optional

from ..core.cell import cell_to_spherical
from ..core.coordinate_transforms import to_cartesian
from ..core.serialization import (
    cell_first_slot, slot_to_cell, QUINTANT_SHIFT, S_MASK, SLOT_COUNTS, FIRST_HILBERT_RESOLUTION,
)
from ..coverings.slot_runs import SlotRuns, append_slot_run
from ..geometry.prepared_polygon import point_in_prepared_polygon
from ..lattice import Triple, s_to_triple, triple_flavor, triple_to_s
from ..traversal.neighbors import NEIGHBOR_DELTAS
from ..traversal.triple_cells import (
    for_each_triple_neighbor, triple_cell_center, QUINTANT_ORIENTATION, QUINTANT_PREFIX, TRIPLE_QUINTANT_BY_ID_ORDER,
)
from .polygon_boundary import Boundary, boundary_neighbors, emits_boundary_cell, inside_next_to

# Cells are ordered on the curve by the leaf slots they occupy (see core/serialization).

def _triple_slot(origin_id: int, quintant: int, x: int, y: int, z: int, hilbert_res: int, unit_shift: int) -> int:
    """The slot of a cell given in triple space."""
    i = origin_id * 5 + quintant
    return QUINTANT_PREFIX[i] | (triple_to_s(Triple(x, y, z), hilbert_res, QUINTANT_ORIENTATION[i]) << unit_shift)


def fill_by_curve_runs(
    boundary: Boundary, triples: List[int], resolution: int, overlapping: bool, cap_holds_quintant: bool,
) -> SlotRuns:
    """
    Fill a polygon by curve runs, given its classified boundary and the boundary
    cells as flat triples. `cap_holds_quintant` says whether the polygon might
    swallow a quintant whole (one holding no band cells at all). Returns the
    cells inside as sorted slot runs.
    """
    hilbert_res = resolution - FIRST_HILBERT_RESOLUTION + 1
    max_row = (1 << hilbert_res) - 1
    # One ring of neighbors (edge and vertex, across quintant edges too), edge neighbors first
    ring_cells, parents = boundary_neighbors(triples, [
        lambda b, c, visit: for_each_triple_neighbor(*b[c:c + 5], max_row, True, visit),
        lambda b, c, visit: for_each_triple_neighbor(*b[c:c + 5], max_row, False, visit),
    ])

    unit = SLOT_COUNTS[resolution]
    unit_shift = 58 - 2 * hilbert_res

    # Band slots carry two flags below the slot: EMIT (the cell is in the output)
    # and RING. (Python ints are unbounded, so `slot << 2` always has room.)
    EMIT = 1
    RING = 2

    slots: List[int] = []
    for i in range(len(boundary.cells)):
        emit = emits_boundary_cell(boundary, i, overlapping)
        slots.append((cell_first_slot(boundary.cells[i]) << 2) | (EMIT if emit else 0))
    # Ring cells by flagged slot (as their offset into ring_cells), with their class
    ring_by_slot: Dict[int, int] = {}
    ring_inside: List[bool] = []
    for c in range(0, len(ring_cells), 5):
        cell = ring_cells[c:c + 5]
        center = to_cartesian(triple_cell_center(*cell, hilbert_res, max_row))
        inside = inside_next_to(boundary, center, parents[c // 5])
        ring_inside.append(inside)
        slot = (_triple_slot(*cell, hilbert_res, unit_shift) << 2) | (EMIT | RING if inside else RING)
        slots.append(slot)
        ring_by_slot[slot] = c
    slots.sort()
    n_band = len(slots)

    def class_from_ring(slot: int, ring_slot: int) -> Optional[bool]:
        """
        The class of a run cell from a ring cell next to it on the curve, when the
        two are lattice neighbors: any boundary cell near the run cell would have
        put it in the ring, so nothing between them can cross the boundary.
        """
        if (ring_slot & RING) == 0:
            return None
        c = ring_by_slot[ring_slot]
        q = slot >> QUINTANT_SHIFT
        t = s_to_triple((slot & S_MASK) >> unit_shift, hilbert_res, QUINTANT_ORIENTATION[TRIPLE_QUINTANT_BY_ID_ORDER[q]])
        rx, ry, rz = ring_cells[c + 2], ring_cells[c + 3], ring_cells[c + 4]
        dx, dy, dz = t.x - rx, t.y - ry, t.z - rz
        for d in NEIGHBOR_DELTAS[triple_flavor(Triple(rx, ry, rz), max_row)].all:
            if d.x == dx and d.y == dy and d.z == dz:
                return ring_inside[c // 5]
        return None

    # Walk each quintant's slots in curve order, emitting the inside band cells and
    # runs as they come, so the output is sorted.
    out: SlotRuns = []

    def probe_run(lo: int, hi: int, prev: int, next_: int) -> None:
        inside = class_from_ring(lo, prev) if prev >= 0 else None
        if inside is None and next_ >= 0:
            inside = class_from_ring(hi - unit, next_)
        if inside is None:
            center = to_cartesian(cell_to_spherical(slot_to_cell(lo, resolution)))
            inside = point_in_prepared_polygon(center, boundary.prep)
        if inside:
            append_slot_run(out, lo, hi)

    i = 0
    q = 0
    while q < 60:
        # Skip straight to the next quintant holding band cells, unless whole ones may be inside
        if not cap_holds_quintant:
            if i >= n_band:
                break
            q = slots[i] >> (QUINTANT_SHIFT + 2)
        q_end = (q + 1) << QUINTANT_SHIFT
        cursor = q << QUINTANT_SHIFT
        if i >= n_band or (slots[i] >> 2) >= q_end:
            if cap_holds_quintant:
                probe_run(cursor, q_end, -1, -1)
            q += 1
            continue
        prev = -1
        while i < n_band and (slots[i] >> 2) < q_end:
            flagged = slots[i]
            slot = flagged >> 2
            if slot > cursor:
                probe_run(cursor, slot, prev, flagged)
            if flagged & EMIT:
                append_slot_run(out, slot, slot + unit)
            prev = flagged
            cursor = slot + unit
            i += 1
        if cursor < q_end:
            probe_run(cursor, q_end, prev, -1)
        q += 1

    return out
