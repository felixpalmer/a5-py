# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

# Polygon fill by curve runs. Within a quintant consecutive cells on the curve
# are neighbors, or at most a step over one or two cells. So the band of
# boundary cells plus one ring of their neighbors splits each quintant's
# stretch of the curve (a range of keys) into runs that lie wholly inside or
# wholly outside the polygon: a step over the boundary would have to land in
# the band. One probe classifies a run, and an inside run is emitted directly
# as the coarsest cells covering it, so the interior costs O(boundary), not
# O(area).

from typing import Dict, List, Optional

from ..core.cell import cell_to_spherical
from ..core.coordinate_transforms import to_cartesian
from ..core.serialization import (
    cell_to_parent, deserialize, get_resolution, get_stride, is_first_child, serialize,
    FIRST_HILBERT_RESOLUTION, MAX_RESOLUTION,
)
from ..core.compact import compact
from ..core.origin import origins, quintant_to_segment, segment_to_quintant
from ..geometry.prepared_polygon import point_in_prepared_polygon
from ..lattice import Triple, s_to_triple, triple_flavor, triple_to_s
from ..traversal.neighbors import NEIGHBOR_DELTAS
from ..traversal.triple_cells import for_each_triple_neighbor, triple_cell_center
from .polygon_boundary import Boundary, boundary_neighbors, emits_boundary_cell, inside_next_to

# Cells are ordered on the curve by a 64-bit key: the 6-bit quintant (as in
# the ID's top bits) then S, left-aligned below it. Below resolution 30 that is
# the cell ID without its resolution marker; at resolution 30 S fills all 58
# bits. A cell at resolution r < 30 is its aligned key plus the marker.
_QUINTANT_SHIFT = 58
_S_MASK = (1 << _QUINTANT_SHIFT) - 1

# Curve orientation of each quintant by its 6-bit key prefix, and the key
# prefix and orientation by triple quintant (origin.id * 5 + quintant).
_PREFIX_ORIENTATION = [
    segment_to_quintant((q + origins[q // 5].first_quintant) % 5, origins[q // 5])[1] for q in range(60)
]
_TRIPLE_PREFIX: List[int] = []
_TRIPLE_ORIENTATION = []
for _origin in origins:
    for _quintant in range(5):
        _segment, _orientation = quintant_to_segment(_quintant, _origin)
        _q = 5 * _origin.id + (_segment - _origin.first_quintant + 5) % 5
        _TRIPLE_PREFIX.append(_q << _QUINTANT_SHIFT)
        _TRIPLE_ORIENTATION.append(_orientation)


def _triple_key(origin_id: int, quintant: int, x: int, y: int, z: int, hilbert_res: int, unit_shift: int) -> int:
    """The key of a cell given in triple space."""
    i = origin_id * 5 + quintant
    return _TRIPLE_PREFIX[i] | (triple_to_s(Triple(x, y, z), hilbert_res, _TRIPLE_ORIENTATION[i]) << unit_shift)


def _marker_bit(resolution: int) -> int:
    return 1 << 56 if resolution == 1 else 1 << (59 - 2 * resolution)


def _cell_to_key(cell: int, resolution: int) -> int:
    if resolution < MAX_RESOLUTION:
        return cell - _marker_bit(resolution)
    c = deserialize(cell)
    origin = c['origin']
    q = 5 * origin.id + (c['segment'] - origin.first_quintant + 5) % 5
    return (q << _QUINTANT_SHIFT) | c['S']


def _key_to_cell(key: int, resolution: int) -> int:
    if resolution < MAX_RESOLUTION:
        return key + _marker_bit(resolution)
    q = key >> _QUINTANT_SHIFT
    origin = origins[q // 5]
    return serialize({'origin': origin, 'segment': (q + origin.first_quintant) % 5, 'S': key & _S_MASK,
                      'resolution': resolution})


def _emit_range(lo: int, hi: int, resolution: int, out: List[int]) -> None:
    """
    Append the cells covering the key range [lo, hi) at `resolution`, as the
    coarsest aligned blocks (a block of 4^k cells is their resolution - k parent).
    """
    hilbert_res = resolution - FIRST_HILBERT_RESOLUTION + 1
    unit_shift = 58 - 2 * hilbert_res
    while lo < hi:
        k = 0
        while k < hilbert_res:
            size = 1 << (unit_shift + 2 * (k + 1))
            if (lo & (size - 1)) != 0 or lo + size > hi:
                break
            k += 1
        out.append(_key_to_cell(lo, resolution - k))
        lo += 1 << (unit_shift + 2 * k)


def _compact_sorted(cells: List[int]) -> List[int]:
    """
    Compact cells that are already sorted and disjoint, in one pass: a stack
    whose top is merged into its parent whenever it ends in a full sibling group.
    """
    stack: List[int] = []
    for cell in cells:
        stack.append(cell)
        while True:
            top = len(stack) - 1
            resolution = get_resolution(stack[top])
            if resolution < 0:
                break
            n = 4 if resolution >= FIRST_HILBERT_RESOLUTION else (12 if resolution == 0 else 5)
            if len(stack) < n:
                break
            first = stack[top - n + 1]
            if not is_first_child(first, resolution):
                break
            stride = get_stride(resolution)
            complete = True
            for j in range(1, n):
                if stack[top - n + 1 + j] != first + j * stride:
                    complete = False
                    break
            if not complete:
                break
            del stack[top - n + 1:]
            stack.append(cell_to_parent(first))
    return stack


def fill_by_curve_runs(
    boundary: Boundary, triples: List[int], resolution: int, overlapping: bool, cap_holds_quintant: bool,
) -> List[int]:
    """
    Fill a polygon by curve runs, given its classified boundary and the boundary
    cells as flat triples. `cap_holds_quintant` says whether the polygon might
    swallow a quintant whole (one holding no band cells at all).
    """
    hilbert_res = resolution - FIRST_HILBERT_RESOLUTION + 1
    max_row = (1 << hilbert_res) - 1
    # One ring of neighbors (edge and vertex, across quintant edges too), edge neighbors first
    ring_cells, parents = boundary_neighbors(triples, [
        lambda b, c, visit: for_each_triple_neighbor(*b[c:c + 5], max_row, True, visit),
        lambda b, c, visit: for_each_triple_neighbor(*b[c:c + 5], max_row, False, visit),
    ])

    unit_shift = 58 - 2 * hilbert_res
    unit = 1 << unit_shift

    # Band keys carry two flags below the key: EMIT (the cell is in the output)
    # and RING. (Python ints are unbounded, so `key << 2` always has room.)
    EMIT = 1
    RING = 2

    keys: List[int] = []
    for i in range(len(boundary.cells)):
        emit = emits_boundary_cell(boundary, i, overlapping)
        keys.append((_cell_to_key(boundary.cells[i], resolution) << 2) | (EMIT if emit else 0))
    # Ring cells by flagged key (as their offset into ring_cells), with their class
    ring_by_key: Dict[int, int] = {}
    ring_inside: List[bool] = []
    for c in range(0, len(ring_cells), 5):
        cell = ring_cells[c:c + 5]
        center = to_cartesian(triple_cell_center(*cell, hilbert_res, max_row))
        inside = inside_next_to(boundary, center, parents[c // 5])
        ring_inside.append(inside)
        key = (_triple_key(*cell, hilbert_res, unit_shift) << 2) | (EMIT | RING if inside else RING)
        keys.append(key)
        ring_by_key[key] = c
    keys.sort()
    n_band = len(keys)

    def class_from_ring(key: int, ring_key: int) -> Optional[bool]:
        """
        The class of a run cell from a ring cell next to it on the curve, when the
        two are lattice neighbors: any boundary cell near the run cell would have
        put it in the ring, so nothing between them can cross the boundary.
        """
        if (ring_key & RING) == 0:
            return None
        c = ring_by_key[ring_key]
        q = key >> _QUINTANT_SHIFT
        t = s_to_triple((key & _S_MASK) >> unit_shift, hilbert_res, _PREFIX_ORIENTATION[q])
        rx, ry, rz = ring_cells[c + 2], ring_cells[c + 3], ring_cells[c + 4]
        dx, dy, dz = t.x - rx, t.y - ry, t.z - rz
        for d in NEIGHBOR_DELTAS[triple_flavor(Triple(rx, ry, rz), max_row)].all:
            if d.x == dx and d.y == dy and d.z == dz:
                return ring_inside[c // 5]
        return None

    # Walk each quintant's keys in curve order, emitting the inside band cells and
    # runs as they come, so the output is sorted.
    out: List[int] = []

    def probe_run(lo: int, hi: int, prev: int, next_: int) -> None:
        inside = class_from_ring(lo, prev) if prev >= 0 else None
        if inside is None and next_ >= 0:
            inside = class_from_ring(hi - unit, next_)
        if inside is None:
            center = to_cartesian(cell_to_spherical(_key_to_cell(lo, resolution)))
            inside = point_in_prepared_polygon(center, boundary.prep)
        if inside:
            _emit_range(lo, hi, resolution, out)

    i = 0
    q = 0
    while q < 60:
        # Skip straight to the next quintant holding band cells, unless whole ones may be inside
        if not cap_holds_quintant:
            if i >= n_band:
                break
            q = keys[i] >> (_QUINTANT_SHIFT + 2)
        q_end = (q + 1) << _QUINTANT_SHIFT
        cursor = q << _QUINTANT_SHIFT
        if i >= n_band or (keys[i] >> 2) >= q_end:
            if cap_holds_quintant:
                probe_run(cursor, q_end, -1, -1)
            q += 1
            continue
        prev = -1
        while i < n_band and (keys[i] >> 2) < q_end:
            flagged = keys[i]
            key = flagged >> 2
            if key > cursor:
                probe_run(cursor, key, prev, flagged)
            if flagged & EMIT:
                out.append(_key_to_cell(key, resolution))
            prev = flagged
            cursor = key + unit
            i += 1
        if cursor < q_end:
            probe_run(cursor, q_end, prev, -1)
        q += 1

    # Resolution 30 IDs don't sort like their keys (the quintant field varies in width)
    return compact(out) if resolution == MAX_RESOLUTION else _compact_sorted(out)
