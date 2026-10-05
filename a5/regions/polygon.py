# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

import math
from typing import Dict, List, Optional, Sequence, Set, Tuple, TypedDict, Union

from ..core.coordinate_systems import LonLat, Cartesian
from ..core.cell import lonlat_to_cell, spherical_to_cell, cell_to_spherical
from ..core.coordinate_transforms import from_lonlat, to_cartesian, to_spherical
from ..core.serialization import (
    cell_to_parent, cell_to_children, deserialize, get_resolution, get_stride, is_first_child, serialize,
    FIRST_HILBERT_RESOLUTION, MAX_RESOLUTION, WORLD_CELL,
)
from ..core.compact import compact
from ..core.cell_info import get_num_cells
from ..geometry.spherical_polygon import ring_winding_sign, spherical_triangle_area
from ..geometry.prepared_polygon import (
    PreparedPolygon, prepare_polygon, point_in_prepared_polygon,
)
from ..traversal.cap import estimate_cell_radius
from ..utils.great_circle import sample_great_circle_arc
from ..core.origin import origins, quintant_to_segment, segment_to_quintant
from ..lattice import Triple, s_to_triple, triple_flavor, triple_to_s
from ..traversal.neighbors import NEIGHBOR_DELTAS
from ..traversal.lattice_flood_fill import triple_space_flood_fill
from ..traversal.triple_cells import (
    cell_ids_to_triples, for_each_lattice_neighbor, for_each_triple_neighbor, triple_cell_center, triple_cell_key,
    triple_cell_to_id,
)


# Maps each boundary cell to the indices of the ring segments that produced it.
# Segment indices are global across rings (outer ring first, then holes).
# Used by `_classify_boundary_cells` to short-circuit PIP via segment-side dot
# products, and to classify ring cells locally.
SegmentMap = Dict[int, List[int]]


def _dense_sample_boundary(
    rings: List[List[LonLat]], ring_vecs_list: List[List[Cartesian]], resolution: int,
) -> Tuple[List[int], Set[int], SegmentMap]:
    """
    Dense-sample boundary cells along every closed ring (outer + holes) at
    cell_radius * 0.4 spacing, calling spherical_to_cell per sample.
    """
    boundary_cells: List[int] = []
    boundary_set: Set[int] = set()
    segment_map: SegmentMap = {}
    cell_radius = estimate_cell_radius(resolution)
    sample_interval = cell_radius * 0.4

    def record_cell(cell: int, seg_idx: int) -> None:
        if cell not in boundary_set:
            boundary_set.add(cell)
            boundary_cells.append(cell)
        existing = segment_map.get(cell)
        if existing is not None:
            if existing[-1] != seg_idx:
                existing.append(seg_idx)
        else:
            segment_map[cell] = [seg_idx]

    seg_offset = 0
    for r in range(len(rings)):
        ring = rings[r]
        ring_vecs = ring_vecs_list[r]

        n = len(ring)
        vertex_cells: List[int] = [0] * n
        for i in range(n):
            vertex_cells[i] = lonlat_to_cell(ring[i], resolution)

        for i in range(n):
            next_i = (i + 1) % n
            record_cell(vertex_cells[i], seg_offset + i)

            # Skip the lonLat round-trip: samples are authalic-Cartesian already.
            samples = sample_great_circle_arc(ring_vecs[i], ring_vecs[next_i], sample_interval)
            for s in samples:
                record_cell(spherical_to_cell(to_spherical(s), resolution), seg_offset + i)
            record_cell(vertex_cells[next_i], seg_offset + i)
        seg_offset += n

    return boundary_cells, boundary_set, segment_map


def _projects_onto_segment(p: Cartesian, a: Cartesian, b: Cartesian, n: Cartesian) -> bool:
    """
    Whether `p` lies in the lune of the segment a->b (normal `n` = a x b): its
    projection onto the great circle falls between a and b.
    """
    # n x a points along the arc from a towards b, b x n from b back towards a
    from_a = (p[0] * (n[1] * a[2] - n[2] * a[1]) + p[1] * (n[2] * a[0] - n[0] * a[2])
              + p[2] * (n[0] * a[1] - n[1] * a[0]))
    from_b = (p[0] * (b[1] * n[2] - b[2] * n[1]) + p[1] * (b[2] * n[0] - b[0] * n[2])
              + p[2] * (b[0] * n[1] - b[1] * n[0]))
    return from_a > 0 and from_b > 0


def _classify_boundary_cells(
    boundary_cells: List[int], segment_map: SegmentMap,
    seg_starts: List[Cartesian], seg_ends: List[Cartesian],
    seg_normals: List[Cartesian], seg_signs: List[int],
    prep: PreparedPolygon,
) -> Tuple[List[bool], List[Cartesian]]:
    """
    Classify boundary cells by whether their center is inside the polygon.

    For each cell we know which ring segment(s) sampled it. When all of those
    segments place the cell on the same side (cheap signed-dot test), that
    decides it. When they disagree (vertex / concave corner) or the cell wasn't
    recorded, fall back to full PIP. Returns the centers too, for reuse.
    """
    inside: List[bool] = []
    centers: List[Cartesian] = []
    for cell in boundary_cells:
        cv = to_cartesian(cell_to_spherical(cell))
        centers.append(cv)
        segments = segment_map.get(cell)
        if segments is None:
            inside.append(point_in_prepared_polygon(cv, prep))
            continue
        all_inside = True
        any_inside = False
        ambiguous = False
        for seg_idx in segments:
            n = seg_normals[seg_idx]
            dot = n[0] * cv[0] + n[1] * cv[1] + n[2] * cv[2]
            if abs(dot) < 1e-14:
                ambiguous = True
                break
            # The side of the segment's great circle only decides when the center
            # projects onto the segment itself, not beyond one of its endpoints
            if not _projects_onto_segment(cv, seg_starts[seg_idx], seg_ends[seg_idx], n):
                ambiguous = True
                break
            if dot * seg_signs[seg_idx] > 0:
                any_inside = True
            else:
                all_inside = False
        if ambiguous or (any_inside and not all_inside):
            inside.append(point_in_prepared_polygon(cv, prep))
        else:
            inside.append(all_inside)
    return inside, centers


_CROSSING_EPS = 1e-14


def _arc_crossing_parity(
    p: Cartesian, q: Cartesian, segments: List[int],
    seg_starts: List[Cartesian], seg_ends: List[Cartesian], seg_normals: List[Cartesian],
) -> Optional[bool]:
    """
    Parity of the crossings of the short arc p->q with the given ring segments
    (proper crossings, by the signs of four triple products), or None on a
    near-degenerate sign.
    """
    abx = p[1] * q[2] - p[2] * q[1]
    aby = p[2] * q[0] - p[0] * q[2]
    abz = p[0] * q[1] - p[1] * q[0]
    odd = False
    for seg in segments:
        c = seg_starts[seg]
        d = seg_ends[seg]
        acb = -(abx * c[0] + aby * c[1] + abz * c[2])
        bda = abx * d[0] + aby * d[1] + abz * d[2]
        if abs(acb) < _CROSSING_EPS or abs(bda) < _CROSSING_EPS:
            return None
        if acb * bda < 0:
            continue
        cd = seg_normals[seg]
        cbd = -(cd[0] * q[0] + cd[1] * q[1] + cd[2] * q[2])
        dac = cd[0] * p[0] + cd[1] * p[1] + cd[2] * p[2]
        if abs(cbd) < _CROSSING_EPS or abs(dac) < _CROSSING_EPS:
            return None
        if acb * cbd > 0 and acb * dac > 0:
            odd = not odd
    return odd


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


def _grow_ring(boundary: List[int], max_row: int) -> Tuple[List[int], List[int]]:
    """
    The ring of neighbors (edge and vertex, across quintant edges too) around
    the boundary cells (flat triples). Each ring cell records a boundary cell
    next to it (`parents`, an index into the boundary): one it shares an edge
    with when there is one, as edge neighbors are visited first. The arc between
    their centers then crosses no other cell holding boundary samples: a ring
    cell found by a vertex has no boundary cell across any of its edges, which
    covers every other cell around that vertex.
    """
    seen: Set[int] = set()
    for c in range(0, len(boundary), 5):
        seen.add(triple_cell_key(*boundary[c:c + 5]))
    ring: List[int] = []
    parents: List[int] = []
    parent = 0

    def visit(origin_id: int, quintant: int, x: int, y: int, z: int) -> None:
        key = triple_cell_key(origin_id, quintant, x, y, z)
        if key in seen:
            return
        seen.add(key)
        ring.extend((origin_id, quintant, x, y, z))
        parents.append(parent)

    for edge_only in (True, False):
        for c in range(0, len(boundary), 5):
            parent = c // 5
            for_each_triple_neighbor(*boundary[c:c + 5], max_row, edge_only, visit)
    return ring, parents


def _polygon_area(ring_vecs_list: List[List[Cartesian]]) -> float:
    """Area of the polygon (outer ring minus holes) on the unit sphere, in steradians."""
    total = 0.0
    for r, ring in enumerate(ring_vecs_list):
        # Signed fan from the first vertex: concave rings come out right too
        area = 0.0
        for i in range(1, len(ring) - 1):
            area += spherical_triangle_area(ring[0], ring[i], ring[i + 1])
        total += abs(area) if r == 0 else -abs(area)
    return total


# Below this many estimated interior cells per boundary cell, flooding the
# interior beats splitting the curve into runs (measured crossover: ~3.3).
_FLOOD_INTERIOR_PER_BOUNDARY = 3


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


def _strip_closing(ring: List[LonLat]) -> List[LonLat]:
    """GeoJSON rings repeat the first vertex at the end -- drop the duplicate."""
    last = len(ring) - 1
    if last > 0 and ring[0][0] == ring[last][0] and ring[0][1] == ring[last][1]:
        return ring[:-1]
    return ring


class PolygonToCellsOptions(TypedDict, total=False):
    """Options for polygon_to_cells.

    containment: Which cells to include relative to the polygon.
        'center' (default) includes a cell iff its center lies inside the
        polygon; 'overlapping' additionally includes every cell that overlaps
        the polygon boundary, giving gap-free coverage (a superset of 'center').
    """
    containment: str


def polygon_to_cells(
    polygon: Union[Sequence[LonLat], Sequence[Sequence[LonLat]]], resolution: int,
    options: Optional[PolygonToCellsOptions] = None,
) -> List[int]:
    """
    Find all cells within a polygon. The result is compacted -- use `uncompact`
    to expand to the input resolution.

    Args:
        polygon: Either a single ring of [longitude, latitude] vertices, or
            GeoJSON-style rings `[outer, *holes]` where cells inside a hole are
            excluded. Rings may be open or closed (GeoJSON-style, first vertex
            repeated at the end) -- closure is automatic either way. Holes with
            fewer than 3 distinct vertices are ignored.
        resolution: Target resolution (0..30)
        options: `containment` selects 'center' (default, cell center inside the
            polygon) or 'overlapping' (any cell touching the polygon, for
            gap-free coverage).

    Returns:
        Sorted, compacted list of cell IDs
    """
    containment = (options or {}).get('containment', 'center')
    # Normalize: a flat ring is shorthand for a polygon with no holes.
    is_nested = len(polygon) > 0 and not isinstance(polygon[0][0], (int, float))
    input_rings: List[List[LonLat]] = list(polygon) if is_nested else [list(polygon)]  # type: ignore[arg-type]

    if len(input_rings) == 0:
        return []
    outer = _strip_closing(list(input_rings[0]))
    if len(outer) < 3:
        return []
    rings: List[List[LonLat]] = [outer]
    for r in range(1, len(input_rings)):
        hole = _strip_closing(list(input_rings[r]))
        if len(hole) >= 3:
            rings.append(hole)

    # Authalic-sphere ring vectors -- A5's internal sphere, so cell centers
    # compare directly with no geodetic<->authalic round-trip.
    ring_vecs_list: List[List[Cartesian]] = []
    for ring in rings:
        ring_vecs_list.append([to_cartesian(from_lonlat(ring[i])) for i in range(len(ring))])

    prep = prepare_polygon(ring_vecs_list)

    boundary_cells, boundary_set, segment_map = _dense_sample_boundary(rings, ring_vecs_list, resolution)

    # Res 30 covers only quintants 0-41 (elsewhere A5 answers at res 29, see
    # serialize), so a polygon reaching past them is filled at res 29: mixing the
    # two lattices would leave the fill without a consistent grid.
    if resolution == MAX_RESOLUTION and any(get_resolution(cell) != resolution for cell in boundary_cells):
        return polygon_to_cells(polygon, resolution - 1, options)

    # Flattened per-segment endpoints, normals and interior-side signs, indexed
    # like the segment map. The polygon interior lies on the *outside* of a hole
    # ring, so hole segments get the opposite sign.
    seg_starts: List[Cartesian] = []
    seg_ends: List[Cartesian] = []
    seg_normals: List[Cartesian] = []
    seg_signs: List[int] = []
    for r in range(len(rings)):
        sign = (1 if r == 0 else -1) * ring_winding_sign(ring_vecs_list[r])
        vecs = ring_vecs_list[r]
        normals = prep.ring_normals[r]
        for i in range(len(normals)):
            seg_starts.append(vecs[i])
            seg_ends.append(vecs[(i + 1) % len(vecs)])
            seg_normals.append(normals[i])
            seg_signs.append(sign)
    boundary_inside, boundary_centers = _classify_boundary_cells(
        boundary_cells, segment_map, seg_starts, seg_ends, seg_normals, seg_signs, prep)

    # In 'overlapping' mode every densely-sampled boundary cell contains a point
    # on the polygon boundary, so it overlaps the polygon -- keep them all. In
    # 'center' mode keep those whose center lies inside.
    overlapping = containment == 'overlapping'

    # Resolutions 0 and 1 have no lattice (a quintant is a single cell): every
    # cell off the boundary is in or out by its center, and there are at most 60
    # of them.
    if resolution < FIRST_HILBERT_RESOLUTION:
        out = [cell for c, cell in enumerate(boundary_cells) if overlapping or boundary_inside[c]]
        for cell in cell_to_children(WORLD_CELL, resolution):
            if cell not in boundary_set and point_in_prepared_polygon(to_cartesian(cell_to_spherical(cell)), prep):
                out.append(cell)
        return compact(out)

    # The rest relies on the curve. Within a quintant consecutive cells are
    # neighbors, or at most a step over one or two cells. So the band of boundary
    # cells plus one ring of their neighbors splits each quintant's stretch of the
    # curve (a range of keys) into runs that lie wholly inside or wholly outside
    # the polygon: a step over the boundary would have to land in the band. One
    # probe classifies a run, and an inside run is emitted directly as the
    # coarsest cells covering it, so the interior costs O(boundary), not O(area).
    hilbert_res = resolution - FIRST_HILBERT_RESOLUTION + 1
    max_row = (1 << hilbert_res) - 1
    boundary = cell_ids_to_triples(boundary_cells)

    # A quintant without band cells is wholly inside or outside; it can only be
    # inside when the polygon's bounding cap holds a quintant's area (4pi/60)
    cap_holds_quintant = 2 * math.pi * (1 - prep.cap.min_dot) >= (4 * math.pi) / 60

    # A small interior is cheaper to flood than to split into curve runs: the
    # flood costs about boundary + interior cells, the runs a sorted band of
    # boundary plus ring keys. The flood can't reach a quintant the polygon
    # swallows whole, which a polygon smaller than its bounding cap never does.
    if not cap_holds_quintant and (_polygon_area(ring_vecs_list) / (4 * math.pi) * get_num_cells(resolution)
                                   < _FLOOD_INTERIOR_PER_BOUNDARY * len(boundary_cells)):
        out = [cell for c, cell in enumerate(boundary_cells) if overlapping or boundary_inside[c]]
        # The shell: the flood's own moves out of the boundary, each cell classified
        # from the boundary cell it was found from (they share an edge)
        seen: Set[int] = set()
        for c in range(0, len(boundary), 5):
            seen.add(triple_cell_key(*boundary[c:c + 5]))
        seeds: List[int] = []
        firewall: List[int] = list(boundary)
        parent = 0

        def visit(origin_id: int, quintant: int, x: int, y: int, z: int) -> None:
            key = triple_cell_key(origin_id, quintant, x, y, z)
            if key in seen:
                return
            seen.add(key)
            center = to_cartesian(triple_cell_center(origin_id, quintant, x, y, z, hilbert_res, max_row))
            segments = segment_map[boundary_cells[parent]]
            odd = _arc_crossing_parity(center, boundary_centers[parent], segments, seg_starts, seg_ends, seg_normals)
            inside = point_in_prepared_polygon(center, prep) if odd is None else boundary_inside[parent] != odd
            (seeds if inside else firewall).extend((origin_id, quintant, x, y, z))

        for c in range(0, len(boundary), 5):
            parent = c // 5
            for_each_lattice_neighbor(*boundary[c:c + 5], max_row, visit)
        if seeds:
            for c in range(0, len(seeds), 5):
                out.append(triple_cell_to_id(*seeds[c:c + 5], hilbert_res, resolution))
            out.extend(triple_space_flood_fill(firewall, seeds, resolution)['interior_cells'])
        return compact(out)

    ring_cells, parents = _grow_ring(boundary, max_row)

    unit_shift = 58 - 2 * hilbert_res
    unit = 1 << unit_shift

    # Band keys carry two flags below the key: EMIT (the cell is in the output)
    # and RING. (Python ints are unbounded, so `key << 2` always has room.)
    EMIT = 1
    RING = 2

    n_boundary = len(boundary_cells)
    keys: List[int] = []
    for i in range(n_boundary):
        emit = overlapping or boundary_inside[i]
        keys.append((_cell_to_key(boundary_cells[i], resolution) << 2) | (EMIT if emit else 0))
    # Ring cells by flagged key (as their offset into ring_cells), with their class
    ring_by_key: Dict[int, int] = {}
    ring_inside: List[bool] = []
    for c in range(0, len(ring_cells), 5):
        cell = ring_cells[c:c + 5]
        center = to_cartesian(triple_cell_center(*cell, hilbert_res, max_row))
        # Locally: the parent's class, flipped by each ring segment crossed on the
        # way (full PIP only on a near-degenerate crossing)
        parent = parents[c // 5]
        segments = segment_map[boundary_cells[parent]]
        odd = _arc_crossing_parity(center, boundary_centers[parent], segments, seg_starts, seg_ends, seg_normals)
        inside = point_in_prepared_polygon(center, prep) if odd is None else boundary_inside[parent] != odd
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
            inside = point_in_prepared_polygon(to_cartesian(cell_to_spherical(_key_to_cell(lo, resolution))), prep)
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
