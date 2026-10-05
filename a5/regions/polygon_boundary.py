# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

# The boundary of a polygon in cells: sampled densely along every ring,
# classified by whether each cell's center is inside, and used to classify the
# cells next to it without a full point-in-polygon test.

from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

from ..core.coordinate_systems import LonLat, Cartesian
from ..core.cell import cell_to_spherical
from ..core.coordinate_transforms import to_cartesian
from ..geometry.spherical_polygon import ring_winding_sign, spherical_triangle_area
from ..geometry.prepared_polygon import PreparedPolygon, point_in_prepared_polygon
from ..traversal.line import trace_path
from ..traversal.triple_cells import TripleCellVisitor, triple_cell_key

# Maps each boundary cell to the indices of the ring segments that produced it.
# Segment indices are global across rings (outer ring first, then holes).
SegmentMap = Dict[int, List[int]]


class Segments:
    """
    Every ring segment, flattened across rings and indexed like the segment map:
    endpoints, great-circle normal, and the side the polygon interior lies on.
    """
    __slots__ = ('starts', 'ends', 'normals', 'signs')

    def __init__(self) -> None:
        self.starts: List[Cartesian] = []
        self.ends: List[Cartesian] = []
        self.normals: List[Cartesian] = []
        self.signs: List[int] = []


class Boundary:
    """The polygon's boundary cells, classified, with what's needed to classify their neighbors."""
    __slots__ = ('cells', 'set', 'inside', 'centers', 'segment_map', 'segments', 'prep')

    def __init__(self, cells: List[int], cell_set: Set[int], inside: List[bool], centers: List[Cartesian],
                 segment_map: SegmentMap, segments: Segments, prep: PreparedPolygon) -> None:
        self.cells = cells  # cell IDs in the order they were sampled
        self.set = cell_set
        self.inside = inside  # whether the cell's center is inside the polygon, by index into `cells`
        self.centers = centers  # cell centers, by index into `cells`
        self.segment_map = segment_map
        self.segments = segments
        self.prep = prep


def sample_boundary(
    rings: List[List[LonLat]], resolution: int, exact: bool,
) -> Tuple[List[int], Set[int], SegmentMap]:
    """
    The boundary cells, each recorded with the ring segments (outer ring and
    holes) that reached it: with `exact`, every cell a segment touches; without,
    the cells holding samples along the segments at half-cell-radius spacing,
    which can miss a cell whose corner a segment clips between samples.
    """
    cells: List[int] = []
    cell_set: Set[int] = set()
    segment_map: SegmentMap = {}
    seg_offset = 0

    def record_cell(cell: int, arc: int) -> None:
        seg_idx = seg_offset + arc
        if cell not in cell_set:
            cell_set.add(cell)
            cells.append(cell)
        existing = segment_map.get(cell)
        if existing is not None:
            if existing[-1] != seg_idx:
                existing.append(seg_idx)
        else:
            segment_map[cell] = [seg_idx]

    for ring in rings:
        trace_path(ring, True, resolution, record_cell, exact)
        seg_offset += len(ring)

    return cells, cell_set, segment_map


def _ring_segments(ring_vecs_list: List[List[Cartesian]], prep: PreparedPolygon) -> Segments:
    """
    The polygon's ring segments, flattened. The polygon interior lies on the
    *outside* of a hole ring, so hole segments get the opposite sign.
    """
    segments = Segments()
    for r in range(len(ring_vecs_list)):
        sign = (1 if r == 0 else -1) * ring_winding_sign(ring_vecs_list[r])
        vecs = ring_vecs_list[r]
        normals = prep.ring_normals[r]
        for i in range(len(normals)):
            segments.starts.append(vecs[i])
            segments.ends.append(vecs[(i + 1) % len(vecs)])
            segments.normals.append(normals[i])
            segments.signs.append(sign)
    return segments


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


def classify_boundary(
    sampled: Tuple[List[int], Set[int], SegmentMap],
    ring_vecs_list: List[List[Cartesian]],
    prep: PreparedPolygon,
) -> Boundary:
    """
    Classify the sampled boundary cells by whether their center is inside the
    polygon.

    For each cell we know which ring segment(s) sampled it. When all of those
    segments place the cell on the same side (cheap signed-dot test), that
    decides it. When they disagree (vertex / concave corner) or the cell wasn't
    recorded, fall back to full PIP.
    """
    cells, cell_set, segment_map = sampled
    segments = _ring_segments(ring_vecs_list, prep)
    inside: List[bool] = []
    centers: List[Cartesian] = []
    for cell in cells:
        cv = to_cartesian(cell_to_spherical(cell))
        centers.append(cv)
        segs = segment_map.get(cell)
        if segs is None:
            inside.append(point_in_prepared_polygon(cv, prep))
            continue
        all_inside = True
        any_inside = False
        ambiguous = False
        for seg_idx in segs:
            n = segments.normals[seg_idx]
            dot = n[0] * cv[0] + n[1] * cv[1] + n[2] * cv[2]
            if abs(dot) < 1e-14:
                ambiguous = True
                break
            # The side of the segment's great circle only decides when the center
            # projects onto the segment itself, not beyond one of its endpoints
            if not _projects_onto_segment(cv, segments.starts[seg_idx], segments.ends[seg_idx], n):
                ambiguous = True
                break
            if dot * segments.signs[seg_idx] > 0:
                any_inside = True
            else:
                all_inside = False
        if ambiguous or (any_inside and not all_inside):
            inside.append(point_in_prepared_polygon(cv, prep))
        else:
            inside.append(all_inside)
    return Boundary(cells, cell_set, inside, centers, segment_map, segments, prep)


def emits_boundary_cell(boundary: Boundary, c: int, overlapping: bool) -> bool:
    """
    Whether boundary cell `c` is in the output. In 'overlapping' mode every
    densely-sampled boundary cell contains a point on the polygon boundary, so
    it overlaps the polygon -- keep them all. In 'center' mode keep those whose
    center lies inside.
    """
    return overlapping or boundary.inside[c]


def boundary_output(boundary: Boundary, overlapping: bool) -> List[int]:
    """The boundary cells in the output (see `emits_boundary_cell`), as a new list."""
    return [cell for c, cell in enumerate(boundary.cells) if emits_boundary_cell(boundary, c, overlapping)]


_CROSSING_EPS = 1e-14


def _arc_crossing_parity(p: Cartesian, q: Cartesian, seg_idxs: List[int], segments: Segments) -> Optional[bool]:
    """
    Parity of the crossings of the short arc p->q with the given ring segments
    (proper crossings, by the signs of four triple products), or None on a
    near-degenerate sign.
    """
    abx = p[1] * q[2] - p[2] * q[1]
    aby = p[2] * q[0] - p[0] * q[2]
    abz = p[0] * q[1] - p[1] * q[0]
    odd = False
    for seg in seg_idxs:
        c = segments.starts[seg]
        d = segments.ends[seg]
        acb = -(abx * c[0] + aby * c[1] + abz * c[2])
        bda = abx * d[0] + aby * d[1] + abz * d[2]
        if abs(acb) < _CROSSING_EPS or abs(bda) < _CROSSING_EPS:
            return None
        if acb * bda < 0:
            continue
        cd = segments.normals[seg]
        cbd = -(cd[0] * q[0] + cd[1] * q[1] + cd[2] * q[2])
        dac = cd[0] * p[0] + cd[1] * p[1] + cd[2] * p[2]
        if abs(cbd) < _CROSSING_EPS or abs(dac) < _CROSSING_EPS:
            return None
        if acb * cbd > 0 and acb * dac > 0:
            odd = not odd
    return odd


def inside_next_to(boundary: Boundary, center: Cartesian, parent: int) -> bool:
    """
    Whether a point next to boundary cell `parent` (see `boundary_neighbors`) is
    inside the polygon: the parent's class, flipped by each of its ring segments
    crossed on the way. Full PIP only on a near-degenerate crossing.
    """
    seg_idxs = boundary.segment_map[boundary.cells[parent]]
    odd = _arc_crossing_parity(center, boundary.centers[parent], seg_idxs, boundary.segments)
    return point_in_prepared_polygon(center, boundary.prep) if odd is None else boundary.inside[parent] != odd


# Calls `visit` for neighbors of the cell at offset `c` of a flat triple list
NeighborWalk = Callable[[List[int], int, TripleCellVisitor], None]


def boundary_neighbors(boundary: List[int], walks: Sequence[NeighborWalk]) -> Tuple[List[int], List[int]]:
    """
    The cells next to the boundary (flat triples), found by `walks` in turn, each
    with the boundary cell it was first found from (`parents`, an index into the
    boundary). Listing edge-neighbor walks first gives every cell an edge-sharing
    parent when it has one, which `inside_next_to` needs: the arc between their
    centers then crosses no other cell holding boundary samples. A cell found
    only by a vertex has no boundary cell across any of its edges, which covers
    every other cell around that vertex.
    """
    seen: Set[int] = set()
    for c in range(0, len(boundary), 5):
        seen.add(triple_cell_key(*boundary[c:c + 5]))
    cells: List[int] = []
    parents: List[int] = []
    parent = 0

    def visit(origin_id: int, quintant: int, x: int, y: int, z: int) -> None:
        key = triple_cell_key(origin_id, quintant, x, y, z)
        if key in seen:
            return
        seen.add(key)
        cells.extend((origin_id, quintant, x, y, z))
        parents.append(parent)

    for walk in walks:
        for c in range(0, len(boundary), 5):
            parent = c // 5
            walk(boundary, c, visit)
    return cells, parents


def polygon_area(ring_vecs_list: List[List[Cartesian]]) -> float:
    """Area of the polygon (outer ring minus holes) on the unit sphere, in steradians."""
    total = 0.0
    for r, ring in enumerate(ring_vecs_list):
        # Signed fan from the first vertex: concave rings come out right too
        area = 0.0
        for i in range(1, len(ring) - 1):
            area += spherical_triangle_area(ring[0], ring[i], ring[i + 1])
        total += abs(area) if r == 0 else -abs(area)
    return total
