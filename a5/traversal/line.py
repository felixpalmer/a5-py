# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

import math
from typing import Callable, List, Optional, Set, Tuple

from ..core.coordinate_systems import LonLat, Face
from ..core.cell import lonlat_to_cell, spherical_to_cell, cell_intersects_segment, last_cell_shape, last_projection
from ..core.coordinate_transforms import from_lonlat, to_cartesian, to_spherical, to_lonlat
from ..core.serialization import deserialize, serialize, FIRST_HILBERT_RESOLUTION
from ..core.origin import origins
from ..core.face_adjacency import walk_faces
from ..core.tiling import get_pentagon_vertices
from ..lattice import Triple, triple_flavor
from ..projections.dodecahedron import DodecahedronProjection
from ..utils.great_circle import sample_great_circle_arc
from .cap import estimate_cell_radius
from .triple_cells import cell_ids_to_triples, for_each_triple_neighbor, triple_cell_key, triple_cell_to_id

_dodecahedron = DodecahedronProjection()


def _trace_faces(cell_a: int, cell_b: int, a: LonLat, b: LonLat, add_cell: Callable[[int], None]) -> None:
    """
    Resolution 0 version of the sub-segment BFS below: the cells are the 12
    dodecahedron faces, adjacent across their edges.
    """
    def expand(face: int) -> bool:
        cell = serialize({'origin': origins[face], 'segment': 0, 'S': 0, 'resolution': 0})
        if not cell_intersects_segment(cell, a, b):
            return False
        add_cell(cell)
        return True

    walk_faces([deserialize(cell_a)['origin'].id, deserialize(cell_b)['origin'].id], expand)


# Tolerance on where (as a fraction of the sub-segment) the part in one cell
# ends and the part in the next begins, and how far (as a fraction of the
# edge) the crossing must be from the edge's ends for no third cell to touch it
_SHARED_EDGE_EPS = 1e-9
_SHARED_EDGE_MARGIN = 1e-6


def _clip_to_pentagon(pentagon, a: Face, b: Face) -> Optional[Tuple[float, float, float]]:
    """
    The part of the segment a->b inside a convex pentagon, as parameters
    (start, end) along it, start <= end, with where along the edge it leaves
    through (0..1, from the edge's first vertex); None when it misses the pentagon.
    """
    vertices = pentagon.get_vertices()
    sx = b[0] - a[0]
    sy = b[1] - a[1]
    start = -math.inf
    end = math.inf
    exit_edge = -1
    for i in range(5):
        v1 = vertices[i]
        v2 = vertices[(i + 1) % 5]
        # Inside an edge where (v1 - v2) x (p - v1) >= 0 (as contains_point)
        ex = v1[0] - v2[0]
        ey = v1[1] - v2[1]
        f = ex * (a[1] - v1[1]) - ey * (a[0] - v1[0])
        g = ex * sy - ey * sx
        if g == 0:
            if f < 0:
                return None
        elif g > 0:
            start = max(start, -f / g)
        else:
            t = -f / g
            if t < end:
                end = t
                exit_edge = i
    if start > end or exit_edge < 0:
        return None
    # Where the exit point falls along the exit edge, from its first vertex
    v1 = vertices[exit_edge]
    v2 = vertices[(exit_edge + 1) % 5]
    px = a[0] + end * sx - v1[0]
    py = a[1] + end * sy - v1[1]
    ex = v2[0] - v1[0]
    ey = v2[1] - v1[1]
    return start, end, (px * ex + py * ey) / (ex * ex + ey * ey)


def trace_path(points: List[LonLat], closed: bool, resolution: int,
               visit: Callable[[int, int], None], exact: bool = True) -> None:
    """
    Visit every cell a path of great-circle arcs touches, arc by arc and in order
    along each arc, with the index of the arc (a cell may be visited more than
    once). The path joins consecutive `points`, and the last back to the first
    when `closed`. With `exact` false only the cells holding the samples are
    visited, which can miss a cell whose corner an arc clips between samples.

    Each arc is sampled at half-cell-radius intervals. A pair of consecutive
    samples within one cell needs nothing more: cells are convex and the
    sub-segment between them is short enough to be straight (projected onto the
    cell's Face). Between two cells, clipping the sub-segment to their pentagons
    usually shows it crossing straight from one into the other, or clipping one
    cell between them; otherwise a strict local BFS finds every cell whose
    pentagon it touches.

    The BFS runs in triple space: a cell's neighbors come from its flavor's
    triple deltas plus the boundary delta tables, and its pentagon straight from
    its triple, so a candidate is never decoded and only touched cells are
    encoded.
    """
    sample_interval = estimate_cell_radius(resolution) * 0.5
    hilbert_res = resolution - FIRST_HILBERT_RESOLUTION + 1
    max_row = (1 << hilbert_res) - 1 if resolution > 0 else 0

    # Each point once: on the sphere, as a vector, and its cell, with the cell's
    # pentagon and origin and the point's projection there, as its lookup made them
    n = len(points)
    point_spherical = [from_lonlat(p) for p in points]
    point_vecs = [to_cartesian(p) for p in point_spherical]
    point_cells: List[int] = []
    point_shapes: List[Optional[dict]] = []
    point_faces: List[Optional[Face]] = []
    for p in point_spherical:
        cell = spherical_to_cell(p, resolution)
        shape = last_cell_shape(cell)
        point_cells.append(cell)
        point_shapes.append(shape)
        point_faces.append(None if shape is None else last_projection(p, shape['origin_id']))

    # The current sub-segment's ends, each with its cell's shape and own
    # projection where the lookup made them (reused below rather than projecting
    # again), and both ends projected onto each face, filled on demand
    state = {'a': None, 'b': None, 'shape_a': None, 'shape_b': None, 'face_of_a': None, 'face_of_b': None}
    face_a: List[Optional[Face]] = [None] * len(origins)
    face_b: List[Optional[Face]] = [None] * len(origins)

    def project(origin_id: int) -> None:
        if face_a[origin_id] is None:
            sa, sb = state['shape_a'], state['shape_b']
            face_a[origin_id] = (state['face_of_a'] if state['face_of_a'] is not None and sa is not None
                                 and origin_id == sa['origin_id'] else _dodecahedron.forward(state['a'], origin_id))
            face_b[origin_id] = (state['face_of_b'] if state['face_of_b'] is not None and sb is not None
                                 and origin_id == sb['origin_id'] else _dodecahedron.forward(state['b'], origin_id))

    def touches(origin_id: int, quintant: int, triple: Triple) -> bool:
        project(origin_id)
        pentagon = get_pentagon_vertices(hilbert_res, quintant, triple, triple_flavor(triple, max_row))
        return pentagon.intersects_segment(face_a[origin_id], face_b[origin_id])

    def covers_exactly(shapes: List[dict]) -> bool:
        """
        Whether the sub-segment runs through the cells of `shapes` in turn, all on
        one origin, and through nothing else: from a (in the first) to b (in the
        last), the part inside each cell ends where the next one's begins, at a
        point well inside an edge, so no third cell meets it there.
        """
        origin_id = shapes[0]['origin_id']
        project(origin_id)
        prev_end = 0.0
        for i, shape in enumerate(shapes):
            if shape['origin_id'] != origin_id:
                return False
            part = _clip_to_pentagon(shape['pentagon'], face_a[origin_id], face_b[origin_id])
            if part is None:
                return False
            start, end, exit_edge_t = part
            if (start > _SHARED_EDGE_EPS) if i == 0 else (abs(start - prev_end) > _SHARED_EDGE_EPS):
                return False
            if i == len(shapes) - 1:
                return end >= 1 - _SHARED_EDGE_EPS
            if exit_edge_t <= _SHARED_EDGE_MARGIN or exit_edge_t >= 1 - _SHARED_EDGE_MARGIN:
                return False
            prev_end = end
        return False

    def settle(cell_a: int, shape_a: dict, cell_b: int, shape_b: dict, arc: int) -> bool:
        """
        Settle the sub-segment from cell A to cell B without the full search. It
        usually runs straight from A into B; failing that, it usually clips one
        cell C between them, found at the middle of the gap and then visited.
        False sends the sub-segment to the full search.
        """
        if shape_a['origin_id'] != shape_b['origin_id']:
            return False
        if covers_exactly([shape_a, shape_b]):
            return True
        origin_id = shape_a['origin_id']
        in_a = _clip_to_pentagon(shape_a['pentagon'], face_a[origin_id], face_b[origin_id])
        in_b = _clip_to_pentagon(shape_b['pentagon'], face_a[origin_id], face_b[origin_id])
        if in_a is None or in_b is None or in_b[0] <= in_a[1]:
            return False
        t = (in_a[1] + in_b[0]) / 2
        av = to_cartesian(state['a'])
        bv = to_cartesian(state['b'])
        m = [av[k] + (bv[k] - av[k]) * t for k in range(3)]
        length = math.sqrt(m[0] * m[0] + m[1] * m[1] + m[2] * m[2])
        cell_c = spherical_to_cell(to_spherical((m[0] / length, m[1] / length, m[2] / length)), resolution)
        shape_c = last_cell_shape(cell_c)
        if shape_c is None or cell_c == cell_a or cell_c == cell_b or not covers_exactly([shape_a, shape_c, shape_b]):
            return False
        visit(cell_c, arc)
        return True

    def search_subsegment(cell_a: int, cell_b: int, arc: int) -> None:
        """
        Strict local BFS: expand neighbors of every cell known to touch the
        sub-segment, keeping anything whose pentagon the sub-segment crosses.
        Terminates as soon as no new touching cells are found -- typically 1-2
        hops, since a sub-segment <= cellRadius/2 reaches at most a couple of
        cells beyond its endpoint cells.
        """
        frontier = cell_ids_to_triples([cell_a, cell_b])
        visited: Set[int] = {triple_cell_key(*frontier[0:5]), triple_cell_key(*frontier[5:10])}
        while frontier:
            next_frontier: List[int] = []

            def visit_neighbor(origin_id: int, quintant: int, x: int, y: int, z: int) -> None:
                key = triple_cell_key(origin_id, quintant, x, y, z)
                if key in visited:
                    return
                visited.add(key)
                if touches(origin_id, quintant, Triple(x, y, z)):
                    visit(triple_cell_to_id(origin_id, quintant, x, y, z, hilbert_res, resolution), arc)
                    next_frontier.extend((origin_id, quintant, x, y, z))

            for c in range(0, len(frontier), 5):
                for_each_triple_neighbor(frontier[c], frontier[c + 1], frontier[c + 2], frontier[c + 3],
                                         frontier[c + 4], max_row, False, visit_neighbor)
            frontier = next_frontier

    arcs = n if closed else n - 1
    for arc in range(arcs):
        end = (arc + 1) % n
        # Sample the great-circle at half-cell-radius spacing, endpoints included
        interior = sample_great_circle_arc(point_vecs[arc], point_vecs[end], sample_interval)
        last = len(interior) + 1

        cell_a = point_cells[arc]
        state['shape_a'] = point_shapes[arc]
        state['face_of_a'] = point_faces[arc]
        state['b'] = point_spherical[arc]
        visit(cell_a, arc)
        # Walk pairwise. Each (P_j, P_{j+1}) sub-segment is short enough that its
        # projection onto any nearby cell's Face is essentially straight, so we
        # can use exact 2D segment-vs-pentagon intersection.
        for j in range(1, last + 1):
            state['a'] = state['b']
            if j == last:
                b = point_spherical[end]
                cell_b = point_cells[end]
                shape_b = point_shapes[end]
                face_of_b = point_faces[end]
            else:
                b = to_spherical(interior[j - 1])
                cell_b = spherical_to_cell(b, resolution)
                shape_b = last_cell_shape(cell_b)
                face_of_b = None if shape_b is None else last_projection(b, shape_b['origin_id'])
            state['b'] = b
            state['shape_b'] = shape_b
            state['face_of_b'] = face_of_b
            visit(cell_b, arc)
            if cell_a != cell_b and exact:
                if resolution == 0:
                    _trace_faces(cell_a, cell_b, to_lonlat(state['a']), to_lonlat(b), lambda cell: visit(cell, arc))
                else:
                    for k in range(len(origins)):
                        face_a[k] = None
                        face_b[k] = None
                    shape_a = state['shape_a']
                    if shape_a is None or shape_b is None or not settle(cell_a, shape_a, cell_b, shape_b, arc):
                        search_subsegment(cell_a, cell_b, arc)
            cell_a = cell_b
            state['shape_a'] = shape_b
            state['face_of_a'] = face_of_b


def line_string_to_cells(waypoints: List[LonLat], resolution: int) -> List[int]:
    """
    Trace cells along a polyline defined by a sequence of waypoints.

    Consecutive waypoints are connected with great-circle arcs, traced by
    `trace_path`: every cell whose pentagon an arc touches. Cells at waypoint
    junctions are deduplicated.

    Pass [start, end] for a simple two-point line segment.

    Returns:
        Array of unique cell IDs along the polyline, in order.
    """
    if len(waypoints) == 0:
        return []
    if len(waypoints) == 1:
        return [lonlat_to_cell(waypoints[0], resolution)]

    seen: Set[int] = set()
    result: List[int] = []

    def add_cell(cell: int, _arc: int) -> None:
        if cell not in seen:
            seen.add(cell)
            result.append(cell)

    trace_path(waypoints, False, resolution, add_cell)
    return result
