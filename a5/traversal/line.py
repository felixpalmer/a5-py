# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

from typing import Callable, List, Optional, Set

from ..core.coordinate_systems import LonLat, Face
from ..core.cell import lonlat_to_cell, cell_intersects_segment
from ..core.coordinate_transforms import from_lonlat, to_cartesian, to_spherical, to_lonlat
from ..core.serialization import deserialize, serialize, FIRST_HILBERT_RESOLUTION
from ..core.origin import origins, segment_to_quintant
from ..core.face_adjacency import FACE_ADJACENCY
from ..core.tiling import get_pentagon_vertices
from ..lattice import Triple, s_to_triple, triple_flavor, triple_in_bounds
from ..projections.dodecahedron import DodecahedronProjection
from ..utils.great_circle import sample_great_circle_arc
from .cap import estimate_cell_radius
from .lattice_boundary import get_boundary_neighbor_triples
from .neighbors import NEIGHBOR_DELTAS
from .triple_cells import triple_cell_key, triple_cell_to_id

_dodecahedron = DodecahedronProjection()


def _trace_faces(cell_a: int, cell_b: int, a: LonLat, b: LonLat, add_cell: Callable[[int], None]) -> None:
    """
    Resolution 0 version of the sub-segment BFS below: the cells are the 12
    dodecahedron faces, adjacent across their edges.
    """
    frontier = [deserialize(cell_a)['origin'].id, deserialize(cell_b)['origin'].id]
    visited: Set[int] = set(frontier)
    while frontier:
        next_frontier: List[int] = []
        for face_id in frontier:
            for q in range(5):
                face = FACE_ADJACENCY[face_id][q][0]
                if face in visited:
                    continue
                visited.add(face)
                cell = serialize({'origin': origins[face], 'segment': 0, 'S': 0, 'resolution': 0})
                if cell_intersects_segment(cell, a, b):
                    add_cell(cell)
                    next_frontier.append(face)
        frontier = next_frontier


def line_string_to_cells(waypoints: List[LonLat], resolution: int) -> List[int]:
    """
    Trace cells along a polyline defined by a sequence of waypoints.

    Consecutive waypoints are connected with great-circle arcs. Each arc is
    sampled at half-cell-radius intervals; for each consecutive pair of samples,
    a strict local BFS finds every cell whose pentagon is touched by the
    straight 2D segment between the two samples (projected onto each candidate
    cell's Face). Cells at waypoint junctions are deduplicated.

    The BFS runs in triple space: a cell's neighbors come from its flavor's
    triple deltas plus the boundary delta tables, and its pentagon straight from
    its triple, so a candidate is never decoded and only touched cells are
    encoded.

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
    cell_radius = estimate_cell_radius(resolution)
    sample_interval = cell_radius * 0.5
    hilbert_res = resolution - FIRST_HILBERT_RESOLUTION + 1
    max_row = (1 << hilbert_res) - 1 if resolution > 0 else 0

    def add_cell(cell: int) -> None:
        if cell not in seen:
            seen.add(cell)
            result.append(cell)

    # The current sub-segment, projected onto each face it is tested against
    face_a: List[Optional[Face]] = [None] * len(origins)
    face_b: List[Optional[Face]] = [None] * len(origins)

    def touches(origin_id: int, quintant: int, triple: Triple, a: LonLat, b: LonLat) -> bool:
        if face_a[origin_id] is None:
            face_a[origin_id] = _dodecahedron.forward(from_lonlat(a), origin_id)
            face_b[origin_id] = _dodecahedron.forward(from_lonlat(b), origin_id)
        pentagon = get_pentagon_vertices(hilbert_res, quintant, triple, triple_flavor(triple, max_row))
        return pentagon.intersects_segment(face_a[origin_id], face_b[origin_id])

    boundary: List[int] = []
    for i in range(len(waypoints) - 1):
        start = waypoints[i]
        end = waypoints[i + 1]
        start_vec = to_cartesian(from_lonlat(start))
        end_vec = to_cartesian(from_lonlat(end))

        # Sample the great-circle at half-cell-radius spacing. Endpoints are
        # always included; even for short hops we get the start->end pair.
        interior = sample_great_circle_arc(start_vec, end_vec, sample_interval)
        num_subsegments = len(interior) + 1
        samples: List[LonLat] = [start] * (num_subsegments + 1)
        samples[0] = start
        samples[num_subsegments] = end
        for j in range(len(interior)):
            samples[j + 1] = to_lonlat(to_spherical(interior[j]))
        # Each sample's cell, as its ID and in triple space as flat (origin_id, quintant, x, y, z)
        sample_cells = [lonlat_to_cell(s, resolution) for s in samples]
        sample_triples: List[int] = []
        if resolution > 0:
            for cell_id in sample_cells:
                cell = deserialize(cell_id)
                quintant, orientation = segment_to_quintant(cell['segment'], cell['origin'])
                triple = s_to_triple(cell['S'], hilbert_res, orientation)
                sample_triples.extend((cell['origin'].id, quintant, triple.x, triple.y, triple.z))

        # Walk pairwise. Each (P_j, P_{j+1}) sub-segment is short enough that its
        # projection onto any nearby cell's Face is essentially straight, so we
        # can use exact 2D segment-vs-pentagon intersection.
        for j in range(num_subsegments):
            a = samples[j]
            b = samples[j + 1]
            cell_a = sample_cells[j]
            cell_b = sample_cells[j + 1]

            add_cell(cell_a)
            add_cell(cell_b)
            if cell_a == cell_b:
                continue
            if resolution == 0:
                _trace_faces(cell_a, cell_b, a, b, add_cell)
                continue
            for k in range(len(origins)):
                face_a[k] = None
                face_b[k] = None

            # Strict local BFS: expand neighbors of every cell known to touch this
            # sub-segment, keeping anything whose pentagon the sub-segment crosses.
            # Terminates as soon as no new touching cells are found -- typically 1-2
            # hops, since a sub-segment <= cellRadius/2 reaches at most a couple of
            # cells beyond its endpoint cells.
            frontier = sample_triples[j * 5:j * 5 + 10]
            visited: Set[int] = {triple_cell_key(*frontier[0:5]), triple_cell_key(*frontier[5:10])}
            while frontier:
                next_frontier: List[int] = []

                def visit(origin_id: int, quintant: int, x: int, y: int, z: int) -> None:
                    key = triple_cell_key(origin_id, quintant, x, y, z)
                    if key in visited:
                        return
                    visited.add(key)
                    if touches(origin_id, quintant, Triple(x, y, z), a, b):
                        add_cell(triple_cell_to_id(origin_id, quintant, x, y, z, hilbert_res, resolution))
                        next_frontier.extend((origin_id, quintant, x, y, z))

                for c in range(0, len(frontier), 5):
                    origin_id = frontier[c]
                    q = frontier[c + 1]
                    x = frontier[c + 2]
                    y = frontier[c + 3]
                    z = frontier[c + 4]
                    triple = Triple(x, y, z)

                    # Within the quintant: the fixed per-flavor deltas (edge and vertex neighbors)
                    for d in NEIGHBOR_DELTAS[triple_flavor(triple, max_row)].all:
                        neighbor = Triple(x + d.x, y + d.y, z + d.z)
                        if triple_in_bounds(neighbor, max_row):
                            visit(origin_id, q, neighbor.x, neighbor.y, neighbor.z)

                    # Across a quintant edge: the boundary delta tables
                    if x == 0 or z == 0 or y == max_row:
                        boundary.clear()
                        get_boundary_neighbor_triples(triple, x + y + z, q, origins[origin_id], max_row,
                                                      False, False, boundary)
                        for k in range(0, len(boundary), 5):
                            visit(boundary[k], boundary[k + 1], boundary[k + 2], boundary[k + 3], boundary[k + 4])
                frontier = next_frontier

    return result
