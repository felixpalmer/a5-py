# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

import math
from typing import List
from ..core.coordinate_systems import Spherical
from ..core.serialization import (
    get_resolution, cell_to_parent, cell_to_children, deserialize, serialize, FIRST_HILBERT_RESOLUTION,
)
from ..core.cell import cell_to_spherical
from ..core.cell_info import cell_area
from ..core.constants import AUTHALIC_RADIUS_EARTH
from ..core.face_adjacency import FACE_ADJACENCY
from ..core.tiling import get_pentagon_center
from ..core.origin import haversine, origins, segment_to_quintant
from ..lattice import Triple, s_to_triple, triple_flavor, triple_in_bounds
from ..projections.dodecahedron import DodecahedronProjection
from .lattice_boundary import get_boundary_neighbor_triples
from .neighbors import NEIGHBOR_DELTAS
from .triple_cells import triple_cell_key, triple_cell_to_id

_dodecahedron = DodecahedronProjection()

# Safety factor applied to equal-area circle radius to get conservative circumradius estimate
CELL_RADIUS_SAFETY_FACTOR = 2.0

# Minimum cells in the cap before hierarchical subdivision is worthwhile
MIN_CELLS_FOR_SUBDIVISION = 20

# Pre-compute cell radii
# Derived from: cellRadius = SAFETY * sqrt(cellArea / pi)
#             = SAFETY * sqrt(4*pi*R^2 / (numCells * pi))
#             = SAFETY * 2R / sqrt(numCells)
# For r >= 1: numCells = 60 * 4^(r-1), so sqrt(numCells) = 2*sqrt(15) * 2^(r-1)
# giving: cellRadius(r) = BASE / 2^(r-1) — halves at each resolution level.
_BASE_CELL_RADIUS = CELL_RADIUS_SAFETY_FACTOR * AUTHALIC_RADIUS_EARTH / math.sqrt(15)
_cell_radius: List[float] = [
    CELL_RADIUS_SAFETY_FACTOR * AUTHALIC_RADIUS_EARTH / math.sqrt(3)
] + [
    _BASE_CELL_RADIUS / (1 << (r - 1))
    for r in range(1, 31)
]


def meters_to_h(meters: float) -> float:
    """
    Convert a distance in meters to a haversine threshold value.
    Since haversine h = sin^2(d/2R) is monotonic in d for d in [0, piR],
    comparing h <= threshold is equivalent to comparing dist <= radius
    but avoids the asin/sqrt per point.
    """
    s = math.sin(meters / (2 * AUTHALIC_RADIUS_EARTH))
    return s * s


def estimate_cell_radius(resolution: int) -> float:
    """Estimate a conservative cell circumradius in meters for a given resolution."""
    return _cell_radius[resolution]


def pick_coarse_resolution(radius: float, target_res: int) -> int:
    """
    Pick the coarsest resolution where the cap contains enough cells
    to make hierarchical subdivision worthwhile.
    """
    # Spherical cap area in m^2: 2*pi*R^2*(1 - cos(r/R)) computed as
    # 4*pi*R^2*sin^2(r/2R), which keeps full precision for small radii
    # where 1 - cos cancels
    half_angle_sin = math.sin(radius / (2 * AUTHALIC_RADIUS_EARTH))
    cap_area_m2 = 4 * math.pi * AUTHALIC_RADIUS_EARTH * AUTHALIC_RADIUS_EARTH * half_angle_sin * half_angle_sin

    for res in range(FIRST_HILBERT_RESOLUTION, target_res + 1):
        c_area = cell_area(res)
        if cap_area_m2 / c_area >= MIN_CELLS_FOR_SUBDIVISION:
            return res
    return target_res  # No coarsening benefit


def _coarse_cap_cells(start_cell: int, center: Spherical, h_expanded: float) -> List[int]:
    """
    BFS at the cap's coarse resolution from `start_cell` through every cell whose
    center lies within `h_expanded` of `center`, returning every cell reached: the
    cells within, plus the ring just outside (the subdivision classifies them).

    Runs in triple space: neighbors (edge and vertex) come from the per-flavor
    triple deltas plus the boundary delta tables, and a cell's center straight
    from its triple, so no cell is decoded and each is encoded once.
    """
    cell = deserialize(start_cell)
    origin = cell['origin']
    resolution = cell['resolution']
    if resolution == 0:
        # The cells are the 12 dodecahedron faces, adjacent across their edges
        visited_faces = {origin.id}
        frontier_faces = [origin.id]
        while frontier_faces:
            next_faces: List[int] = []
            for face_id in frontier_faces:
                for q in range(5):
                    face = FACE_ADJACENCY[face_id][q][0]
                    if face in visited_faces:
                        continue
                    visited_faces.add(face)
                    face_cell = serialize({'origin': origins[face], 'segment': 0, 'S': 0, 'resolution': 0})
                    if haversine(center, cell_to_spherical(face_cell)) <= h_expanded:
                        next_faces.append(face)
            frontier_faces = next_faces
        return [serialize({'origin': origins[i], 'segment': 0, 'S': 0, 'resolution': 0}) for i in visited_faces]

    hilbert_res = resolution - FIRST_HILBERT_RESOLUTION + 1
    max_row = (1 << hilbert_res) - 1
    quintant, orientation = segment_to_quintant(cell['segment'], origin)
    seed = s_to_triple(cell['S'], hilbert_res, orientation)
    visited = {triple_cell_key(origin.id, quintant, seed.x, seed.y, seed.z)}
    cells: List[int] = [start_cell]
    frontier: List[int] = [origin.id, quintant, seed.x, seed.y, seed.z]
    boundary: List[int] = []

    while frontier:
        next_frontier: List[int] = []

        def visit(origin_id: int, q: int, x: int, y: int, z: int) -> None:
            key = triple_cell_key(origin_id, q, x, y, z)
            if key in visited:
                return
            visited.add(key)
            cells.append(triple_cell_to_id(origin_id, q, x, y, z, hilbert_res, resolution))
            triple = Triple(x, y, z)
            face = get_pentagon_center(hilbert_res, q, triple, triple_flavor(triple, max_row))
            if haversine(center, _dodecahedron.inverse(face, origin_id)) <= h_expanded:
                next_frontier.extend((origin_id, q, x, y, z))

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
    return cells


def spherical_cap(cell_id: int, radius: float) -> List[int]:
    """
    Compute all cells within a great-circle radius, returning a naturally
    compacted result (mix of resolutions).

    Uses hierarchical BFS: starts at a coarse resolution and recursively
    subdivides boundary cells, keeping interior cells at coarser resolutions.
    Only cells whose centers fall within the radius are included.
    """
    target_res = get_resolution(cell_id)
    coarse_res = pick_coarse_resolution(radius, target_res)
    center = cell_to_spherical(cell_id)

    # Pre-compute haversine threshold for the exact radius
    h_radius = meters_to_h(radius)

    # BFS at coarse resolution with expanded radius to capture all overlapping cells.
    start_cell = cell_to_parent(cell_id, coarse_res) if coarse_res < target_res else cell_id
    coarse_cell_radius = estimate_cell_radius(coarse_res)
    h_expanded = meters_to_h(radius + coarse_cell_radius)
    coarse_cells = _coarse_cap_cells(start_cell, center, h_expanded)

    # Recursive subdivision from coarseRes to targetRes.
    result: List[int] = []
    boundary = coarse_cells

    for res in range(coarse_res, target_res):
        cell_radius_val = estimate_cell_radius(res)
        h_inner = meters_to_h(radius - cell_radius_val) if radius > cell_radius_val else -1
        h_outer = meters_to_h(radius + cell_radius_val)
        next_boundary: List[int] = []

        for cell in boundary:
            h = haversine(center, cell_to_spherical(cell))
            if h <= h_inner:
                result.append(cell)
            elif h > h_outer:
                # Cell's entire extent is outside the cap -- discard
                pass
            else:
                for child in cell_to_children(cell, res + 1):
                    next_boundary.append(child)

        boundary = next_boundary

    # Final target resolution: strict haversine check
    for cell in boundary:
        if haversine(center, cell_to_spherical(cell)) <= h_radius:
            result.append(cell)

    result.sort()
    return result
