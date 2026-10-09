# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

import math
from typing import List
from ..core.coordinate_systems import Spherical
from ..core.serialization import (
    get_resolution, cell_to_parent, deserialize, serialize, FIRST_HILBERT_RESOLUTION,
)
from ..core.cell import cell_to_spherical
from ..core.cell_info import cell_area
from ..collections.slot_runs import SlotRuns, slot_runs_to_covering, to_covering
from ..core.constants import AUTHALIC_RADIUS_EARTH
from ..core.face_adjacency import walk_faces
from ..core.origin import haversine, origins
from ..projections.dodecahedron import DodecahedronProjection
from .curve_descent import descend_in_curve_order, INSIDE, OUTSIDE, SPLIT
from .triple_cells import cell_ids_to_triples, triple_cell_center, walk_triple_cells

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
    BFS at the cap's coarse resolution (1 or above) from `start_cell` through every
    cell whose center lies within `h_expanded` of `center`, returning those cells
    (the ring just outside lies beyond every threshold the descent applies).

    Runs in triple space (cells as flat (origin_id, quintant, x, y, z)): neighbors
    (edge and vertex) come from the per-flavor triple deltas plus the boundary
    delta tables, and a cell's center straight from its triple.
    """
    hilbert_res = get_resolution(start_cell) - FIRST_HILBERT_RESOLUTION + 1
    max_row = (1 << hilbert_res) - 1
    cells = cell_ids_to_triples([start_cell])

    def expand(origin_id: int, q: int, x: int, y: int, z: int) -> bool:
        within = haversine(center, triple_cell_center(origin_id, q, x, y, z, hilbert_res, max_row)) <= h_expanded
        if within:
            cells.extend((origin_id, q, x, y, z))
        return within

    walk_triple_cells(list(cells), max_row, expand)
    return cells


def spherical_cap(cell_id: int, radius: float) -> List[int]:
    """
    Compute all cells within a great-circle radius, returning a compacted result
    (mix of resolutions), with a compaction marker recording the resolution.

    Descends the hierarchy (see curve_descent): starts at a coarse resolution and
    subdivides boundary cells, keeping interior cells at coarser resolutions.
    Only cells whose centers fall within the radius are included.
    """
    target_res = get_resolution(cell_id)
    coarse_res = pick_coarse_resolution(radius, target_res)
    center = cell_to_spherical(cell_id)

    # Pre-compute haversine thresholds: the exact radius, and the radius expanded
    # so the coarse BFS captures every overlapping cell
    h_radius = meters_to_h(radius)
    h_expanded = meters_to_h(radius + estimate_cell_radius(coarse_res))
    start_cell = cell_to_parent(cell_id, coarse_res) if coarse_res < target_res else cell_id
    if coarse_res == 0:
        # The target is resolution 0: the cells are the 12 dodecahedron faces
        result: List[int] = []

        def face_cell(face: int) -> int:
            return serialize({'origin': origins[face], 'segment': 0, 'S': 0, 'resolution': 0})

        def near(face: int, h: float) -> bool:
            return haversine(center, cell_to_spherical(face_cell(face))) <= h

        for face in walk_faces([deserialize(start_cell)['origin'].id], lambda face: near(face, h_expanded)):
            if near(face, h_radius):
                result.append(face_cell(face))
        return to_covering(result, target_res)

    # Descend from the coarse cells to target_res, classifying each cell by
    # comparing haversine(center, cell) against pre-computed h thresholds:
    # - Interior (h <= h_inner): keep whole, all descendants inside
    # - Outside  (h > h_outer): discard, no descendants inside
    # - Boundary: split into children
    # At the target resolution both thresholds are the exact radius.
    h_inner = [0.0] * (target_res + 1)
    h_outer = [0.0] * (target_res + 1)
    for res in range(coarse_res, target_res + 1):
        cell_radius = estimate_cell_radius(res)
        last = res == target_res
        h_inner[res] = h_radius if last else (meters_to_h(radius - cell_radius) if radius > cell_radius else -1)
        h_outer[res] = h_radius if last else meters_to_h(radius + cell_radius)

    inverse = _dodecahedron.inverse

    def classify(origin_id: int, res: int, face, slot: int) -> int:
        h = haversine(center, inverse(face, origin_id))
        return INSIDE if h <= h_inner[res] else SPLIT if h <= h_outer[res] else OUTSIDE

    runs: SlotRuns = []
    descend_in_curve_order(
        _coarse_cap_cells(start_cell, center, h_expanded),
        coarse_res - FIRST_HILBERT_RESOLUTION + 1,
        target_res,
        classify,
        runs,
    )
    return slot_runs_to_covering(runs, target_res)
