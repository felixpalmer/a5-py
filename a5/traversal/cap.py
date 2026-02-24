# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

import math
from typing import List
from ..core.serialization import get_resolution, cell_to_parent, cell_to_children, FIRST_HILBERT_RESOLUTION
from ..core.cell import cell_to_spherical
from ..core.cell_info import cell_area
from ..core.constants import AUTHALIC_RADIUS_EARTH
from .global_neighbors import get_global_cell_neighbors
from ..core.origin import haversine

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
    cap_area_m2 = 2 * math.pi * AUTHALIC_RADIUS_EARTH * AUTHALIC_RADIUS_EARTH * \
        (1 - math.cos(radius / AUTHALIC_RADIUS_EARTH))

    for res in range(FIRST_HILBERT_RESOLUTION, target_res + 1):
        c_area = cell_area(res)
        if cap_area_m2 / c_area >= MIN_CELLS_FOR_SUBDIVISION:
            return res
    return target_res  # No coarsening benefit


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
    coarse_visited = {start_cell}
    coarse_frontier = {start_cell}

    while len(coarse_frontier) > 0:
        next_frontier = set()
        for cid in coarse_frontier:
            for neighbor in get_global_cell_neighbors(cid):
                if neighbor in coarse_visited:
                    continue
                coarse_visited.add(neighbor)
                if haversine(center, cell_to_spherical(neighbor)) <= h_expanded:
                    next_frontier.add(neighbor)
        coarse_frontier = next_frontier

    # Recursive subdivision from coarseRes to targetRes.
    result: List[int] = []
    boundary = list(coarse_visited)

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
