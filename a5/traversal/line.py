# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

import math
from typing import List, Set, cast

from ..core.coordinate_systems import LonLat, Cartesian
from ..core.cell import lonlat_to_cell, cell_intersects_segment
from ..core.coordinate_transforms import from_lonlat, to_cartesian, to_spherical, to_lonlat
from ..core.constants import AUTHALIC_RADIUS_EARTH
from ..math import vec3
from .cap import estimate_cell_radius
from .lattice_neighbors import get_lattice_neighbors


def line_string_to_cells(waypoints: List[LonLat], resolution: int) -> List[int]:
    """
    Trace cells along a polyline defined by a sequence of waypoints.

    Consecutive waypoints are connected with great-circle arcs. Each arc is
    sampled at half-cell-radius intervals; for each consecutive pair of samples,
    a strict local BFS finds every cell whose pentagon is touched by the
    straight 2D segment between the two samples (projected onto each candidate
    cell's Face). Cells at waypoint junctions are deduplicated.

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

    def add_cell(cell: int) -> None:
        if cell not in seen:
            seen.add(cell)
            result.append(cell)

    for i in range(len(waypoints) - 1):
        start = waypoints[i]
        end = waypoints[i + 1]
        start_vec = to_cartesian(from_lonlat(start))
        end_vec = to_cartesian(from_lonlat(end))
        dot = max(-1.0, min(1.0, vec3.dot(start_vec, end_vec)))
        dist = math.acos(dot) * AUTHALIC_RADIUS_EARTH

        # Sample the great-circle at half-cell-radius spacing. The endpoints are
        # always included; num_subsegments >= 1 (so we always get the start->end pair
        # even for short waypoint-to-waypoint hops).
        num_subsegments = max(1, math.ceil(dist / sample_interval))
        samples: List[LonLat] = [start] * (num_subsegments + 1)
        samples[0] = start
        samples[num_subsegments] = end
        if num_subsegments > 1:
            tmp = vec3.create()
            for j in range(1, num_subsegments):
                vec3.slerp(tmp, start_vec, end_vec, j / num_subsegments)
                samples[j] = to_lonlat(to_spherical(cast(Cartesian, (tmp[0], tmp[1], tmp[2]))))
        sample_cells = [lonlat_to_cell(s, resolution) for s in samples]

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

            # Strict local BFS: expand neighbors of every cell known to touch this
            # sub-segment, keeping anything whose pentagon the sub-segment crosses.
            # Terminates as soon as no new touching cells are found -- typically 1-2
            # hops, since a sub-segment <= cellRadius/2 reaches at most a couple of
            # cells beyond its endpoint cells.
            visited: Set[int] = {cell_a, cell_b}
            frontier: List[int] = [cell_a, cell_b]
            while frontier:
                next_frontier: List[int] = []
                for cell in frontier:
                    for neighbor in get_lattice_neighbors(cell, False):
                        if neighbor in visited:
                            continue
                        visited.add(neighbor)
                        if cell_intersects_segment(neighbor, a, b):
                            add_cell(neighbor)
                            next_frontier.append(neighbor)
                frontier = next_frontier

    return result
