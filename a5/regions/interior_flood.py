# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

# Polygon fill by flooding the interior: cheaper than curve runs when the
# interior is small, as the flood costs about boundary + interior cells while
# the runs sort a band of boundary plus ring keys.

import math
from typing import List

from ..core.coordinate_systems import Cartesian
from ..core.coordinate_transforms import to_cartesian
from ..core.serialization import FIRST_HILBERT_RESOLUTION
from ..core.compact import compact
from ..core.cell_info import get_num_cells
from ..traversal.lattice_flood_fill import triple_space_flood_fill
from ..traversal.triple_cells import for_each_lattice_neighbor, triple_cell_center, triple_cell_to_id
from .polygon_boundary import Boundary, boundary_neighbors, boundary_output, inside_next_to, polygon_area

# Below this many estimated interior cells per boundary cell, flooding the
# interior beats splitting the curve into runs (measured crossover: ~3.3).
_FLOOD_INTERIOR_PER_BOUNDARY = 3


def prefers_flood(
    ring_vecs_list: List[List[Cartesian]], boundary_count: int, resolution: int, cap_holds_quintant: bool,
) -> bool:
    """
    Whether to fill by flooding: the interior is small, and the polygon can't
    swallow a quintant whole (`cap_holds_quintant` false), which the flood, never
    crossing a quintant edge from the boundary, would miss.
    """
    return not cap_holds_quintant and (polygon_area(ring_vecs_list) / (4 * math.pi) * get_num_cells(resolution)
                                       < _FLOOD_INTERIOR_PER_BOUNDARY * boundary_count)


def fill_by_flood(boundary: Boundary, triples: List[int], resolution: int, overlapping: bool) -> List[int]:
    """
    Fill a polygon by flooding its interior, given its classified boundary and
    the boundary cells as flat triples.
    """
    hilbert_res = resolution - FIRST_HILBERT_RESOLUTION + 1
    max_row = (1 << hilbert_res) - 1
    out = boundary_output(boundary, overlapping)

    # The shell: the flood's own moves out of the boundary (each an edge
    # neighbor), split into seeds inside and firewall outside
    shell, parents = boundary_neighbors(triples, [
        lambda b, c, visit: for_each_lattice_neighbor(*b[c:c + 5], max_row, visit),
    ])
    seeds: List[int] = []
    firewall: List[int] = list(triples)
    for c in range(0, len(shell), 5):
        cell = shell[c:c + 5]
        center = to_cartesian(triple_cell_center(*cell, hilbert_res, max_row))
        (seeds if inside_next_to(boundary, center, parents[c // 5]) else firewall).extend(cell)
    if seeds:
        for c in range(0, len(seeds), 5):
            out.append(triple_cell_to_id(*seeds[c:c + 5], hilbert_res, resolution))
        out.extend(triple_space_flood_fill(firewall, seeds, resolution)['interior_cells'])
    return compact(out)
