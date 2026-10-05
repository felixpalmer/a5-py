# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

import math
from typing import List, Optional, Sequence, TypedDict, Union

from ..core.coordinate_systems import LonLat, Cartesian
from ..core.cell import cell_to_spherical
from ..core.coordinate_transforms import from_lonlat, to_cartesian
from ..core.serialization import cell_to_children, get_resolution, FIRST_HILBERT_RESOLUTION, MAX_RESOLUTION, WORLD_CELL
from ..core.compact import compact
from ..geometry.prepared_polygon import prepare_polygon, point_in_prepared_polygon
from ..traversal.triple_cells import cell_ids_to_triples
from .polygon_boundary import boundary_output, classify_boundary, sample_boundary
from .curve_runs import fill_by_curve_runs
from .interior_flood import fill_by_flood, prefers_flood


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
    sampled = sample_boundary(rings, ring_vecs_list, resolution)

    # Res 30 covers only quintants 0-41 (elsewhere A5 answers at res 29, see
    # serialize), so a polygon reaching past them is filled at res 29: mixing the
    # two lattices would leave the fill without a consistent grid.
    if resolution == MAX_RESOLUTION and any(get_resolution(cell) != resolution for cell in sampled[0]):
        return polygon_to_cells(polygon, resolution - 1, options)

    boundary = classify_boundary(sampled, ring_vecs_list, prep)
    overlapping = containment == 'overlapping'

    # Resolutions 0 and 1 have no lattice (a quintant is a single cell): every
    # cell off the boundary is in or out by its center, and there are at most 60
    # of them.
    if resolution < FIRST_HILBERT_RESOLUTION:
        out = boundary_output(boundary, overlapping)
        for cell in cell_to_children(WORLD_CELL, resolution):
            if cell not in boundary.set and point_in_prepared_polygon(to_cartesian(cell_to_spherical(cell)), prep):
                out.append(cell)
        return compact(out)

    # A quintant holding no boundary cells is wholly inside or outside; it can
    # only be inside when the polygon's bounding cap holds a quintant's area (4pi/60)
    cap_holds_quintant = 2 * math.pi * (1 - prep.cap.min_dot) >= (4 * math.pi) / 60
    triples = cell_ids_to_triples(boundary.cells)
    if prefers_flood(ring_vecs_list, len(boundary.cells), resolution, cap_holds_quintant):
        return fill_by_flood(boundary, triples, resolution, overlapping)
    return fill_by_curve_runs(boundary, triples, resolution, overlapping, cap_holds_quintant)
