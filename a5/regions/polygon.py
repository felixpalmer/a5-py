# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

import math
from typing import Dict, List, Optional, Sequence, Set, Tuple, TypedDict, Union

from ..core.coordinate_systems import LonLat, Cartesian
from ..core.cell import lonlat_to_cell, spherical_to_cell, cell_to_spherical
from ..core.coordinate_transforms import from_lonlat, to_cartesian, to_spherical
from ..core.serialization import (
    cell_to_children, deserialize, get_resolution, serialize,
    FIRST_HILBERT_RESOLUTION, MAX_RESOLUTION, WORLD_CELL,
)
from ..core.compact import compact
from ..geometry.spherical_polygon import ring_winding_sign
from ..geometry.prepared_polygon import (
    PreparedPolygon, prepare_polygon, point_in_prepared_polygon,
)
from ..traversal.cap import estimate_cell_radius
from ..utils.great_circle import sample_great_circle_arc
from ..traversal.lattice_flood_fill import triple_space_flood_fill
from ..traversal.triple_cells import (
    cell_ids_to_triples, for_each_lattice_neighbor, triple_cell_center, triple_cell_key, triple_cells_to_ids,
    triple_children, triple_parent,
)


# Maps each boundary cell to the indices of the ring segments that produced it.
# Segment indices are global across rings (outer ring first, then holes).
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


def _filter_boundary_cells(
    boundary_cells: List[int], segment_map: SegmentMap,
    seg_normals: List[Cartesian], seg_signs: List[int],
    prep: PreparedPolygon,
) -> List[int]:
    """
    Filter boundary cells to those whose center is inside the polygon.

    For each cell we know which ring segment(s) sampled it. When all of those
    segments place the cell on the interior side (cheap signed-dot test), we
    accept immediately. When they disagree (vertex / concave corner) or the
    cell wasn't recorded, fall back to full PIP.
    """
    out: List[int] = []
    for cell in boundary_cells:
        cv = to_cartesian(cell_to_spherical(cell))
        segments = segment_map.get(cell)
        if segments is None:
            if point_in_prepared_polygon(cv, prep):
                out.append(cell)
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
            if dot * seg_signs[seg_idx] > 0:
                any_inside = True
            else:
                all_inside = False
        if ambiguous or (any_inside and not all_inside):
            if point_in_prepared_polygon(cv, prep):
                out.append(cell)
        elif all_inside:
            out.append(cell)
    return out


def _expand_shell(boundary: List[int], max_row: int) -> List[int]:
    """
    Buffer the boundary by one cell using lattice neighbors, in triple space
    (cells as flat (origin_id, quintant, x, y, z)). The shell matches the
    connectivity of `triple_space_flood_fill` so the firewall (boundary + exterior
    shell) is a tight topological barrier for the subsequent flood.
    """
    seen: Set[int] = set()
    for c in range(0, len(boundary), 5):
        seen.add(triple_cell_key(*boundary[c:c + 5]))
    shell: List[int] = []

    def visit(origin_id: int, quintant: int, x: int, y: int, z: int) -> None:
        key = triple_cell_key(origin_id, quintant, x, y, z)
        if key in seen:
            return
        seen.add(key)
        shell.extend((origin_id, quintant, x, y, z))

    for c in range(0, len(boundary), 5):
        for_each_lattice_neighbor(*boundary[c:c + 5], max_row, visit)
    return shell


def _flood_interior(seeds: List[int], boundary: List[int], exterior_shell: List[int], resolution: int) -> List[int]:
    """
    Hierarchical flood fill from interior seed cells. Runs a few fine BFS layers
    to clear the boundary, then a coarse-resolution BFS through the bulk, then
    resumes fine BFS to fill gaps near the boundary. The coarse phase is skipped
    when the polygon is too small to amortize its setup overhead.

    All in triple space (cells as flat (origin_id, quintant, x, y, z)), moving
    between resolutions with `triple_parent` / `triple_children`; only the cells
    emitted are encoded.
    """
    hilbert_res = resolution - FIRST_HILBERT_RESOLUTION + 1
    firewall = boundary + exterior_shell

    # Isoperimetric bound: B^2 / (4*pi) is the max interior for B boundary cells.
    boundary_size = len(boundary) // 5
    max_interior = boundary_size * boundary_size / (4 * math.pi)
    # res 30 has a different encoding the parent-emit optimization can't use.
    use_coarse_phase = (
        resolution > FIRST_HILBERT_RESOLUTION
        and resolution < MAX_RESOLUTION
        and max_interior > 1000
    )

    if not use_coarse_phase:
        result = triple_space_flood_fill(firewall, seeds, resolution)
        return triple_cells_to_ids(seeds + result['interior'], hilbert_res, resolution)

    parent_max_row = (1 << (hilbert_res - 1)) - 1

    def parents(cells: List[int]) -> List[int]:
        out: List[int] = []
        for c in range(0, len(cells), 5):
            triple_parent(*cells[c:c + 5], parent_max_row, out)
        return out

    def key(cells: List[int], c: int) -> int:
        return triple_cell_key(*cells[c:c + 5])

    coarse_firewall = parents(firewall + seeds)

    # Phase 1: short fine BFS to move the frontier off the boundary.
    phase1 = triple_space_flood_fill(firewall, seeds, resolution, 3)

    # Phase 2: coarse BFS through the bulk interior, seeded by the parents of the
    # phase 1 frontier that aren't firewall parents.
    seen = {key(coarse_firewall, c) for c in range(0, len(coarse_firewall), 5)}
    frontier_parents = parents(phase1['frontier'])
    coarse_seeds: List[int] = []
    for c in range(0, len(frontier_parents), 5):
        k = key(frontier_parents, c)
        if k not in seen:
            seen.add(k)
            coarse_seeds.extend(frontier_parents[c:c + 5])
    coarse_interior: List[int] = []
    phase3_delta: List[int] = []
    if coarse_seeds:
        coarse_interior = coarse_seeds + triple_space_flood_fill(coarse_firewall, coarse_seeds, resolution - 1)['interior']
        # Children become firewall for phase 3; the coarse parent represents
        # them in the output, so we don't emit them individually.
        for c in range(0, len(coarse_interior), 5):
            triple_children(*coarse_interior[c:c + 5], parent_max_row, phase3_delta)

    # Phase 3: resume fine BFS, reusing phase 1's state.
    phase3 = triple_space_flood_fill(
        {'state': phase1['state'], 'delta': phase3_delta},
        phase1['frontier'],
        resolution,
    )

    # Emit fine cells only when not already covered by a coarse parent.
    covered = {key(coarse_interior, c) for c in range(0, len(coarse_interior), 5)}
    fine = seeds + phase1['interior']
    fine_parents = parents(fine)
    emitted: List[int] = []
    for c in range(0, len(fine), 5):
        if key(fine_parents, c) not in covered:
            emitted.extend(fine[c:c + 5])
    out = triple_cells_to_ids(emitted + phase3['interior'], hilbert_res, resolution)
    return triple_cells_to_ids(coarse_interior, hilbert_res - 1, resolution - 1, out)


def _swallowed_quintants(
    boundary: List[int],
    shell: List[int],
    resolution: int,
    prep: PreparedPolygon,
) -> List[int]:
    """
    Quintants the polygon swallows whole. The flood fill never crosses a
    quintant edge, so such a quintant gets no seeds from the boundary shell and
    would be left empty. A quintant holding none of the boundary or shell cells
    has none of the polygon's edge passing through it: its cells lie wholly
    inside or wholly outside, and a single probe cell decides which. Inside
    quintants are emitted as their resolution 1 cell (resolution 0 when that is
    the target), which `compact` merges with the rest of the output.
    """
    # A swallowed quintant lies inside the polygon's bounding cap, so the cap
    # must have at least a quintant's area (4pi/60: cells are equal-area)
    if 2 * math.pi * (1 - prep.cap.min_dot) < (4 * math.pi) / 60:
        return []
    # Quintants by origin.id * 5 + quintant, as the triples carry them
    touched: Set[int] = set()
    for cells in (boundary, shell):
        for c in range(0, len(cells), 5):
            touched.add(cells[c] * 5 + cells[c + 1])

    out: List[int] = []
    quintant_cells = cell_to_children(WORLD_CELL, FIRST_HILBERT_RESOLUTION - 1)
    quintants = cell_ids_to_triples(quintant_cells)
    for i, quintant_cell in enumerate(quintant_cells):
        if quintants[i * 5] * 5 + quintants[i * 5 + 1] in touched:
            continue
        # Any cell of the quintant at the target resolution will do
        probe = serialize({**deserialize(quintant_cell), 'S': 0, 'resolution': resolution})
        if point_in_prepared_polygon(to_cartesian(cell_to_spherical(probe)), prep):
            out.append(quintant_cell)
    return out


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

    # The boundary contribution to the output. In 'overlapping' mode every
    # densely-sampled boundary cell contains a point on the polygon boundary, so
    # it overlaps the polygon -- keep them all, unfiltered. In 'center' mode we
    # filter down to those whose center lies inside.
    if containment == 'overlapping':
        boundary_out = boundary_cells
    else:
        # Flattened per-segment normals and interior-side signs, indexed like the
        # segment map. The polygon interior lies on the *outside* of a hole ring,
        # so hole segments get the opposite sign.
        seg_normals: List[Cartesian] = []
        seg_signs: List[int] = []
        for r in range(len(rings)):
            sign = (1 if r == 0 else -1) * ring_winding_sign(ring_vecs_list[r])
            normals = prep.ring_normals[r]
            for normal in normals:
                seg_normals.append(normal)
                seg_signs.append(sign)
        boundary_out = _filter_boundary_cells(boundary_cells, segment_map, seg_normals, seg_signs, prep)

    # Resolutions 0 and 1 have no lattice to flood (a quintant is a single
    # cell): every cell off the boundary is in or out by its center, and there
    # are at most 60 of them.
    if resolution < FIRST_HILBERT_RESOLUTION:
        out = list(boundary_out)
        for cell in cell_to_children(WORLD_CELL, resolution):
            if cell not in boundary_set and point_in_prepared_polygon(to_cartesian(cell_to_spherical(cell)), prep):
                out.append(cell)
        return compact(out)

    # The rest runs in triple space: cells as flat (origin_id, quintant, x, y, z)
    hilbert_res = resolution - FIRST_HILBERT_RESOLUTION + 1
    max_row = (1 << hilbert_res) - 1
    boundary = cell_ids_to_triples(boundary_cells)

    # Dense sampling can leave gaps; the shell catches them, classifying each cell.
    shell = _expand_shell(boundary, max_row)
    swallowed = _swallowed_quintants(boundary, shell, resolution, prep)
    if len(shell) == 0:
        return compact(boundary_out + swallowed)

    seeds: List[int] = []
    exterior_shell: List[int] = []  # exterior shell (and hole interiors) join the firewall
    for c in range(0, len(shell), 5):
        cell = shell[c:c + 5]
        center = triple_cell_center(*cell, hilbert_res, max_row)
        (seeds if point_in_prepared_polygon(to_cartesian(center), prep) else exterior_shell).extend(cell)
    if len(seeds) == 0:
        return compact(boundary_out + swallowed)

    interior_cells = _flood_interior(seeds, boundary, exterior_shell, resolution)

    return compact(boundary_out + interior_cells + swallowed)
