# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

import math
from typing import Dict, List, Sequence, Set, Tuple, Union

from ..core.coordinate_systems import LonLat, Cartesian
from ..core.cell import lonlat_to_cell, spherical_to_cell, cell_to_spherical
from ..core.coordinate_transforms import from_lonlat, to_cartesian, to_spherical
from ..core.serialization import (
    cell_to_parent, cell_to_children, FIRST_HILBERT_RESOLUTION, MAX_RESOLUTION,
)
from ..core.compact import compact
from ..geometry.spherical_polygon import (
    point_in_spherical_polygon, ring_winding_sign, ring_segment_normals,
)
from ..traversal.cap import estimate_cell_radius
from ..utils.great_circle import sample_great_circle_arc
from ..traversal.lattice_neighbors import get_lattice_neighbors
from ..traversal.lattice_flood_fill import triple_space_flood_fill


# Maps each boundary cell to the indices of the ring segments that produced it.
# Segment indices are global across rings (outer ring first, then holes).
SegmentMap = Dict[int, List[int]]


def _point_in_polygon_rings(point: Cartesian, ring_vecs_list: List[List[Cartesian]]) -> bool:
    """
    Point-in-polygon for a polygon with holes: inside the outer ring and
    outside every hole ring.
    """
    if not point_in_spherical_polygon(point, ring_vecs_list[0]):
        return False
    for ring_vecs in ring_vecs_list[1:]:
        if point_in_spherical_polygon(point, ring_vecs):
            return False
    return True


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
    ring_vecs_list: List[List[Cartesian]],
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
            if _point_in_polygon_rings(cv, ring_vecs_list):
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
            if _point_in_polygon_rings(cv, ring_vecs_list):
                out.append(cell)
        elif all_inside:
            out.append(cell)
    return out


def _expand_shell(boundary_cells: List[int], boundary_set: Set[int]) -> List[int]:
    """
    Buffer the boundary by one cell using 3-edge lattice neighbors. The shell
    matches the connectivity of `triple_space_flood_fill` so the firewall (boundary
    + exterior shell) is a tight topological barrier for the subsequent flood.
    """
    shell_cells: List[int] = []
    shell_set: Set[int] = set()
    for cell in boundary_cells:
        for neighbor in get_lattice_neighbors(cell, True):
            if neighbor in boundary_set:
                continue
            if neighbor not in shell_set:
                shell_set.add(neighbor)
                shell_cells.append(neighbor)
    return shell_cells


def _flood_interior(
    interior_seeds: List[int], visited: Set[int], boundary_size: int, resolution: int,
) -> List[int]:
    """
    Hierarchical flood fill from interior seed cells. Runs a few fine BFS layers
    to clear the boundary, then a coarse-resolution BFS through the bulk, then
    resumes fine BFS to fill gaps near the boundary. The coarse phase is skipped
    when the polygon is too small to amortize its setup overhead.
    """
    for cell in interior_seeds:
        visited.add(cell)

    # Isoperimetric bound: B^2 / (4*pi) is the max interior for B boundary cells.
    max_interior = boundary_size * boundary_size / (4 * math.pi)
    # res 30 has a different encoding the parent-emit optimization can't use.
    use_coarse_phase = (
        resolution > FIRST_HILBERT_RESOLUTION
        and resolution < MAX_RESOLUTION
        and max_interior > 1000
    )

    if not use_coarse_phase:
        result = triple_space_flood_fill(visited, interior_seeds, resolution)
        return list(interior_seeds) + result['interior_cells']

    parent_res = resolution - 1
    coarse_firewall: Set[int] = set()
    for cell in visited:
        coarse_firewall.add(cell_to_parent(cell, parent_res))

    # Phase 1: short fine BFS to move the frontier off the boundary.
    phase1 = triple_space_flood_fill(visited, interior_seeds, resolution, 3)

    # Phase 2: coarse BFS through the bulk interior.
    coarse_interior_set = None
    phase3_delta: List[int] = []
    coarse_interior_cells: List[int] = []
    if len(phase1['frontier_cell_ids']) > 0:
        coarse_seeds: Set[int] = set()
        for cell in phase1['frontier_cell_ids']:
            parent = cell_to_parent(cell, parent_res)
            if parent not in coarse_firewall:
                coarse_seeds.add(parent)

        if len(coarse_seeds) > 0:
            coarse_visited = set(coarse_firewall)
            for seed in coarse_seeds:
                coarse_visited.add(seed)
            coarse_result = triple_space_flood_fill(coarse_visited, list(coarse_seeds), parent_res)
            coarse_interior = list(coarse_seeds) + coarse_result['interior_cells']
            coarse_interior_set = set(coarse_interior)
            coarse_interior_cells.extend(coarse_interior)

            # Children become firewall for phase 3; the coarse parent represents
            # them in the output, so we don't emit them individually.
            for coarse_cell in coarse_interior:
                for child in cell_to_children(coarse_cell, resolution):
                    if child not in visited:
                        visited.add(child)
                        phase3_delta.append(child)

    # Emit fine cells only when not already covered by a coarse parent.
    interior_cells: List[int] = []
    if coarse_interior_set is None:
        interior_cells.extend(interior_seeds)
        interior_cells.extend(phase1['interior_cells'])
    else:
        for cell in interior_seeds:
            if cell_to_parent(cell, parent_res) not in coarse_interior_set:
                interior_cells.append(cell)
        for cell in phase1['interior_cells']:
            if cell_to_parent(cell, parent_res) not in coarse_interior_set:
                interior_cells.append(cell)
        interior_cells.extend(coarse_interior_cells)

    # Phase 3: resume fine BFS, reusing phase 1's packed state.
    phase3 = triple_space_flood_fill(
        {'state': phase1['state'], 'delta': phase3_delta},
        phase1['frontier_cell_ids'],
        resolution,
    )
    interior_cells.extend(phase3['interior_cells'])

    return interior_cells


def _strip_closing(ring: List[LonLat]) -> List[LonLat]:
    """GeoJSON rings repeat the first vertex at the end -- drop the duplicate."""
    last = len(ring) - 1
    if last > 0 and ring[0][0] == ring[last][0] and ring[0][1] == ring[last][1]:
        return ring[:-1]
    return ring


def polygon_to_cells(
    polygon: Union[Sequence[LonLat], Sequence[Sequence[LonLat]]], resolution: int,
) -> List[int]:
    """
    Find all cells within a polygon using center-point containment: a cell is
    included iff its center lies inside the polygon. The result is compacted --
    use `uncompact` to expand to the input resolution.

    Args:
        polygon: Either a single ring of [longitude, latitude] vertices, or
            GeoJSON-style rings `[outer, *holes]` where cells inside a hole are
            excluded. Rings may be open or closed (GeoJSON-style, first vertex
            repeated at the end) -- closure is automatic either way. Holes with
            fewer than 3 distinct vertices are ignored.
        resolution: Target resolution (0..30)

    Returns:
        Sorted, compacted list of cell IDs whose centers lie inside the polygon
    """
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

    boundary_cells, boundary_set, segment_map = _dense_sample_boundary(rings, ring_vecs_list, resolution)

    # Flattened per-segment normals and interior-side signs, indexed like the
    # segment map. The polygon interior lies on the *outside* of a hole ring,
    # so hole segments get the opposite sign.
    seg_normals: List[Cartesian] = []
    seg_signs: List[int] = []
    for r in range(len(rings)):
        sign = (1 if r == 0 else -1) * ring_winding_sign(ring_vecs_list[r])
        normals = ring_segment_normals(ring_vecs_list[r])
        for normal in normals:
            seg_normals.append(normal)
            seg_signs.append(sign)

    filtered_boundary = _filter_boundary_cells(boundary_cells, segment_map, seg_normals, seg_signs, ring_vecs_list)

    # Dense sampling can leave gaps; the shell catches them, classifying each cell.
    shell_cells = _expand_shell(boundary_cells, boundary_set)
    if len(shell_cells) == 0:
        return compact(filtered_boundary)

    interior_seeds: List[int] = []
    visited: Set[int] = set(boundary_set)
    for cell in shell_cells:
        if _point_in_polygon_rings(to_cartesian(cell_to_spherical(cell)), ring_vecs_list):
            interior_seeds.append(cell)
        else:
            visited.add(cell)  # exterior shell (and hole interiors) join the firewall
    if len(interior_seeds) == 0:
        return compact(filtered_boundary)

    interior_cells = _flood_interior(interior_seeds, visited, len(boundary_set), resolution)

    return compact(filtered_boundary + interior_cells)
