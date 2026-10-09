# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

# The spatial counterpart of the cell hierarchy. The index hierarchy nests
# cells by ID, but a cell's children do not tile it exactly: some stick out,
# and children of its neighbors poke in. Here a finer cell belongs to the
# coarser cell holding its center, so every resolution partitions every coarser
# one exactly.

import math
from typing import List, Sequence

from ..core.cell import _get_pentagon, cell_to_spherical, spherical_to_cell
from ..core.face_adjacency import FACE_ADJACENCY, seam_transform
from ..core.serialization import (
    deserialize, get_resolution, slot_to_cell, FIRST_HILBERT_RESOLUTION, MAX_RESOLUTION, RES30_QUINTANTS, WORLD_CELL,
)
from ..core.tiling import get_face_vertices
from ..coverings.slot_runs import SlotRuns, slot_runs_to_covering, to_covering
from ..traversal.curve_descent import descend_in_curve_order, INSIDE, OUTSIDE, SPLIT

# How far the center of any descendant of a cell can lie from the cell's own
# center, in units of the cell's lattice spacing (face units * 2^hilbert_res). A
# child's center is within 0.342 of its parent's (the max over every flavor and
# child), and the offsets halve each level down, so all descendants lie within
# 2 * 0.342 (observed: 0.648).
_DESCENDANT_REACH = 0.7

# Face-unit distance from the parent's edges below which a center is decided by
# `cell_to_supercell` itself, so ties and float noise resolve exactly as it
# does (fine cells at resolution 30 are ~2e-9 across).
_EDGE_EPS = 1e-12


def cell_to_supercell(cell: int, resolution: int) -> int:
    """
    The cell at a coarser `resolution` that contains the center of `cell`: the
    spatial counterpart of `cell_to_parent`. Unlike the parent, the supercell
    always contains (the center of) the cell, so aggregating by supercell
    attributes each fine cell to the coarse cell it lies in.

    Args:
        cell: The cell
        resolution: Target resolution, at most the cell's own

    Returns:
        The cell at `resolution` containing the center of `cell`
    """
    cell_resolution = get_resolution(cell)
    if resolution > cell_resolution:
        raise ValueError(
            f'Target resolution ({resolution}) must be equal to or less than current resolution ({cell_resolution})'
        )
    if resolution == cell_resolution:
        return cell
    return spherical_to_cell(cell_to_spherical(cell), resolution)


def cell_to_subcell(cell: int, resolution: int) -> List[int]:
    """
    The cells at a finer `resolution` whose centers lie in `cell`: the spatial
    counterpart of `cell_to_children`, and the inverse of `cell_to_supercell` -- a
    cell is a subcell of exactly the supercell it maps to, so the subcells of all
    the cells at one resolution partition every finer one. The result is a
    covering: compacted, with a compaction marker recording the resolution --
    use `uncompact` to expand it.

    Resolution 30 covers only part of the world (see `lonlat_to_cell`); for a
    cell reaching past it, the subcells are given at resolution 29.

    Args:
        cell: The cell
        resolution: Target resolution, at least the cell's own

    Returns:
        Compacted cells sorted in curve order, then the compaction marker
    """
    cell_resolution = get_resolution(cell)
    if resolution < cell_resolution:
        raise ValueError(
            f'Target resolution ({resolution}) must be equal to or greater than current resolution ({cell_resolution})'
        )
    if resolution > MAX_RESOLUTION:
        raise ValueError(f'Target resolution ({resolution}) exceeds maximum resolution ({MAX_RESOLUTION})')
    if resolution == cell_resolution or cell == WORLD_CELL:
        return to_covering([cell], resolution)

    # Cells along a dodecahedron edge interlock with the neighboring face's, so a
    # cell's subcells can come from the faces next to its own: search each face
    # the cell's pentagon reaches into, in that face's frame (see seam_transform).
    a5cell = deserialize(cell)
    origin_id = a5cell['origin'].id
    own: List[float] = []
    for v in _get_pentagon(a5cell).get_vertices():
        own.append(v[0])
        own.append(v[1])
    frame_origins = [origin_id]
    frame_vertices = [own]
    face_edges = _face_edges()
    for q in range(5):
        adjacent_id = FACE_ADJACENCY[origin_id][q][0]
        m = seam_transform(origin_id, q)
        mapped = [0.0] * 10
        reaches = False
        for i in range(0, 10, 2):
            x = own[i]
            y = own[i + 1]
            mapped[i] = m[0] * x + m[2] * y + m[4]
            mapped[i + 1] = m[1] * x + m[3] * y + m[5]
            if _signed_margin(face_edges, mapped[i], mapped[i + 1]) > -_EDGE_EPS:
                reaches = True
        if reaches:
            frame_origins.append(adjacent_id)
            frame_vertices.append(mapped)

    # Res-30 IDs only reach the first RES30_QUINTANTS quintants (in ID order)
    if resolution == MAX_RESOLUTION and any(5 * o + 5 > RES30_QUINTANTS for o in frame_origins):
        resolution -= 1

    # Descend each face from its resolution-1 cells, classifying a cell by the
    # signed distance of its center from the pentagon's edges, in that face's frame
    lines_by_origin = [face_edges] * 12
    starts: List[int] = []
    for f in range(len(frame_origins)):
        lines_by_origin[frame_origins[f]] = _edge_lines(frame_vertices[f])
        # The resolution-1 cells: triple (0, 0, 0) of each quintant
        for q in range(5):
            starts.extend((frame_origins[f], q, 0, 0, 0))
    # By resolution: how far a cell's center must lie inside (or outside) for all
    # its descendants' to; none at the target, where each cell decides for itself
    target = resolution
    reaches_by_res = [_EDGE_EPS] * (target + 1)
    for res in range(1, target):
        reaches_by_res[res] = _DESCENDANT_REACH / 2 ** (res - FIRST_HILBERT_RESOLUTION + 1) + _EDGE_EPS

    def classify(oid: int, res: int, center, slot: int) -> int:
        margin = _signed_margin(lines_by_origin[oid], center[0], center[1])
        reach = reaches_by_res[res]
        if margin > reach:
            return INSIDE
        if margin < -reach:
            return OUTSIDE
        if res < target:
            return SPLIT
        # Within float noise of an edge: decide exactly as cell_to_supercell does
        return INSIDE if cell_to_supercell(slot_to_cell(slot, res), cell_resolution) == cell else OUTSIDE

    runs: SlotRuns = []
    descend_in_curve_order(starts, 0, target, classify, runs)
    return slot_runs_to_covering(runs, target)


# The edge lines of the face pentagon (filled on first use)
_FACE_EDGES: List[List[float]] = []


def _face_edges() -> List[float]:
    if not _FACE_EDGES:
        vertices: List[float] = []
        for v in get_face_vertices().get_vertices():
            vertices.append(v[0])
            vertices.append(v[1])
        _FACE_EDGES.append(_edge_lines(vertices))
    return _FACE_EDGES[0]


def _edge_lines(vertices: Sequence[float]) -> List[float]:
    """
    A convex pentagon ([x0, y0, ..., x4, y4]) as its edge lines: inward unit
    normal and offset, so a point's margin (`_signed_margin`) is its signed
    distance to the nearest edge line, positive inside.
    """
    cx = 0.0
    cy = 0.0
    for i in range(0, 10, 2):
        cx += vertices[i] / 5
        cy += vertices[i + 1] / 5
    lines = [0.0] * 15
    for i in range(5):
        x1 = vertices[2 * i]
        y1 = vertices[2 * i + 1]
        j = 0 if i == 4 else 2 * i + 2
        length = math.hypot(vertices[j] - x1, vertices[j + 1] - y1)
        nx = (y1 - vertices[j + 1]) / length
        ny = (vertices[j] - x1) / length
        if nx * (cx - x1) + ny * (cy - y1) < 0:
            nx = -nx
            ny = -ny
        lines[3 * i] = nx
        lines[3 * i + 1] = ny
        lines[3 * i + 2] = nx * x1 + ny * y1
    return lines


def _signed_margin(lines: List[float], x: float, y: float) -> float:
    margin = math.inf
    for i in range(0, 15, 3):
        d = lines[i] * x + lines[i + 1] * y - lines[i + 2]
        if d < margin:
            margin = d
    return margin

