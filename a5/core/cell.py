# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

import math
from typing import List, Tuple, Optional, Dict, TypedDict, Union
from .coordinate_systems import Cartesian, Face, LonLat, Spherical
from .coordinate_transforms import (
    face_to_ij, from_lonlat, to_cartesian, to_face, to_lonlat,
    to_spherical, to_polar, normalize_longitudes
)
from .origin import (
    find_nearest_origin,
    find_nearest_origin_cartesian,
    quintant_to_segment,
    segment_to_quintant,
)
from ..projections.dodecahedron import DodecahedronProjection
from .utils import A5Cell, Origin, OriginId
from ..geometry.pentagon import PentagonShape
from .tiling import get_face_vertices, get_pentagon_vertices, get_quintant_polar, get_quintant_vertices
from .constants import PI_OVER_5
from ..lattice import ij_to_s, s_to_anchor
from .serialization import deserialize, serialize, FIRST_HILBERT_RESOLUTION, WORLD_CELL
from ..geometry.spherical_polygon import SphericalPolygonShape
from ..utils.spiral import Spiral, SPIRAL_SAMPLE_COUNT

# Reuse this object to avoid allocation
_dodecahedron = DodecahedronProjection()

# Single-entry cache of the most recent successful lookup. Speeds up
# dense-sample workloads (polygon boundary tracing, line tracing) where
# consecutive calls often land in the same cell. The cache stores the
# pre-computed pentagon + origin so the hit-test is just one projection
# + one pentagon containment check.
_last_result: Optional[Dict] = None


def _cache_result(cell: A5Cell, cell_id: int, resolution: int) -> int:
    """Update the single-entry cache with a successful (cell, cell_id) pair."""
    global _last_result
    _last_result = {
        'cell_id': cell_id,
        'pentagon': _get_pentagon(cell),
        'origin_id': cell['origin'].id,
        'resolution': resolution,
    }
    return cell_id


class CellToBoundaryOptions(TypedDict, total=False):
    """Options for cell_to_boundary function."""
    closed_ring: bool
    segments: Union[int, str]

def lonlat_to_cell(lon_lat: LonLat, resolution: int) -> int:
    """
    Convert longitude/latitude coordinates to a cell ID.

    Args:
        lon_lat: Tuple of (longitude, latitude) in degrees
        resolution: Resolution level of the cell

    Returns:
        Cell ID as a big integer
    """
    return spherical_to_cell(from_lonlat(lon_lat), resolution)


def spherical_to_cell(spherical: Spherical, resolution: int) -> int:
    """
    Like `lonlat_to_cell`, but accepts a point already in A5's internal spherical
    representation (rotated authalic frame, as produced by `from_lonlat` or
    `to_spherical(authalic_cartesian)`). Skips the redundant authalic
    inverse/forward round-trip in dense-sample loops where the input already
    comes from the authalic Cartesian space (e.g. polygon-fill boundary slerp).
    """
    # Resolution -1 represents WORLD_CELL, which covers the entire world
    if resolution == -1:
        return WORLD_CELL

    if resolution < FIRST_HILBERT_RESOLUTION:
        # For low resolutions there is no Hilbert curve, so we can just return as the result is exact
        return serialize(_spherical_to_estimate(spherical, resolution))

    # Try the cached pentagon first -- skips the full estimate pipeline when
    # consecutive calls land in the same cell (common in dense-sample loops).
    if _last_result is not None and _last_result['resolution'] == resolution:
        projected = _dodecahedron.forward(spherical, _last_result['origin_id'])
        if _last_result['pentagon'].contains_point(projected) > 0:
            return _last_result['cell_id']

    # Try the original point's projection-based estimate. Common case for
    # non-boundary points.
    first_estimate = _spherical_to_estimate(spherical, resolution)
    first_key = serialize(first_estimate)
    first_distance = a5cell_contains_point(first_estimate, spherical)
    if first_distance > 0:
        return _cache_result(first_estimate, first_key, resolution)

    # Spiral search: perturb the point in the tangent plane to find nearby
    # estimate cells (see a5/utils/spiral.py).
    hilbert_resolution = 1 + resolution - FIRST_HILBERT_RESOLUTION
    scale = _SPIRAL_SCALE_RAD / (2 ** hilbert_resolution)
    estimate_set = {first_key}
    cells = [{'cell_id': first_key, 'distance': first_distance}]

    spiral = Spiral(spherical, scale)
    spiral_out: List[float] = [0.0, 0.0, 0.0]
    for i in range(SPIRAL_SAMPLE_COUNT):
        sample = spiral.sample(spiral_out, i)
        estimate = _cartesian_to_estimate(sample, resolution)
        estimate_key = serialize(estimate)
        if estimate_key in estimate_set:
            continue
        estimate_set.add(estimate_key)
        distance = a5cell_contains_point(estimate, spherical)
        if distance > 0:
            return _cache_result(estimate, estimate_key, resolution)
        cells.append({'cell_id': estimate_key, 'distance': distance})

    # Spiral exhausted without finding a strict container. This is reachable
    # for points right at the polar singularity at very high resolutions,
    # where re-projecting any tangent sample snaps back to a small set of
    # cells while the geometrically-containing cell is offset by one
    # adjacency step. Fall back to direct neighbours of the closest spiral
    # candidate, which always finds it.
    # Lazy import to avoid circular dependency with the traversal package.
    from ..traversal.global_neighbors import get_global_cell_neighbors
    cells.sort(key=lambda x: x['distance'], reverse=True)
    K = min(3, len(cells))
    for k in range(K):
        neighbors = get_global_cell_neighbors(cells[k]['cell_id'])
        for neighbor_key in neighbors:
            if neighbor_key in estimate_set:
                continue
            estimate_set.add(neighbor_key)
            neighbor_cell = deserialize(neighbor_key)
            distance = a5cell_contains_point(neighbor_cell, spherical)
            if distance > 0:
                return _cache_result(neighbor_cell, neighbor_key, resolution)
            cells.append({'cell_id': neighbor_key, 'distance': distance})

    # True fallback: closest cell wins, even if technically just outside.
    cells.sort(key=lambda x: x['distance'], reverse=True)
    fallback_key = cells[0]['cell_id']
    return _cache_result(deserialize(fallback_key), fallback_key, resolution)


# Spiral perturbation radius at hilbert_resolution=1 (in radians of tangent
# offset). For higher resolutions we scale by 1/2^hilbert_resolution. Tuned
# via debug-scripts/tune-spiral.ts.
_SPIRAL_SCALE_RAD = 70 * math.pi / 180


# The IJToS function uses the triangular lattice which only approximates the pentagon lattice
# Thus these functions only return a cell nearby, and we need to search the neighborhood to find the correct cell
# TODO: Implement a more accurate function

def _spherical_to_estimate(spherical: Spherical, resolution: int) -> A5Cell:
    origin = find_nearest_origin(spherical)
    dodec_point = _dodecahedron.forward(spherical, origin.id)
    return _face_to_estimate(dodec_point, origin, resolution)


def _cartesian_to_estimate(cartesian: Cartesian, resolution: int) -> A5Cell:
    origin = find_nearest_origin_cartesian(cartesian)
    dodec_point = _dodecahedron.forward_cartesian(cartesian, origin.id)
    return _face_to_estimate(dodec_point, origin, resolution)


def _face_to_estimate(dodec_point: Face, origin: Origin, resolution: int) -> A5Cell:
    polar = to_polar(dodec_point)
    quintant = get_quintant_polar(polar)
    segment, orientation = quintant_to_segment(quintant, origin)

    if resolution < FIRST_HILBERT_RESOLUTION:
        # For low resolutions there is no Hilbert curve
        return A5Cell(S=0, segment=segment, origin=origin, resolution=resolution)

    # Rotate into right fifth
    if quintant != 0:
        extra_angle = 2 * PI_OVER_5 * quintant
        c, s = math.cos(-extra_angle), math.sin(-extra_angle)
        # Manual 2x2 matrix multiplication
        new_x = c * dodec_point[0] - s * dodec_point[1]
        new_y = s * dodec_point[0] + c * dodec_point[1]
        dodec_point = (new_x, new_y)

    hilbert_resolution = 1 + resolution - FIRST_HILBERT_RESOLUTION
    scale_factor = 2 ** hilbert_resolution
    dodec_point = (dodec_point[0] * scale_factor, dodec_point[1] * scale_factor)

    ij = face_to_ij(dodec_point)
    S = ij_to_s(ij, hilbert_resolution, orientation)
    return A5Cell(S=S, segment=segment, origin=origin, resolution=resolution)

def _get_pentagon(cell: A5Cell) -> PentagonShape:
    """
    Get the pentagon shape for a given cell.
    
    Args:
        cell: A5Cell object
        
    Returns:
        PentagonShape object
    """
    quintant, orientation = segment_to_quintant(cell["segment"], cell["origin"])
    if cell["resolution"] == (FIRST_HILBERT_RESOLUTION - 1):
        out = get_quintant_vertices(quintant)
        return out
    elif cell["resolution"] == (FIRST_HILBERT_RESOLUTION - 2):
        return get_face_vertices()

    hilbert_resolution = cell["resolution"] - FIRST_HILBERT_RESOLUTION + 1
    anchor = s_to_anchor(cell["S"], hilbert_resolution, orientation)
    return get_pentagon_vertices(hilbert_resolution, quintant, anchor)

def cell_to_spherical(cell_id: int) -> Spherical:
    """
    Convert a cell ID to spherical coordinates.

    Args:
        cell_id: Cell ID as a big integer

    Returns:
        Spherical coordinates (theta, phi)
    """
    cell = deserialize(cell_id)
    pentagon = _get_pentagon(cell)
    return _dodecahedron.inverse(pentagon.get_center(), cell["origin"].id)


def cell_to_lonlat(cell_id: int) -> LonLat:
    """
    Convert a cell ID to longitude/latitude coordinates.

    Args:
        cell_id: Cell ID as a big integer

    Returns:
        Tuple of (longitude, latitude) in degrees
    """
    # WORLD_CELL represents the entire world, return (0, 0) as a reasonable default
    if cell_id == WORLD_CELL:
        return (0.0, 0.0)

    return to_lonlat(cell_to_spherical(cell_id))

def cell_to_boundary(
    cell_id: int,
    options: Optional[CellToBoundaryOptions] = None
) -> List[LonLat]:
    """
    Get the boundary coordinates of a cell.

    Args:
        cell_id: Cell ID as a big integer
        options: Dictionary with optional parameters:
            - closed_ring: Pass True to close the ring with the first point (default True)
            - segments: Number of segments to use for each edge. Pass 'auto' to use the resolution of the cell (default 'auto')

    Returns:
        List of (longitude, latitude) coordinates forming the cell boundary
    """
    # WORLD_CELL represents the entire world and is unbounded
    if cell_id == WORLD_CELL:
        return []

    if options is None:
        options = {}

    closed_ring = options.get('closed_ring', True)
    segments = options.get('segments', 'auto')

    cell = deserialize(cell_id)
    if segments == 'auto' or segments is None:
        segments = max(1, 2 ** (6 - cell["resolution"]))

    pentagon = _get_pentagon(cell)

    # Split each edge into segments before projection
    # Important to do before projection to obtain equal area cells
    split_pentagon = pentagon.split_edges(segments)
    vertices = split_pentagon.get_vertices()

    # Unproject to obtain lon/lat coordinates. Fused loop avoids the
    # intermediate unprojected_vertices allocation.
    boundary: List[LonLat] = [None] * len(vertices)
    for i in range(len(vertices)):
        boundary[i] = to_lonlat(_dodecahedron.inverse(vertices[i], cell["origin"].id))

    # Normalize longitudes to handle antimeridian crossing
    normalized_boundary = normalize_longitudes(boundary)

    if closed_ring:
        normalized_boundary.append(normalized_boundary[0])
    
    # TODO: This is a patch to make the boundary CCW, but we should fix the winding order of the pentagon
    # throughout the whole codebase
    normalized_boundary.reverse()
    return normalized_boundary

def a5cell_contains_point(cell: A5Cell, spherical: Spherical) -> float:
    """
    Check if a spherical point is contained within a cell.

    Args:
        cell: A5Cell object
        spherical: Spherical coordinates in A5's internal rotated authalic frame

    Returns:
        Positive number if the point is contained within the cell, negative otherwise
    """
    pentagon = _get_pentagon(cell)
    projected_point = _dodecahedron.forward(spherical, cell['origin'].id)

    return pentagon.contains_point(projected_point)


def cell_intersects_segment(cell_id: int, a: LonLat, b: LonLat) -> bool:
    """
    Tests whether the segment between two LonLat points intersects a cell.

    The test runs entirely in the cell's Face coordinate system: both endpoints
    are projected via the dodecahedron projection (with face-plane extension for
    points beyond the face's edge), then checked against the pentagon's straight
    2D edges. The segment is treated as a 2D straight line in Face coords --
    accurate when the segment is short relative to the face (DSEA distortion is
    negligible at sub-cell scales).
    """
    if cell_id == WORLD_CELL:
        return True
    cell = deserialize(cell_id)
    pentagon = _get_pentagon(cell)
    a_face = _dodecahedron.forward(from_lonlat(a), cell['origin'].id)
    b_face = _dodecahedron.forward(from_lonlat(b), cell['origin'].id)
    return pentagon.intersects_segment(a_face, b_face)
