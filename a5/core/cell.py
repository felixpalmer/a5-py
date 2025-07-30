# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

import math
from typing import List, Tuple, Optional, Dict, TypedDict, Union
from .coordinate_systems import Face, LonLat, Spherical
from .coordinate_transforms import (
    face_to_ij, from_lonlat, to_cartesian, to_face, to_lonlat, 
    to_spherical, to_polar, normalize_longitudes
)
from .origin import find_nearest_origin, quintant_to_segment, segment_to_quintant
from ..projections.dodecahedron import DodecahedronProjection
from .utils import A5Cell
from ..geometry.pentagon import PentagonShape
from .tiling import get_face_vertices, get_pentagon_vertices, get_quintant_polar, get_quintant_vertices
from .constants import PI_OVER_5
from .hilbert import ij_to_s, s_to_anchor
from .serialization import deserialize, serialize, FIRST_HILBERT_RESOLUTION
from ..geometry.spherical_polygon import SphericalPolygonShape

# Reuse this object to avoid allocation
_dodecahedron = DodecahedronProjection()

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
    if resolution < FIRST_HILBERT_RESOLUTION:
        # For low resolutions there is no Hilbert curve, so we can just return as the result is exact
        return serialize(_lonlat_to_estimate(lon_lat, resolution))

    hilbert_resolution = 1 + resolution - FIRST_HILBERT_RESOLUTION
    samples: List[LonLat] = [lon_lat]
    N = 25
    scale = 50 / (2 ** hilbert_resolution)
    
    for i in range(N):
        R = (i / N) * scale
        coordinate = (
            math.cos(i) * R + lon_lat[0],
            math.sin(i) * R + lon_lat[1]
        )
        samples.append(coordinate)

    # Deduplicate estimates
    estimate_set = set()
    unique_estimates = []

    cells = []
    for sample in samples:
        estimate = _lonlat_to_estimate(sample, resolution)
        estimate_key = serialize(estimate)
        if estimate_key not in estimate_set:
            # Have new estimate, add to set and list
            estimate_set.add(estimate_key)
            unique_estimates.append(estimate)

            # Check if we have a hit, storing distance if not
            distance = a5cell_contains_point(estimate, lon_lat)
            if distance > 0:
                return serialize(estimate)
            else:
                cells.append({'cell': estimate, 'distance': distance})

    # As fallback, sort cells by distance and use the closest one
    cells.sort(key=lambda x: x['distance'], reverse=True)
    return serialize(cells[0]['cell'])

def _lonlat_to_estimate(lon_lat: LonLat, resolution: int) -> A5Cell:
    """
    Convert longitude/latitude to an approximate cell.
    The IJToS function uses the triangular lattice which only approximates the pentagon lattice.
    Thus this function only returns a cell nearby, and we need to search the neighbourhood to find the correct cell.
    
    Args:
        lon_lat: Tuple of (longitude, latitude) in degrees
        resolution: Resolution level of the cell
        
    Returns:
        Approximate A5Cell
    """
    spherical = from_lonlat(lon_lat)
    origin = find_nearest_origin(spherical)

    dodec_point = _dodecahedron.forward(spherical, origin.id)
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
    estimate = A5Cell(S=S, segment=segment, origin=origin, resolution=resolution)
    return estimate

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

def cell_to_lonlat(cell_id: int) -> LonLat:
    """
    Convert a cell ID to longitude/latitude coordinates.
    
    Args:
        cell_id: Cell ID as a big integer
        
    Returns:
        Tuple of (longitude, latitude) in degrees
    """
    cell = deserialize(cell_id)
    pentagon = _get_pentagon(cell)
    point = _dodecahedron.inverse(pentagon.get_center(), cell["origin"].id)
    return to_lonlat(point)

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

    # Unproject to obtain lon/lat coordinates
    unprojected_vertices = [_dodecahedron.inverse(vertex, cell["origin"].id) for vertex in vertices]
    boundary = [to_lonlat(vertex) for vertex in unprojected_vertices]

    # Normalize longitudes to handle antimeridian crossing
    normalized_boundary = normalize_longitudes(boundary)

    if closed_ring:
        normalized_boundary.append(normalized_boundary[0])
    
    # TODO: This is a patch to make the boundary CCW, but we should fix the winding order of the pentagon
    # throughout the whole codebase
    normalized_boundary.reverse()
    return normalized_boundary

def a5cell_contains_point(cell: A5Cell, point: LonLat) -> float:
    """
    Check if a point is contained within a cell.
    
    Args:
        cell: A5Cell object
        point: Tuple of (longitude, latitude) in degrees
        
    Returns:
        Positive number if the point is contained within the cell, negative otherwise
    """
    pentagon = _get_pentagon(cell)
    
    spherical = from_lonlat(point)
    projected_point = _dodecahedron.forward(spherical, cell['origin'].id)
    
    return pentagon.contains_point(projected_point) 