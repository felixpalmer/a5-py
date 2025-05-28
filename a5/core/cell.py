import numpy as np
from typing import List, Tuple, Optional
from .coordinate_systems import Face, LonLat, Spherical
from .coordinate_transforms import face_to_ij, from_lonlat, to_face
from .origin import find_nearest_origin, quintant_to_segment, segment_to_quintant
from .dodecahedron import unproject_dodecahedron
from .utils import A5Cell, PentagonShape
from .tiling import get_face_vertices, get_pentagon_vertices, get_quintant, get_quintant_vertices
from .constants import PI_OVER_5
from .hilbert import ij_to_s, s_to_anchor
from .project import project_pentagon, project_point
from .serialization import deserialize, serialize, FIRST_HILBERT_RESOLUTION
from .utils import Origin
# Reuse rotation matrix to avoid allocation
_rotation = None

def lonlat_to_cell(lon_lat: LonLat, resolution: int) -> int:
    """
    Convert longitude/latitude coordinates to a cell ID.
    
    Args:
        lon_lat: Tuple of (longitude, latitude) in degrees
        resolution: Resolution level of the cell
        
    Returns:
        Cell ID as a big integer
    """
    hilbert_resolution = 1 + resolution - FIRST_HILBERT_RESOLUTION
    samples: List[LonLat] = [lon_lat]
    N = 25
    scale = 50 / (2 ** hilbert_resolution)
    
    for i in range(N):
        R = (i / N) * scale
        coordinate = np.array([
            np.cos(i) * R + lon_lat[0],
            np.sin(i) * R + lon_lat[1]
        ])
        samples.append(tuple(coordinate))

    cells: List[A5Cell] = []
    for sample in samples:
        estimate = _lonlat_to_estimate(sample, resolution)

        # For resolution 0 there is no Hilbert curve, so we can just return as the result is exact
        if resolution < FIRST_HILBERT_RESOLUTION or a5cell_contains_point(estimate, lon_lat):
            return serialize(estimate)
        else:
            cells.append(estimate)

    # Failed to find based on hit test, just return the closest cell
    D = float('inf')
    best_cell: Optional[A5Cell] = None
    for cell in cells:
        pentagon = _get_pentagon(cell)
        center = project_point(pentagon.get_center(), cell["origin"])
        distance = np.linalg.norm(np.array(center) - np.array(lon_lat))
        if distance < D:
            D = distance
            best_cell = cell

    if best_cell:
        return serialize(best_cell)
    raise ValueError('No cell found')

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
    global _rotation
    if _rotation is None:
        _rotation = np.zeros((2, 2))
        
    spherical = from_lonlat(lon_lat)
    origin = find_nearest_origin(spherical)
    
    # Create rotation matrix
    angle = -origin.angle
    c, s = np.cos(angle), np.sin(angle)
    _rotation[0, 0], _rotation[0, 1] = c, -s
    _rotation[1, 0], _rotation[1, 1] = s, c

    polar = unproject_dodecahedron(spherical, origin.quat, origin.angle)
    dodec_point = to_face(polar)
    quintant = get_quintant(dodec_point)
    segment, orientation = quintant_to_segment(quintant, origin)
    
    if resolution < FIRST_HILBERT_RESOLUTION:
        # For low resolutions there is no Hilbert curve
        return A5Cell(S=0, segment=segment, origin=origin, resolution=resolution)

    # Rotate into right fifth
    if quintant != 0:
        extra_angle = 2 * PI_OVER_5 * quintant
        c, s = np.cos(-extra_angle), np.sin(-extra_angle)
        _rotation[0, 0], _rotation[0, 1] = c, -s
        _rotation[1, 0], _rotation[1, 1] = s, c
        dodec_point = np.dot(_rotation, dodec_point)

    hilbert_resolution = 1 + resolution - FIRST_HILBERT_RESOLUTION
    dodec_point *= (2 ** hilbert_resolution)

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
        return get_quintant_vertices(quintant)
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
    # Ensure quaternion is properly handled
    origin = Origin(
        id=cell["origin"].id,
        axis=cell["origin"].axis,
        quat=cell["origin"].quat,
        angle=cell["origin"].angle,
        orientation=cell["origin"].orientation,
        first_quintant=cell["origin"].first_quintant
    )
    lon_lat = project_point(pentagon.get_center(), origin)
    return PentagonShape.normalize_longitudes([lon_lat])[0]

def cell_to_boundary(cell_id: int) -> List[LonLat]:
    """
    Get the boundary coordinates of a cell.
    
    Args:
        cell_id: Cell ID as a big integer
        
    Returns:
        List of (longitude, latitude) coordinates forming the cell boundary
    """
    cell = deserialize(cell_id)
    pentagon = _get_pentagon(cell)
    # Ensure quaternion is properly handled
    origin = Origin(
        id=cell["origin"].id,
        axis=cell["origin"].axis,
        quat=cell["origin"].quat,
        angle=cell["origin"].angle,
        orientation=cell["origin"].orientation,
        first_quintant=cell["origin"].first_quintant
    )
    return project_pentagon(pentagon, origin)

def a5cell_contains_point(cell: A5Cell, point: LonLat) -> bool:
    """
    Check if a point is contained within a cell.
    
    Args:
        cell: A5Cell object
        point: Tuple of (longitude, latitude) in degrees
        
    Returns:
        True if the point is contained within the cell, False otherwise
    """
    pentagon = _get_pentagon(cell)
    projected_pentagon_vertices = project_pentagon(pentagon, cell["origin"])
    proj_pentagon = PentagonShape(projected_pentagon_vertices)
    return proj_pentagon.contains_point(point) 