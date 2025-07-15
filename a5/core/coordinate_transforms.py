"""
A5
SPDX-License-Identifier: Apache-2.0
Copyright (c) A5 contributors
"""

import math
import numpy as np
from typing import cast, List, Tuple
from .coordinate_systems import (
    Degrees, Radians, Face, Polar, IJ, Cartesian, Spherical, LonLat,
    Vec2, Vec3, Barycentric, FaceTriangle
)
from .quat import rotation_to
from .pentagon import BASIS_INVERSE, BASIS
from ..projections.authalic import AuthalicProjection

# Create singleton instance like TypeScript
authalic = AuthalicProjection()

# Constants
LONGITUDE_OFFSET = cast(Degrees, 93.0)  # degrees

def deg_to_rad(deg: Degrees) -> Radians:
    """Convert degrees to radians."""
    return cast(Radians, deg * (math.pi / 180))

def rad_to_deg(rad: Radians) -> Degrees:
    """Convert radians to degrees."""
    return cast(Degrees, rad * (180 / math.pi))

def to_polar(xy: Face) -> Polar:
    """Convert face coordinates to polar coordinates."""
    rho = math.sqrt(xy[0]**2 + xy[1]**2)  # Radial distance from face center
    gamma = cast(Radians, math.atan2(xy[1], xy[0]))  # Azimuthal angle
    return cast(Polar, (rho, gamma))

def to_face(polar: Polar) -> Face:
    """Convert polar coordinates to face coordinates."""
    rho, gamma = polar
    x = rho * math.cos(gamma)
    y = rho * math.sin(gamma)
    return cast(Face, np.array([x, y], dtype=np.float64))

def face_to_ij(face: Face) -> IJ:
    """Convert face coordinates to IJ coordinates."""
    # Note: BASIS_INVERSE needs to be defined in pentagon.py
    ij_array = np.dot(BASIS_INVERSE, face)
    return cast(IJ, ij_array)

def ij_to_face(ij: IJ) -> Face:
    """Convert IJ coordinates to face coordinates."""
    # Note: BASIS needs to be defined in pentagon.py
    face_array = np.dot(BASIS, ij)
    return cast(Face, face_array)

def to_spherical(xyz: Cartesian) -> Spherical:
    """Convert Cartesian coordinates to spherical coordinates."""
    theta = cast(Radians, math.atan2(xyz[1], xyz[0]))
    r = math.sqrt(xyz[0]**2 + xyz[1]**2 + xyz[2]**2)
    phi = cast(Radians, math.acos(xyz[2] / r))
    return cast(Spherical, (theta, phi))

def to_cartesian(spherical: Spherical) -> Cartesian:
    """Convert spherical coordinates to Cartesian coordinates."""
    theta, phi = spherical
    x = math.sin(phi) * math.cos(theta)
    y = math.sin(phi) * math.sin(theta)
    z = math.cos(phi)
    return cast(Cartesian, np.array([x, y, z], dtype=np.float64))

def from_lonlat(lon_lat: LonLat) -> Spherical:
    """Convert longitude/latitude to spherical coordinates.
    
    Args:
        lon_lat: Tuple of (longitude, latitude) in degrees
            longitude: 0 to 360
            latitude: -90 to 90
    
    Returns:
        Tuple of (theta, phi) in radians
    """
    longitude, latitude = lon_lat
    theta = deg_to_rad(cast(Degrees, longitude + LONGITUDE_OFFSET))
    
    geodetic_lat = deg_to_rad(cast(Degrees, latitude))
    authalic_lat = authalic.forward(geodetic_lat)
    phi = cast(Radians, math.pi / 2 - authalic_lat)
    return cast(Spherical, (theta, phi))

def to_lonlat(spherical: Spherical) -> LonLat:
    """Convert spherical coordinates to longitude/latitude.
    
    Args:
        spherical: Tuple of (theta, phi) in radians
            theta: 0 to 2π
            phi: 0 to π
    
    Returns:
        Tuple of (longitude, latitude) in degrees
            longitude: 0 to 360
            latitude: -90 to 90
    """
    theta, phi = spherical
    longitude = rad_to_deg(theta) - LONGITUDE_OFFSET

    authalic_lat = cast(Radians, math.pi / 2 - phi)
    geodetic_lat = authalic.inverse(authalic_lat)
    latitude = rad_to_deg(geodetic_lat)
    return cast(LonLat, (longitude, latitude))

def face_to_barycentric(p: Face, triangle: FaceTriangle) -> Barycentric:
    """Convert face coordinates to barycentric coordinates."""
    p1, p2, p3 = triangle
    d31 = [p1[0] - p3[0], p1[1] - p3[1]]
    d23 = [p3[0] - p2[0], p3[1] - p2[1]]
    d3p = [p[0] - p3[0], p[1] - p3[1]]
    
    det = d23[0] * d31[1] - d23[1] * d31[0]
    b0 = (d23[0] * d3p[1] - d23[1] * d3p[0]) / det
    b1 = (d31[0] * d3p[1] - d31[1] * d3p[0]) / det
    b2 = 1 - (b0 + b1)
    return cast(Barycentric, (b0, b1, b2))

def barycentric_to_face(b: Barycentric, triangle: FaceTriangle) -> Face:
    """Convert barycentric coordinates to face coordinates."""
    p1, p2, p3 = triangle
    return cast(Face, np.array([
        b[0] * p1[0] + b[1] * p2[0] + b[2] * p3[0],
        b[0] * p1[1] + b[1] * p2[1] + b[2] * p3[1]
    ], dtype=np.float64))

Contour = List[LonLat]

def normalize_longitudes(contour: Contour) -> Contour:
    """Normalizes longitude values in a contour to handle antimeridian crossing.
    
    Args:
        contour: Array of [longitude, latitude] points
        
    Returns:
        Normalized contour with consistent longitude values
    """
    # Calculate center in Cartesian space to avoid poles & antimeridian crossing issues
    points = [to_cartesian(from_lonlat(lonlat)) for lonlat in contour]
    center = np.zeros(3, dtype=np.float64)
    for point in points:
        center += point
    center /= np.linalg.norm(center)
    center_lon, center_lat = to_lonlat(to_spherical(cast(Cartesian, center)))
    
    if center_lat > 89.99 or center_lat < -89.99:
        # Near poles, use first point's longitude
        center_lon = contour[0][0]

    # Normalize center longitude to be in the range -180 to 180
    center_lon = ((center_lon + 180) % 360 + 360) % 360 - 180

    # Normalize each point relative to center
    result = []
    for point in contour:
        longitude, latitude = point
        
        # Adjust longitude to be closer to center
        while longitude - center_lon > 180:
            longitude = cast(Degrees, longitude - 360)
        while longitude - center_lon < -180:
            longitude = cast(Degrees, longitude + 360)
        result.append(cast(LonLat, (longitude, latitude)))
    
    return result 

def quat_from_spherical(axis: Spherical) -> np.ndarray:
    """
    Creates a quaternion representing a rotation from the north pole to a given axis.
    
    Args:
        axis: Spherical coordinate of axis to rotate to
        
    Returns:
        quaternion [x, y, z, w]
    """
    cartesian = to_cartesian(axis)
    return rotation_to(np.array([0, 0, 1], dtype=np.float64), cartesian) 