"""
A5
SPDX-License-Identifier: Apache-2.0
Copyright (c) A5 contributors
"""

import math
import numpy as np
from typing import cast
from .types import (
    Degrees, Radians, Face, Polar, IJ, Cartesian, Spherical, LonLat,
    Vec2, Vec3
)

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
    from .pentagon import BASIS_INVERSE
    ij_array = np.dot(BASIS_INVERSE, face)
    return cast(IJ, ij_array)

def ij_to_face(ij: IJ) -> Face:
    """Convert IJ coordinates to face coordinates."""
    # Note: BASIS needs to be defined in pentagon.py
    from .pentagon import BASIS
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

def from_lon_lat(lon_lat: LonLat) -> Spherical:
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
    phi = deg_to_rad(cast(Degrees, 90 - latitude))
    return cast(Spherical, (theta, phi))

def to_lon_lat(spherical: Spherical) -> LonLat:
    """Convert spherical coordinates to longitude/latitude.
    
    Args:
        spherical: Tuple of (theta, phi) in radians
            theta: 0 to 2π
            phi: 0 to π
    
    Returns:
        Tuple of (longitude, latitude) in degrees
    """
    theta, phi = spherical
    longitude = cast(Degrees, rad_to_deg(theta) - LONGITUDE_OFFSET)
    latitude = cast(Degrees, 90 - rad_to_deg(phi))
    return cast(LonLat, (longitude, latitude))

def quat_from_spherical(axis: Spherical) -> Vec3:
    """Create a quaternion rotation from spherical coordinates."""
    cartesian = to_cartesian(axis)
    # Create rotation quaternion from [0,0,1] to cartesian
    v1 = np.array([0, 0, 1], dtype=np.float64)
    v2 = cartesian
    
    # Normalize vectors
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    
    # Calculate rotation axis and angle
    cross = np.cross(v1, v2)
    dot = np.dot(v1, v2)
    
    # Handle special cases
    if np.allclose(cross, 0):
        if dot > 0:
            return np.array([0, 0, 0, 1], dtype=np.float64)  # Identity quaternion
        else:
            return np.array([1, 0, 0, 0], dtype=np.float64)  # 180-degree rotation around x-axis
    
    # Normalize cross product
    cross = cross / np.linalg.norm(cross)
    
    # Calculate quaternion components
    angle = math.acos(dot)
    s = math.sin(angle / 2)
    return np.array([cross[0] * s, cross[1] * s, cross[2] * s, math.cos(angle / 2)], dtype=np.float64) 