"""
A5
SPDX-License-Identifier: Apache-2.0
Copyright (c) A5 contributors
"""

import math
from typing import cast, List
from .coordinate_systems import (
    Degrees, Radians, Face, Polar, IJ, Cartesian, Spherical, LonLat,
    Barycentric, FaceTriangle
)
from .pentagon import BASIS_INVERSE, BASIS
from ..projections.authalic import AuthalicProjection
from ..math import vec2, vec3

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
    rho = vec2.length(xy)  # Radial distance from face center
    gamma = cast(Radians, math.atan2(xy[1], xy[0]))  # Azimuthal angle
    return cast(Polar, (rho, gamma))

def to_face(polar: Polar) -> Face:
    """Convert polar coordinates to face coordinates."""
    rho, gamma = polar
    x = rho * math.cos(gamma)
    y = rho * math.sin(gamma)
    return cast(Face, (x, y))

def face_to_ij(face: Face) -> IJ:
    """Convert face coordinates to IJ coordinates."""
    # Use gl-matrix style transformation
    # Convert 2x2 matrix from ((a,b),(c,d)) to [a,c,b,d] (column-major)
    basis_flat = [BASIS_INVERSE[0][0], BASIS_INVERSE[1][0], BASIS_INVERSE[0][1], BASIS_INVERSE[1][1]]
    out = vec2.create()
    vec2.transformMat2(out, face, basis_flat)
    return cast(IJ, (out[0], out[1]))

def ij_to_face(ij: IJ) -> Face:
    """Convert IJ coordinates to face coordinates."""
    # Use gl-matrix style transformation
    # Convert 2x2 matrix from ((a,b),(c,d)) to [a,c,b,d] (column-major)
    basis_flat = [BASIS[0][0], BASIS[1][0], BASIS[0][1], BASIS[1][1]]
    out = vec2.create()
    vec2.transformMat2(out, ij, basis_flat)
    return cast(Face, (out[0], out[1]))

def to_spherical(xyz: Cartesian) -> Spherical:
    """Convert Cartesian coordinates to spherical coordinates."""
    theta = cast(Radians, math.atan2(xyz[1], xyz[0]))
    r = vec3.length(xyz)
    phi = cast(Radians, math.acos(xyz[2] / r))
    return cast(Spherical, (theta, phi))

def to_cartesian(spherical: Spherical) -> Cartesian:
    """Convert spherical coordinates to Cartesian coordinates."""
    theta, phi = spherical
    x = math.sin(phi) * math.cos(theta)
    y = math.sin(phi) * math.sin(theta)
    z = math.cos(phi)
    return cast(Cartesian, (x, y, z))

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
    return cast(Face, (
        b[0] * p1[0] + b[1] * p2[0] + b[2] * p3[0],
        b[0] * p1[1] + b[1] * p2[1] + b[2] * p3[1]
    ))

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
    center = vec3.create()
    for point in points:
        vec3.add(center, center, point)
    
    # Normalize the center
    vec3.normalize(center, center)
    center_lon, center_lat = to_lonlat(to_spherical(cast(Cartesian, (center[0], center[1], center[2]))))
    
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