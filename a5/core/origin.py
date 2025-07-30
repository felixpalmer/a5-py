"""
A5 - Global Pentagonal Geospatial Index
SPDX-License-Identifier: Apache-2.0
Copyright (c) A5 contributors
"""

import math
from typing import List, Tuple, NamedTuple
from .coordinate_transforms import to_cartesian
from .coordinate_systems import Radians, Spherical, Face
from .constants import interhedral_angle, PI_OVER_5, TWO_PI_OVER_5, distance_to_edge
from .hilbert import Orientation
from ..math import quat
from .utils import Origin
from .dodecahedron_quaternions import quaternions

UP = (0, 0, 1)
origins: List[Origin] = []

# Quintant layouts (clockwise & counterclockwise)
clockwise_fan = ['vu', 'uw', 'vw', 'vw', 'vw']
clockwise_step = ['wu', 'uw', 'vw', 'vu', 'uw']
counter_step = ['wu', 'uv', 'wv', 'wu', 'uw']
counter_jump = ['vu', 'uv', 'wv', 'wu', 'uw']

QUINTANT_ORIENTATIONS = [
    clockwise_fan,   # 0 Arctic
    counter_jump,    # 1 North America
    counter_step,    # 2 South America
    clockwise_step,  # 3 North Atlantic & Western Europe & Africa
    counter_step,    # 4 South Atlantic & Africa
    counter_jump,    # 5 Europe, Middle East & CentralAfrica
    counter_step,    # 6 Indian Ocean
    clockwise_step,  # 7 Asia
    clockwise_step,  # 8 Australia
    clockwise_step,  # 9 North Pacific
    counter_jump,    # 10 South Pacific
    counter_jump,    # 11 Antarctic
]

# Within each face, these are the indices of the first quintant
QUINTANT_FIRST = [4, 2, 3, 2, 0, 4, 3, 2, 2, 0, 3, 0]

# Placements of dodecahedron faces along the Hilbert curve
ORIGIN_ORDER = [0, 1, 2, 4, 3, 5, 7, 8, 6, 11, 10, 9]


def generate_origins() -> None:
    """Generate all origin points for the dodecahedron faces."""
    # North pole
    add_origin((0, 0), 0, quaternions[0])

    # Middle band
    for i in range(5):
        alpha = i * TWO_PI_OVER_5
        alpha2 = alpha + PI_OVER_5
        add_origin((alpha, interhedral_angle), PI_OVER_5, quaternions[i + 1])
        add_origin((alpha2, math.pi - interhedral_angle), PI_OVER_5, quaternions[(i + 3) % 5 + 6])

    # South pole
    add_origin((0, math.pi), 0, quaternions[11])

def add_origin(axis: Spherical, angle: Radians, quaternion: Tuple[float, float, float, float]) -> None:
    """Add a new origin point."""
    global origin_id
    if origin_id > 11:
        raise ValueError(f"Too many origins: {origin_id}")
    
    inverse_quat = quat.create()
    quat.conjugate(inverse_quat, quaternion)
    origin = Origin(
        id=origin_id,
        axis=axis,
        quat=quaternion,
        inverse_quat=inverse_quat,
        angle=angle,
        orientation=QUINTANT_ORIENTATIONS[origin_id],
        first_quintant=QUINTANT_FIRST[origin_id]
    )
    origins.append(origin)
    origin_id += 1

origin_id = 0
generate_origins()

# Reorder origins to match the order of the hilbert curve
origins.sort(key=lambda x: ORIGIN_ORDER.index(x.id))
for i, origin in enumerate(origins):
    origins[i] = Origin(
        id=i,
        axis=origin.axis,
        quat=origin.quat,
        inverse_quat=origin.inverse_quat,
        angle=origin.angle,
        orientation=origin.orientation,
        first_quintant=origin.first_quintant
    )

def quintant_to_segment(quintant: int, origin: Origin) -> Tuple[int, Orientation]:
    """Convert a quintant to a segment number and orientation."""
    # Lookup winding direction of this face
    layout = origin.orientation
    step = -1 if layout in (clockwise_fan, clockwise_step) else 1

    # Find (CCW) delta from first quintant of this face
    delta = (quintant - origin.first_quintant + 5) % 5

    # To look up the orientation, we need to use clockwise/counterclockwise counting
    face_relative_quintant = (step * delta + 5) % 5
    orientation = layout[face_relative_quintant]
    segment = (origin.first_quintant + face_relative_quintant) % 5

    return segment, orientation

def segment_to_quintant(segment: int, origin: Origin) -> Tuple[int, Orientation]:
    """Convert a segment number to a quintant and orientation."""
    # Lookup winding direction of this face
    layout = origin.orientation
    step = -1 if layout in (clockwise_fan, clockwise_step) else 1

    face_relative_quintant = (segment - origin.first_quintant + 5) % 5
    orientation = layout[face_relative_quintant]
    quintant = (origin.first_quintant + step * face_relative_quintant + 5) % 5

    return quintant, orientation

def find_nearest_origin(point: Spherical) -> Origin:
    """
    Find the nearest origin to a point on the sphere.
    Uses haversine formula to calculate great-circle distance.
    """
    min_distance = float('inf')
    nearest = origins[0]
    for origin in origins:
        distance = haversine(point, origin.axis)
        if distance < min_distance:
            min_distance = distance
            nearest = origin
    return nearest

def is_nearest_origin(point: Spherical, origin: Origin) -> bool:
    """Check if the given origin is the nearest to the point."""
    return haversine(point, origin.axis) > 0.49999999

def haversine(point: Spherical, axis: Spherical) -> float:
    """
    Modified haversine formula to calculate great-circle distance.
    Returns the "angle" between the two points.
    
    Args:
        point: The point to calculate distance from
        axis: The axis to calculate distance to
        
    Returns:
        The "angle" between the two points
    """
    theta, phi = point
    theta2, phi2 = axis
    dtheta = theta2 - theta
    dphi = phi2 - phi
    a1 = math.sin(dphi / 2)
    a2 = math.sin(dtheta / 2)
    angle = a1 * a1 + a2 * a2 * math.sin(phi) * math.sin(phi2)
    return angle