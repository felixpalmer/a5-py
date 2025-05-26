"""
A5 - Global Pentagonal Geospatial Index
SPDX-License-Identifier: Apache-2.0
Copyright (c) A5 contributors
"""

import numpy as np
from typing import List, Tuple, NamedTuple
from .coordinate_transforms import to_cartesian, quat_from_spherical
from .coordinate_systems import Radians, Spherical, Face
from .constants import interhedral_angle, PI_OVER_5, TWO_PI_OVER_5, distance_to_edge
from .hilbert import Orientation
from .quat import conjugate, transform_quat, rotation_to
from .utils import Origin, origins

UP = np.array([0, 0, 1], dtype=np.float64)


class FaceTransform(NamedTuple):
    point: Face
    quat: np.ndarray

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
    add_origin((0, 0), 0)

    # Middle band
    for i in range(5):
        alpha = i * TWO_PI_OVER_5
        alpha2 = alpha + PI_OVER_5
        add_origin((alpha, interhedral_angle), PI_OVER_5)
        add_origin((alpha2, np.pi - interhedral_angle), PI_OVER_5)

    # South pole
    add_origin((0, np.pi), 0)

def add_origin(axis: Spherical, angle: Radians) -> None:
    """Add a new origin point."""
    global origin_id
    origin = Origin(
        id=origin_id,
        axis=axis,
        quat=quat_from_spherical(axis),
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

def move_point_to_face(point: Face, from_origin: Origin, to_origin: Origin) -> FaceTransform:
    """
    Move a point defined in the coordinate system of one dodecahedron face to the coordinate system of another face.
    
    Args:
        point: The point to move
        from_origin: The origin of the current face
        to_origin: The origin of the target face
        
    Returns:
        FaceTransform containing the new point and the quaternion representing the transform
    """
    # Get inverse quaternion
    from_quat = from_origin.quat.flatten()
    inverse_quat = conjugate(from_quat)

    to_axis = to_cartesian(to_origin.axis)

    # Transform destination axis into face space
    local_to_axis = transform_quat(to_axis, inverse_quat)

    # Flatten axis to XY plane to obtain direction, scale to get distance to new origin
    direction = np.array([local_to_axis[0], local_to_axis[1]], dtype=np.float64)
    direction = direction / np.linalg.norm(direction)
    direction *= 2 * distance_to_edge

    # Move point to be relative to new origin
    offset_point = point - direction

    # Construct relative transform from old origin to new origin
    interface_quat = rotation_to(UP, local_to_axis)
    interface_quat = np.dot(from_origin.quat.flatten(), interface_quat)

    return FaceTransform(point=offset_point, quat=interface_quat)

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
    return haversine(point, origin.axis) > 0.49

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
    a1 = np.sin(dphi / 2)
    a2 = np.sin(dtheta / 2)
    angle = a1 * a1 + a2 * a2 * np.sin(phi) * np.sin(phi2)
    return angle 