"""
Tests for origin-related functionality.
"""

import pytest
import numpy as np
from a5.core.origin import (    
    find_nearest_origin,
    haversine,
    quintant_to_segment,
    segment_to_quintant,
    move_point_to_face,
)
from a5.core.constants import distance_to_edge, PI_OVER_5, TWO_PI_OVER_5
from a5.core.coordinate_systems import Face, Radians, Spherical
from a5.core.coordinate_transforms import to_cartesian
from a5.core.origin import origins

def test_origin_constants():
    """Test that we have 12 origins for dodecahedron faces."""
    assert len(origins) == 12


def test_origin_properties():
    """Test that each origin has required properties."""
    for origin in origins:
        # Check properties exist
        assert origin.axis is not None
        assert origin.quat is not None
        assert origin.angle is not None
        
        # Check axis is unit vector when converted to cartesian
        cartesian = to_cartesian(origin.axis)
        length = np.linalg.norm(cartesian)
        assert np.isclose(length, 1.0)
        
        # Check quaternion is normalized
        q_length = np.linalg.norm(origin.quat)
        assert np.isclose(q_length, 1.0)


def test_find_nearest_origin():
    """Test finding nearest origin for various points."""
    # Test points at face centers
    for origin in origins:
        point = origin.axis
        nearest = find_nearest_origin(point)
        assert nearest == origin

    # Test points at face boundaries
    boundary_points = [
        # Between north pole and equatorial faces
        {"point": [0, PI_OVER_5/2], "expected_origins": [0, 1]},
        # Between equatorial faces
        {"point": [2*PI_OVER_5, PI_OVER_5], "expected_origins": [3, 4]},
        # Between equatorial and south pole
        {"point": [0, np.pi - PI_OVER_5/2], "expected_origins": [9, 10]},
    ]

    for test_case in boundary_points:
        nearest = find_nearest_origin(test_case["point"])
        assert nearest.id in test_case["expected_origins"]

    # Test antipodal points
    for origin in origins:
        theta, phi = origin.axis
        # Add π to theta and π-phi to get antipodal point
        antipodal = [theta + np.pi, np.pi - phi]
        
        nearest = find_nearest_origin(antipodal)
        # Should find one of the faces near the antipodal point
        assert nearest != origin


def test_haversine():
    """Test haversine distance calculations."""
    # Test identical points
    point = [0, 0]
    assert haversine(point, point) == 0

    point2 = [np.pi/4, np.pi/3]
    assert haversine(point2, point2) == 0

    # Test symmetry
    p1 = [0, np.pi/4]
    p2 = [np.pi/2, np.pi/3]
    
    d1 = haversine(p1, p2)
    d2 = haversine(p2, p1)
    
    assert np.isclose(d1, d2)

    # Test increasing distance
    origin = [0, 0]
    distances = [
        [0, np.pi/6],      # 30°
        [0, np.pi/4],      # 45°
        [0, np.pi/3],      # 60°
        [0, np.pi/2],      # 90°
    ]

    last_distance = 0
    for point in distances:
        distance = haversine(origin, point)
        assert distance > last_distance
        last_distance = distance

    # Test longitude separation
    lat = np.pi/4  # Fixed latitude
    p1 = [0, lat]
    p2 = [np.pi, lat]
    p3 = [np.pi/2, lat]

    d1 = haversine(p1, p2)  # 180° separation
    d2 = haversine(p1, p3)  # 90° separation

    assert d1 > d2

    # Test known cases
    test_cases = [
        {
            "p1": [0, 0],
            "p2": [0, np.pi/2],
            "expected": 0.5  # sin²(π/4) = 0.5
        },
        {
            "p1": [0, np.pi/4],
            "p2": [np.pi/2, np.pi/4],
            "expected": 0.25  # For points at same latitude
        }
    ]

    for case in test_cases:
        assert np.isclose(haversine(case["p1"], case["p2"]), case["expected"], atol=1e-4)

def test_face_movement():
    """Test moving points between faces."""
    # First origin should be top
    origin1 = origins[0]
    assert np.array_equal(origin1.axis, [0, 0])

    # Move all the way to next origin
    origin2 = origins[1]
    direction = np.array([np.cos(origin2.axis[0]), np.sin(origin2.axis[0])])
    point = direction * 2 * distance_to_edge
    result = move_point_to_face(point, origin1, origin2)
    
    # Result should include new point and interface quaternion
    assert result.point is not None
    assert result.quat is not None
    
    # New point should be on second origin
    assert np.array_equal(result.point, [0, 0])


def test_quintant_conversion():
    """Test conversion between quintants and segments."""
    origin = origins[0]
    for quintant in range(5):
        segment, orientation = quintant_to_segment(quintant, origin)
        round_trip_quintant, round_trip_orientation = segment_to_quintant(segment, origin)
        assert round_trip_quintant == quintant 