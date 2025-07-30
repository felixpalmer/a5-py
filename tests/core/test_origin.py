"""
Tests for origin-related functionality.
"""

import pytest
import json
import math
from pathlib import Path
from tests.matchers import is_close_array

from a5.core.origin import (    
    origins,
    find_nearest_origin,
    haversine,
    quintant_to_segment,
    segment_to_quintant
)
from a5.core.constants import PI_OVER_5
from a5.core.coordinate_transforms import to_cartesian
from a5.math.vec3 import length
from a5.math.quat import length as quat_length

# Load test fixtures
FIXTURES_DIR = Path(__file__).parent / "fixtures"
with open(FIXTURES_DIR / "origins.json") as f:
    EXPECTED_ORIGINS = json.load(f)

def test_origin_constants():
    """Test that we have 12 origins for dodecahedron faces."""
    assert len(origins) == 12


def test_should_match_expected_origins_from_json_file():
    """Test that origins match expected values from JSON fixture"""
    assert len(origins) == len(EXPECTED_ORIGINS)
    for i, origin in enumerate(origins):
        expected = EXPECTED_ORIGINS[i]
        assert origin.id == expected['id']
        assert origin.angle == expected['angle']
        assert origin.orientation == expected['orientation']
        assert origin.first_quintant == expected['firstQuintant']
        assert is_close_array(origin.axis, expected['axis']), \
            f"Origin {i} axis wrong: expected {expected['axis']}, got {origin.axis}"
        assert is_close_array(origin.quat, expected['quat']), \
            f"Origin {i} quat wrong: expected {expected['quat']}, got {origin.quat}"

def test_origin_properties():
    """Test that each origin has required properties."""
    for origin in origins:
        # Check properties exist
        assert origin.axis is not None
        assert origin.quat is not None
        assert origin.angle is not None
        
        # Check axis is unit vector when converted to cartesian
        cartesian = to_cartesian(origin.axis)
        vector_length = length(cartesian)
        assert abs(vector_length - 1.0) < 1e-15
        
        # Check quaternion is normalized
        q_length = quat_length(origin.quat)
        assert abs(q_length - 1.0) < 1e-10


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
        {"point": [0, math.pi - PI_OVER_5/2], "expected_origins": [9, 10]},
    ]

    for test_case in boundary_points:
        nearest = find_nearest_origin(test_case["point"])
        assert nearest.id in test_case["expected_origins"]

    # Test antipodal points
    for origin in origins:
        theta, phi = origin.axis
        # Add π to theta and π-phi to get antipodal point
        antipodal = [theta + math.pi, math.pi - phi]
        
        nearest = find_nearest_origin(antipodal)
        # Should find one of the faces near the antipodal point
        assert nearest != origin


def test_haversine():
    """Test haversine distance calculations."""
    # Test identical points
    point = [0, 0]
    assert haversine(point, point) == 0

    point2 = [math.pi/4, math.pi/3]
    assert haversine(point2, point2) == 0

    # Test symmetry
    p1 = [0, math.pi/4]
    p2 = [math.pi/2, math.pi/3]
    
    d1 = haversine(p1, p2)
    d2 = haversine(p2, p1)
    
    assert abs(d1 - d2) < 1e-15

    # Test increasing distance
    origin = [0, 0]
    distances = [
        [0, math.pi/6],      # 30°
        [0, math.pi/4],      # 45°
        [0, math.pi/3],      # 60°
        [0, math.pi/2],      # 90°
    ]

    last_distance = 0
    for point in distances:
        distance = haversine(origin, point)
        assert distance > last_distance
        last_distance = distance

    # Test longitude separation
    lat = math.pi/4  # Fixed latitude
    p1 = [0, lat]
    p2 = [math.pi, lat]
    p3 = [math.pi/2, lat]

    d1 = haversine(p1, p2)  # 180° separation
    d2 = haversine(p1, p3)  # 90° separation

    assert d1 > d2

    # Test known cases
    test_cases = [
        {
            "p1": [0, 0],
            "p2": [0, math.pi/2],
            "expected": 0.5  # sin²(π/4) = 0.5
        },
        {
            "p1": [0, math.pi/4],
            "p2": [math.pi/2, math.pi/4],
            "expected": 0.25  # For points at same latitude
        }
    ]

    for case in test_cases:
        assert abs(haversine(case["p1"], case["p2"]) - case["expected"]) < 1e-4

def test_quintant_conversion():
    """Test conversion between quintants and segments."""
    origin = origins[0]
    for quintant in range(5):
        segment, orientation = quintant_to_segment(quintant, origin)
        round_trip_quintant, round_trip_orientation = segment_to_quintant(segment, origin)
        assert round_trip_quintant == quintant 