# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

import pytest
import json
import os
from pathlib import Path
import math
from a5.geometry import SphericalPolygonShape
from a5.core.coordinate_systems import Cartesian
from a5.math.vec3 import length
from tests.matchers import is_close_array

# Load test fixtures
FIXTURES_DIR = Path(__file__).parent / "fixtures"
with open(FIXTURES_DIR / "spherical-polygon.json") as f:
    fixtures = json.load(f)


def test_get_boundary():
    """Test boundary points with different segment counts"""
    for fixture in fixtures:
        polygon = SphericalPolygonShape([tuple(v) for v in fixture["vertices"]])
        
        # Test boundaries with 1-3 segments
        for n_segments in [1, 2, 3]:
            boundary = polygon.get_boundary(n_segments, True)
            expected_boundary = fixture[f"boundary{n_segments}"]
            assert len(boundary) == len(expected_boundary)
            for point, expected in zip(boundary, expected_boundary):
                assert is_close_array(list(point), expected, 6), \
                    f"Expected {expected}, got {list(point)}"

def test_slerp():
    """Test interpolation between vertices"""
    for fixture in fixtures:
        polygon = SphericalPolygonShape([tuple(v) for v in fixture["vertices"]])
        
        for test in fixture["slerpTests"]:
            actual = polygon.slerp(test["t"])
            assert is_close_array(list(actual), test["result"], 6), \
                f"Expected {test['result']}, got {list(actual)}"
            # Should be normalized
            magnitude = length(actual)
            assert abs(magnitude - 1) < 1e-10, \
                f"Vector not normalized, magnitude: {magnitude}"

def test_contains_point():
    """Test point containment checks"""
    for fixture in fixtures:
        polygon = SphericalPolygonShape([tuple(v) for v in fixture["vertices"]])
        
        for test in fixture["containsPointTests"]:
            point = tuple(test["point"])
            actual = polygon.contains_point(point)
            assert abs(actual - test["result"]) < 1e-6, \
                f"Expected {test['result']}, got {actual}"

def test_get_area():
    """Test area calculations"""
    for fixture in fixtures:
        polygon = SphericalPolygonShape([tuple(v) for v in fixture["vertices"]])
        area = polygon.get_area()
        assert abs(area - fixture["area"]) < 1e-6, \
            f"Expected {fixture['area']}, got {area}"
        # Area can be negative for some winding orders, so check absolute value
        assert abs(area) > 0, "Area should be non-zero"
        assert abs(area) <= 2 * math.pi, "Area should be less than 2π"

def test_degenerate_polygons():
    """Test area calculations for degenerate polygons"""
    # Empty polygon
    assert SphericalPolygonShape([]).get_area() == 0
    
    # Single point
    assert SphericalPolygonShape([(1, 0, 0)]).get_area() == 0
    
    # Two points
    assert SphericalPolygonShape([
        (1, 0, 0),
        (0, 1, 0)
    ]).get_area() == 0 