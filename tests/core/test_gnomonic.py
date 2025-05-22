"""
Tests for a5.core.gnomonic module
"""

import json
import math
import os
import pytest
from typing import cast
from a5.core.gnomonic import project_gnomonic, unproject_gnomonic
from a5.core.coordinate_systems import Polar, Spherical

# Test values for basic projection tests
TEST_VALUES = [
    {"input": (0.001, 0.0), "expected": (0.0, 0.001)},
    {"input": (0.001, 0.321), "expected": (0.321, 0.001)},
    {"input": (1.0, math.pi), "expected": (math.pi, math.pi / 4)},
    {"input": (0.5, 0.777), "expected": (0.777, math.atan(0.5))},
]

@pytest.mark.parametrize("test_case", TEST_VALUES)
def test_project_gnomonic(test_case):
    """Test projection from polar to spherical coordinates."""
    input_coords = cast(Polar, test_case["input"])
    expected = cast(Spherical, test_case["expected"])
    result = project_gnomonic(input_coords)
    
    assert result[0] == pytest.approx(expected[0], rel=1e-4)
    assert result[1] == pytest.approx(expected[1], rel=1e-4)

@pytest.mark.parametrize("test_case", TEST_VALUES)
def test_unproject_gnomonic(test_case):
    """Test unprojection from spherical to polar coordinates."""
    input_coords = cast(Polar, test_case["input"])
    expected = cast(Spherical, test_case["expected"])
    result = unproject_gnomonic(expected)
    
    assert result[0] == pytest.approx(input_coords[0], rel=1e-4)
    assert result[1] == pytest.approx(input_coords[1], rel=1e-4)

def test_round_trip():
    """Test round trip conversion through projection and unprojection."""
    polar = cast(Polar, (0.3, 0.4))
    spherical = project_gnomonic(polar)
    result = unproject_gnomonic(spherical)
    
    assert result[0] == pytest.approx(polar[0], rel=1e-4)
    assert result[1] == pytest.approx(polar[1], rel=1e-4)

def test_polar_coordinates_round_trip():
    """Test round trip conversion for all test coordinates."""
    # Get the directory containing this test file
    test_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(test_dir, "test-polar-coordinates.json")
    
    # Load test coordinates from JSON file
    with open(json_path, "r") as f:
        test_coords = json.load(f)
    
    for coord in test_coords:
        polar = cast(Polar, (coord["rho"], coord["beta"]))
        spherical = project_gnomonic(polar)
        result = unproject_gnomonic(spherical)
        
        # Check that result values are close to original
        assert result[0] == pytest.approx(polar[0])
        assert result[1] == pytest.approx(polar[1]) 