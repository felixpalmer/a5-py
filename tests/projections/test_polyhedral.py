# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

import pytest
import json
import math
from pathlib import Path
from a5.projections.polyhedral import PolyhedralProjection
from a5.core.coordinate_systems import Cartesian
from a5.math.vec3 import length
from tests.matchers import is_close_array

# Load test fixtures
FIXTURES_DIR = Path(__file__).parent / "fixtures"
with open(FIXTURES_DIR / "polyhedral.json") as f:
    TEST_DATA = json.load(f)

# Extract static data from test data
TEST_SPHERICAL_TRIANGLE = TEST_DATA["static"]["TEST_SPHERICAL_TRIANGLE"]
TEST_FACE_TRIANGLE = TEST_DATA["static"]["TEST_FACE_TRIANGLE"]

AUTHALIC_RADIUS = 6371.0072  # km
def dot_product(a, b):
    return sum(x * y for x, y in zip(a, b))

MAX_ANGLE = max(
    math.acos(max(-1, min(1, dot_product(TEST_SPHERICAL_TRIANGLE[0], TEST_SPHERICAL_TRIANGLE[1])))),
    math.acos(max(-1, min(1, dot_product(TEST_SPHERICAL_TRIANGLE[1], TEST_SPHERICAL_TRIANGLE[2])))),
    math.acos(max(-1, min(1, dot_product(TEST_SPHERICAL_TRIANGLE[2], TEST_SPHERICAL_TRIANGLE[0]))))
)
MAX_ARC_LENGTH_MM = AUTHALIC_RADIUS * MAX_ANGLE * 1e9
DESIRED_MM_PRECISION = 0.01


@pytest.fixture
def polyhedral():
    return PolyhedralProjection()

class TestPolyhedralProjectionForward:
    """Test forward projection functionality"""
    
    def test_forward_projections(self, polyhedral):
        """Test forward projections match expected values"""
        for test_case in TEST_DATA["forward"]:
            result = polyhedral.forward(
                test_case["input"],
                TEST_SPHERICAL_TRIANGLE, 
                TEST_FACE_TRIANGLE
            )
            assert is_close_array(list(result), test_case["expected"]), \
                f"Expected {test_case['expected']}, got {list(result)}"

    def test_round_trip_forward_projections(self, polyhedral):
        """Test round trip forward projections"""
        largest_error = 0
        
        for test_case in TEST_DATA["forward"]:
            spherical = test_case["input"]
            polar = polyhedral.forward(spherical, TEST_SPHERICAL_TRIANGLE, TEST_FACE_TRIANGLE)
            result = polyhedral.inverse(polar, TEST_FACE_TRIANGLE, TEST_SPHERICAL_TRIANGLE)
            error = length([r - s for r, s in zip(result, spherical)])
            largest_error = max(largest_error, error)
            assert is_close_array(list(result), spherical), \
                f"Round trip failed: expected {spherical}, got {list(result)}"
        
        # Check precision requirement
        assert largest_error * MAX_ARC_LENGTH_MM < DESIRED_MM_PRECISION, \
            f"Accuracy requirement not met: {largest_error * MAX_ARC_LENGTH_MM:.6f}mm > {DESIRED_MM_PRECISION}mm"


class TestPolyhedralProjectionInverse:
    """Test inverse projection functionality"""
    
    def test_inverse_projections(self, polyhedral):
        """Test inverse projections match expected values"""
        for test_case in TEST_DATA["inverse"]:
            result = polyhedral.inverse(
                test_case["input"],
                TEST_FACE_TRIANGLE,
                TEST_SPHERICAL_TRIANGLE
            )
            assert is_close_array(list(result), test_case["expected"]), \
                f"Expected {test_case['expected']}, got {list(result)}"

    def test_round_trip_inverse_projections(self, polyhedral):
        """Test round trip inverse projections"""
        for test_case in TEST_DATA["inverse"]:
            face_point = test_case["input"]
            spherical = polyhedral.inverse(face_point, TEST_FACE_TRIANGLE, TEST_SPHERICAL_TRIANGLE)
            result = polyhedral.forward(spherical, TEST_SPHERICAL_TRIANGLE, TEST_FACE_TRIANGLE)
            assert is_close_array(list(result), face_point), \
                f"Round trip failed: expected {face_point}, got {list(result)}" 