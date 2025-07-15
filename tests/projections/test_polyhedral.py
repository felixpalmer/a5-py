# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

import pytest
import json
import math
from pathlib import Path
import numpy as np
from a5.projections.polyhedral import PolyhedralProjection
from a5.core.coordinate_systems import Cartesian

# Load test fixtures
FIXTURES_DIR = Path(__file__).parent / "fixtures"
with open(FIXTURES_DIR / "polyhedral.json") as f:
    TEST_DATA = json.load(f)

# Extract static data from test data
TEST_SPHERICAL_TRIANGLE = TEST_DATA["static"]["TEST_SPHERICAL_TRIANGLE"]
TEST_FACE_TRIANGLE = TEST_DATA["static"]["TEST_FACE_TRIANGLE"]

AUTHALIC_RADIUS = 6371.0072  # km
MAX_ANGLE = max(
    np.arccos(np.clip(np.dot(TEST_SPHERICAL_TRIANGLE[0], TEST_SPHERICAL_TRIANGLE[1]), -1, 1)),
    np.arccos(np.clip(np.dot(TEST_SPHERICAL_TRIANGLE[1], TEST_SPHERICAL_TRIANGLE[2]), -1, 1)),
    np.arccos(np.clip(np.dot(TEST_SPHERICAL_TRIANGLE[2], TEST_SPHERICAL_TRIANGLE[0]), -1, 1))
)
MAX_ARC_LENGTH_MM = AUTHALIC_RADIUS * MAX_ANGLE * 1e9
DESIRED_MM_PRECISION = 0.01

def is_close_to_array(actual: np.ndarray, expected: list, decimal: int = 7) -> bool:
    """Helper function to check if arrays are close within tolerance"""
    expected_array = np.array(expected)
    # Use absolute tolerance - adjusted for cross-language floating point precision
    return np.allclose(actual, expected_array, atol=10**(-decimal), rtol=0)

@pytest.fixture
def polyhedral():
    return PolyhedralProjection()

class TestPolyhedralProjectionForward:
    """Test forward projection functionality"""
    
    def test_forward_projections(self, polyhedral):
        """Test forward projections match expected values"""
        for test_case in TEST_DATA["forward"]:
            result = polyhedral.forward(
                np.array(test_case["input"], dtype=np.float64), 
                TEST_SPHERICAL_TRIANGLE, 
                TEST_FACE_TRIANGLE
            )
            assert is_close_to_array(result, test_case["expected"]), \
                f"Expected {test_case['expected']}, got {result.tolist()}"

    def test_round_trip_forward_projections(self, polyhedral):
        """Test round trip forward projections"""
        largest_error = 0
        
        for test_case in TEST_DATA["forward"]:
            spherical = np.array(test_case["input"], dtype=np.float64)
            polar = polyhedral.forward(spherical, TEST_SPHERICAL_TRIANGLE, TEST_FACE_TRIANGLE)
            result = polyhedral.inverse(polar, TEST_FACE_TRIANGLE, TEST_SPHERICAL_TRIANGLE)
            error = np.linalg.norm(result - spherical)
            largest_error = max(largest_error, error)
            assert is_close_to_array(result, spherical.tolist()), \
                f"Round trip failed: expected {spherical.tolist()}, got {result.tolist()}"
        
        # Check precision requirement
        assert largest_error * MAX_ARC_LENGTH_MM < DESIRED_MM_PRECISION, \
            f"Accuracy requirement not met: {largest_error * MAX_ARC_LENGTH_MM:.6f}mm > {DESIRED_MM_PRECISION}mm"


class TestPolyhedralProjectionInverse:
    """Test inverse projection functionality"""
    
    def test_inverse_projections(self, polyhedral):
        """Test inverse projections match expected values"""
        for test_case in TEST_DATA["inverse"]:
            result = polyhedral.inverse(
                np.array(test_case["input"], dtype=np.float64),
                TEST_FACE_TRIANGLE,
                TEST_SPHERICAL_TRIANGLE
            )
            assert is_close_to_array(result, test_case["expected"]), \
                f"Expected {test_case['expected']}, got {result.tolist()}"

    def test_round_trip_inverse_projections(self, polyhedral):
        """Test round trip inverse projections"""
        for test_case in TEST_DATA["inverse"]:
            face_point = np.array(test_case["input"], dtype=np.float64)
            spherical = polyhedral.inverse(face_point, TEST_FACE_TRIANGLE, TEST_SPHERICAL_TRIANGLE)
            result = polyhedral.forward(spherical, TEST_SPHERICAL_TRIANGLE, TEST_FACE_TRIANGLE)
            assert is_close_to_array(result, face_point.tolist()), \
                f"Round trip failed: expected {face_point.tolist()}, got {result.tolist()}" 