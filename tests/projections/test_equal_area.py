# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

import pytest
import json
import math
from pathlib import Path
from a5.projections.equal_area import EqualAreaProjection
from a5.projections.dodecahedron import DodecahedronProjection
from a5.projections.crs import CRS
from a5.core.coordinate_systems import Cartesian
from a5.math.vec3 import length
from tests.matchers import is_close_array

# Load test fixtures
FIXTURES_DIR = Path(__file__).parent / "fixtures"
with open(FIXTURES_DIR / "equal-area.json") as f:
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
def equal_area():
    return EqualAreaProjection(TEST_SPHERICAL_TRIANGLE)

class TestEqualAreaProjectionForward:
    """Test forward projection functionality"""

    def test_forward_projections(self, equal_area):
        """Test forward projections match expected values"""
        for test_case in TEST_DATA["forward"]:
            result = equal_area.forward(
                test_case["input"],
                TEST_SPHERICAL_TRIANGLE,
                TEST_FACE_TRIANGLE
            )
            assert is_close_array(list(result), test_case["expected"]), \
                f"Expected {test_case['expected']}, got {list(result)}"

    def test_round_trip_forward_projections(self, equal_area):
        """Test round trip forward projections"""
        largest_error = 0

        for test_case in TEST_DATA["forward"]:
            spherical = test_case["input"]
            polar = equal_area.forward(spherical, TEST_SPHERICAL_TRIANGLE, TEST_FACE_TRIANGLE)
            result = equal_area.inverse(polar, TEST_FACE_TRIANGLE, TEST_SPHERICAL_TRIANGLE)
            error = length([r - s for r, s in zip(result, spherical)])
            largest_error = max(largest_error, error)
            assert is_close_array(list(result), spherical), \
                f"Round trip failed: expected {spherical}, got {list(result)}"

        # Check precision requirement
        assert largest_error * MAX_ARC_LENGTH_MM < DESIRED_MM_PRECISION, \
            f"Accuracy requirement not met: {largest_error * MAX_ARC_LENGTH_MM:.6f}mm > {DESIRED_MM_PRECISION}mm"


class TestEqualAreaProjectionInverse:
    """Test inverse projection functionality"""

    def test_inverse_projections(self, equal_area):
        """Test inverse projections match expected values"""
        for test_case in TEST_DATA["inverse"]:
            result = equal_area.inverse(
                test_case["input"],
                TEST_FACE_TRIANGLE,
                TEST_SPHERICAL_TRIANGLE
            )
            assert is_close_array(list(result), test_case["expected"]), \
                f"Expected {test_case['expected']}, got {list(result)}"

    def test_round_trip_inverse_projections(self, equal_area):
        """Test round trip inverse projections"""
        for test_case in TEST_DATA["inverse"]:
            face_point = test_case["input"]
            spherical = equal_area.inverse(face_point, TEST_FACE_TRIANGLE, TEST_SPHERICAL_TRIANGLE)
            result = equal_area.forward(spherical, TEST_SPHERICAL_TRIANGLE, TEST_FACE_TRIANGLE)
            assert is_close_array(list(result), face_point), \
                f"Round trip failed: expected {face_point}, got {list(result)}"


class TestEqualAreaProjectionTriangleConstants:
    """The projection caches shape constants from one canonical triangle, which
    is only valid if every spherical triangle the dodecahedron can supply is
    congruent AND consistently wound (the sign of V is chirality-sensitive).
    This enforces the invariant documented in equal_area.py."""

    def test_constants_agree_across_all_triangles(self):
        dodecahedron = DodecahedronProjection()
        canonical = EqualAreaProjection.compute_constants(CRS().get_canonical_triangle())

        RELATIVE_TOLERANCE = 1e-13
        for origin_id in range(12):
            for face_triangle_index in range(10):
                for reflected in (False, True):
                    triangle = dodecahedron.get_spherical_triangle(face_triangle_index, origin_id, reflected)
                    constants = EqualAreaProjection.compute_constants(triangle)
                    for key, expected in canonical.items():
                        actual = constants[key]
                        assert abs(actual - expected) < abs(expected) * RELATIVE_TOLERANCE, \
                            f"{key} mismatch at face {face_triangle_index}, origin {origin_id}, reflected {reflected}"
