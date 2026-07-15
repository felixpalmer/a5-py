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
from tests.matchers import is_close_array, is_close

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
    """The projection caches constants from one canonical triangle and reuses
    them for every face. That is valid because all dodecahedron face triangles
    are congruent and consistently wound:
      - volumeABC and area_abc are identical on every face (forward relies on this);
      - A·B / A·C only ever take the two canonical values — they are equal for
        "even" faces and swapped for the mirror-image "odd" faces, which
        inverse() handles by swapping B↔C — so on even faces the cached
        alphaTransform matrix matches exactly."""

    def test_constants_agree_across_all_triangles(self):
        dodecahedron = DodecahedronProjection()
        canonical = EqualAreaProjection.compute_constants(CRS().get_canonical_triangle())

        for origin_id in range(12):
            for face_triangle_index in range(10):
                for reflected in (False, True):
                    triangle = dodecahedron.get_spherical_triangle(face_triangle_index, origin_id, reflected)
                    c = EqualAreaProjection.compute_constants(triangle)
                    where = f"face {face_triangle_index}, origin {origin_id}, reflected {reflected}"

                    # Invariant on every face.
                    assert is_close(c['volumeABC'], canonical['volumeABC'], 12), f"volumeABC at {where}"
                    assert is_close(c['area_abc'], canonical['area_abc'], 12), f"area_abc at {where}"

                    # A·B / A·C take the two canonical values; the orientation is
                    # whichever canonical value A·B is nearer to (as inverse() uses).
                    even = abs(c['AdotB'] - canonical['AdotB']) < abs(c['AdotB'] - canonical['AdotC'])
                    if even:
                        assert is_close(c['AdotB'], canonical['AdotB'], 12), f"A·B at {where}"
                        assert is_close(c['AdotC'], canonical['AdotC'], 12), f"A·C at {where}"
                        # The cached coefficient matrix matches exactly on even faces.
                        assert is_close_array(c['alphaTransform'], canonical['alphaTransform']), \
                            f"alphaTransform at {where}"
                    else:
                        # Mirror-image face: A·B and A·C swapped (inverse() swaps B↔C).
                        assert is_close(c['AdotB'], canonical['AdotC'], 12), f"A·B at {where}"
                        assert is_close(c['AdotC'], canonical['AdotB'], 12), f"A·C at {where}"
