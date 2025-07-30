# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

import pytest
import json
from pathlib import Path
from a5.projections.dodecahedron import DodecahedronProjection, OriginId
from tests.matchers import is_close_array

# Load test fixtures
FIXTURES_DIR = Path(__file__).parent / "fixtures"
with open(FIXTURES_DIR / "dodecahedron.json") as f:
    TEST_DATA = json.load(f)

# Extract static data from test data
ORIGIN_ID = TEST_DATA["static"]["ORIGIN_ID"]


@pytest.fixture
def dodecahedron():
    return DodecahedronProjection()


class TestDodecahedronProjectionForward:
    """Test forward projection functionality"""
    
    def test_forward_projections(self, dodecahedron):
        """Test forward projections match expected values"""
        for test_case in TEST_DATA["forward"]:
            result = dodecahedron.forward(
                tuple(test_case["input"]), 
                ORIGIN_ID
            )
            assert is_close_array(result, test_case["expected"]), \
                f"Expected {test_case['expected']}, got {result}"

    def test_round_trip_forward_projections(self, dodecahedron):
        """Test round trip forward projections"""
        for test_case in TEST_DATA["forward"]:
            spherical = tuple(test_case["input"])
            face = dodecahedron.forward(spherical, ORIGIN_ID)
            result = dodecahedron.inverse(face, ORIGIN_ID)
            assert is_close_array(result, list(spherical)), \
                f"Round trip failed: expected {spherical}, got {result}"


class TestDodecahedronProjectionInverse:
    """Test inverse projection functionality"""
    
    def test_inverse_projections(self, dodecahedron):
        """Test inverse projections match expected values"""
        for test_case in TEST_DATA["inverse"]:
            result = dodecahedron.inverse(
                tuple(test_case["input"]),
                ORIGIN_ID
            )
            assert is_close_array(result, test_case["expected"]), \
                f"Expected {test_case['expected']}, got {result}"

    def test_round_trip_inverse_projections(self, dodecahedron):
        """Test round trip inverse projections"""
        for test_case in TEST_DATA["inverse"]:
            face_point = tuple(test_case["input"])
            spherical = dodecahedron.inverse(face_point, ORIGIN_ID)
            result = dodecahedron.forward(spherical, ORIGIN_ID)
            assert is_close_array(result, list(face_point)), \
                f"Round trip failed: expected {face_point}, got {result}" 