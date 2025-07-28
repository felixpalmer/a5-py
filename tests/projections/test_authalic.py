# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

import pytest
import json
from pathlib import Path
from a5.projections.authalic import AuthalicProjection
from a5.core.coordinate_systems import Radians
from tests.matchers import is_close

# Load test fixtures
FIXTURES_DIR = Path(__file__).parent / "fixtures"
with open(FIXTURES_DIR / "authalic.json") as f:
    TEST_DATA = json.load(f)


@pytest.fixture
def authalic():
    return AuthalicProjection()

def test_forward_projections(authalic):
    """Test forward projections match expected values"""
    for test_case in TEST_DATA["forward"]:
        result = authalic.forward(test_case["input"])
        assert is_close(result, test_case["expected"]), \
            f"Expected {test_case['expected']}, got {result}"

def test_round_trip_forward_projections(authalic):
    """Test forward projections can be reversed accurately"""
    for test_case in TEST_DATA["forward"]:
        geodetic = test_case["input"]
        authalic_lat = authalic.forward(geodetic)
        result = authalic.inverse(authalic_lat)
        assert is_close(result, geodetic), \
            f"Expected {geodetic}, got {result}"

def test_inverse_projections(authalic):
    """Test inverse projections match expected values"""
    for test_case in TEST_DATA["inverse"]:
        result = authalic.inverse(test_case["input"])
        assert is_close(result, test_case["expected"]), \
            f"Expected {test_case['expected']}, got {result}"

def test_round_trip_inverse_projections(authalic):
    """Test inverse projections can be reversed accurately"""
    for test_case in TEST_DATA["inverse"]:
        authalic_lat = test_case["input"]
        geodetic = authalic.inverse(authalic_lat)
        result = authalic.forward(geodetic)
        assert is_close(result, authalic_lat), \
            f"Expected {authalic_lat}, got {result}" 