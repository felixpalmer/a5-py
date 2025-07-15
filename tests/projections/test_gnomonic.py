# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

import pytest
import json
from pathlib import Path
import numpy as np
from a5.projections.gnomonic import GnomonicProjection
from a5.core.coordinate_systems import Polar, Spherical

# Load test fixtures
FIXTURES_DIR = Path(__file__).parent / "fixtures"
with open(FIXTURES_DIR / "gnomonic.json") as f:
    TEST_DATA = json.load(f)

def is_close_to_array(actual: tuple, expected: list, decimal: int = 6) -> bool:
    """Helper function to check if arrays are close within tolerance"""
    return np.allclose(actual, expected, rtol=10**(-decimal))

@pytest.fixture
def gnomonic():
    return GnomonicProjection()

def test_forward_projections(gnomonic):
    """Test forward projections match expected values"""
    for test_case in TEST_DATA["forward"]:
        result = gnomonic.forward(tuple(test_case["input"]))
        assert is_close_to_array(result, test_case["expected"]), \
            f"Expected {test_case['expected']}, got {result}"

def test_round_trip_forward_projections(gnomonic):
    """Test forward projections can be reversed accurately"""
    for test_case in TEST_DATA["forward"]:
        spherical = tuple(test_case["input"])
        polar = gnomonic.forward(spherical)
        result = gnomonic.inverse(polar)
        assert is_close_to_array(result, spherical), \
            f"Expected {spherical}, got {result}"

def test_inverse_projections(gnomonic):
    """Test inverse projections match expected values"""
    for test_case in TEST_DATA["inverse"]:
        result = gnomonic.inverse(tuple(test_case["input"]))
        assert is_close_to_array(result, test_case["expected"]), \
            f"Expected {test_case['expected']}, got {result}"

def test_round_trip_inverse_projections(gnomonic):
    """Test inverse projections can be reversed accurately"""
    for test_case in TEST_DATA["inverse"]:
        polar = tuple(test_case["input"])
        spherical = gnomonic.inverse(polar)
        result = gnomonic.forward(spherical)
        assert is_close_to_array(result, polar), \
            f"Expected {polar}, got {result}" 