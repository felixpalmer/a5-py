# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

import pytest
import json
import math
from pathlib import Path
from a5.projections.crs import CRS
from a5.core.coordinate_systems import Cartesian

# Load test fixtures
FIXTURES_DIR = Path(__file__).parent / "fixtures"
with open(FIXTURES_DIR / "crs-vertices.json") as f:
    EXPECTED_VERTICES = json.load(f)

def is_close_to_array(actual: list, expected: list, decimal: int = 7) -> bool:
    """Helper function to check if arrays are close within tolerance"""
    # Use absolute tolerance - adjusted for cross-language floating point precision
    tolerance = 10**(-decimal)
    if len(actual) != len(expected):
        return False
    return all(abs(a - e) < tolerance for a, e in zip(actual, expected))

@pytest.fixture
def crs():
    return CRS()

def test_should_have_exactly_62_vertices(crs):
    """Test that CRS has exactly 62 vertices"""
    vertices = crs.vertices
    assert len(vertices) == 62

def test_should_match_expected_vertices_from_json_file(crs):
    """Test that CRS vertices match expected values from JSON fixture"""
    vertices = crs.vertices
    
    assert len(vertices) == len(EXPECTED_VERTICES)
    for i, vertex in enumerate(vertices):
        expected = EXPECTED_VERTICES[i]
        assert is_close_to_array(list(vertex), expected), \
            f"Vertex {i}: expected {expected}, got {list(vertex)}"

def test_should_throw_error_for_non_existent_vertex(crs):
    """Test that get_vertex raises error for non-existent vertex"""
    non_vertex_point = (1, 0, 0)
    with pytest.raises(ValueError, match="Failed to find vertex in CRS"):
        crs.get_vertex(non_vertex_point)

def test_should_validate_vertex_structure(crs):
    """Test that all vertices are normalized (unit length)"""
    vertices = crs.vertices
    
    # All vertices should be normalized (unit length)
    for i, vertex in enumerate(vertices):
        length = math.sqrt(sum(x*x for x in vertex))
        assert abs(length - 1.0) < 1e-15, \
            f"Vertex {i} is not normalized: length = {length}" 