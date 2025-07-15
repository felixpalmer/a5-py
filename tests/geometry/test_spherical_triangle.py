# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

import pytest
import json
import math
from pathlib import Path
import numpy as np
from a5.geometry.spherical_triangle import SphericalTriangleShape
from a5.core.coordinate_systems import Cartesian

# Load test fixtures
FIXTURES_DIR = Path(__file__).parent / "fixtures"
with open(FIXTURES_DIR / "spherical-triangle.json") as f:
    FIXTURES = json.load(f)

def is_close_to_array(actual: np.ndarray, expected: list, decimal: int = 6) -> bool:
    """Helper function to check if arrays are close within tolerance"""
    expected_array = np.array(expected)
    return np.allclose(actual, expected_array, atol=10**(-decimal), rtol=0)

def is_close_to(actual: float, expected: float, decimal: int = 6) -> bool:
    """Helper function to check if values are close within tolerance"""
    return abs(actual - expected) < 10**(-decimal)

class TestSphericalTriangleConstructor:
    """Test constructor functionality"""
    
    def test_requires_exactly_3_vertices(self):
        """Test that constructor requires exactly 3 vertices"""
        with pytest.raises(ValueError, match="SphericalTriangleShape requires exactly 3 vertices"):
            SphericalTriangleShape([])
        
        with pytest.raises(ValueError, match="SphericalTriangleShape requires exactly 3 vertices"):
            SphericalTriangleShape([
                np.array([1, 0, 0]), 
                np.array([0, 1, 0])
            ])
        
        with pytest.raises(ValueError, match="SphericalTriangleShape requires exactly 3 vertices"):
            SphericalTriangleShape([
                np.array([1, 0, 0]), 
                np.array([0, 1, 0]), 
                np.array([0, 0, 1]), 
                np.array([1, 1, 1])
            ])

    def test_accepts_exactly_3_vertices(self):
        """Test that constructor accepts exactly 3 vertices"""
        # Should not raise an exception
        triangle = SphericalTriangleShape([
            np.array([1, 0, 0]), 
            np.array([0, 1, 0]), 
            np.array([0, 0, 1])
        ])
        assert len(triangle.vertices) == 3


class TestSphericalTriangleGetBoundary:
    """Test getBoundary functionality"""
    
    def test_returns_boundary_points_with_different_segment_counts(self):
        """Test boundary points with different segment counts"""
        for i, fixture in enumerate(FIXTURES):
            triangle = SphericalTriangleShape([
                np.array(vertex, dtype=np.float64) for vertex in fixture["vertices"]
            ])
            
            # Test boundaries with 1-3 segments
            for n_segments in [1, 2, 3]:
                boundary = triangle.get_boundary(n_segments, True)
                expected_boundary = fixture[f"boundary{n_segments}"]
                
                assert len(boundary) == len(expected_boundary), \
                    f"Fixture {i}, segments {n_segments}: expected {len(expected_boundary)} points, got {len(boundary)}"
                
                for j, point in enumerate(boundary):
                    assert is_close_to_array(point, expected_boundary[j], 6), \
                        f"Fixture {i}, segments {n_segments}, point {j}: expected {expected_boundary[j]}, got {point.tolist()}"


class TestSphericalTriangleSlerp:
    """Test slerp functionality"""
    
    def test_interpolates_between_vertices(self):
        """Test spherical linear interpolation between vertices"""
        for i, fixture in enumerate(FIXTURES):
            triangle = SphericalTriangleShape([
                np.array(vertex, dtype=np.float64) for vertex in fixture["vertices"]
            ])
            
            for test_case in fixture["slerpTests"]:
                t = test_case["t"]
                expected = test_case["result"]
                
                actual = triangle.slerp(t)
                assert is_close_to_array(actual, expected, 6), \
                    f"Fixture {i}, t={t}: expected {expected}, got {actual.tolist()}"
                
                # Should be normalized
                length = np.linalg.norm(actual)
                assert abs(length - 1) < 1e-10, \
                    f"Fixture {i}, t={t}: result not normalized, length={length}"


class TestSphericalTriangleContainsPoint:
    """Test containsPoint functionality"""
    
    def test_correctly_identifies_points_inside_and_outside_triangle(self):
        """Test point containment detection"""
        for i, fixture in enumerate(FIXTURES):
            triangle = SphericalTriangleShape([
                np.array(vertex, dtype=np.float64) for vertex in fixture["vertices"]
            ])
            
            for test_case in fixture["containsPointTests"]:
                point = np.array(test_case["point"], dtype=np.float64)
                expected = test_case["result"]
                
                actual = triangle.contains_point(point)
                assert is_close_to(actual, expected, 6), \
                    f"Fixture {i}, point={point.tolist()}: expected {expected}, got {actual}"


class TestSphericalTriangleGetArea:
    """Test getArea functionality"""
    
    def test_returns_correct_area_for_all_triangles(self):
        """Test area calculation for all triangles"""
        for i, fixture in enumerate(FIXTURES):
            triangle = SphericalTriangleShape([
                np.array(vertex, dtype=np.float64) for vertex in fixture["vertices"]
            ])
            
            area = triangle.get_area()
            expected_area = fixture["area"]
            
            assert is_close_to(area, expected_area, 6), \
                f"Fixture {i}: expected area {expected_area}, got {area}"
            
            # Area can be negative for some winding orders, so check absolute value
            assert abs(area) > 0, \
                f"Fixture {i}: area should be non-zero, got {area}"
            assert abs(area) <= 2 * math.pi, \
                f"Fixture {i}: area should be <= 2π, got {abs(area)}" 