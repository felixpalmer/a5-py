import math
import pytest
from a5.core.triangle import Triangle

def test_points_inside_triangle():
    # Create a simple right triangle
    triangle = Triangle(
        [0, 0],   # Origin
        [1, 0],   # 1 unit right
        [0, 1]    # 1 unit up
    )

    # Test points that should be inside
    inside_points = [
        [0.25, 0.25],  # Center-ish
        [0.1, 0.1],    # Near origin
        [0.9, 0.05],   # Near right edge
        [0.05, 0.9],   # Near top edge
    ]

    for point in inside_points:
        assert triangle.contains_point(point)

    # Test points that should be outside
    outside_points = [
        [-0.1, -0.1],  # Below and left
        [1.1, 0],      # Right
        [0, 1.1],      # Above
        [1, 1],        # Above and right
        [0.6, 0.6],    # Outside hypotenuse
    ]

    for point in outside_points:
        assert not triangle.contains_point(point)

def test_degenerate_triangles():
    # Line segment
    line_triangle = Triangle(
        [0, 0],
        [1, 0],
        [2, 0]
    )

    assert not line_triangle.contains_point([0.5, 0])

    # Single point
    point_triangle = Triangle(
        [1, 1],
        [1, 1],
        [1, 1]
    )

    assert not point_triangle.contains_point([1, 1])

def test_different_sized_triangles():
    # Large triangle
    large_triangle = Triangle(
        [0, 0],
        [100, 0],
        [50, 86.6]  # Equilateral triangle
    )

    assert large_triangle.contains_point([50, 43.3])
    assert not large_triangle.contains_point([0, 100])

    # Tiny triangle
    tiny_triangle = Triangle(
        [0, 0],
        [0.001, 0],
        [0, 0.001]
    )

    assert tiny_triangle.contains_point([0.0005, 0.0005])
    assert not tiny_triangle.contains_point([0.002, 0.002])

def test_different_orientations():
    angles = [45, 90, 135, 180]
    radius = 1

    for angle in angles:
        radians = math.radians(angle)
        triangle = Triangle(
            [0, 0],
            [radius * math.cos(radians), radius * math.sin(radians)],
            [-radius * math.sin(radians), radius * math.cos(radians)]
        )

        # Test point that should be inside
        inside_point = [
            0.2 * math.cos(radians),
            0.2 * math.sin(radians)
        ]
        assert triangle.contains_point(inside_point)

        # Test point that should be outside
        outside_point = [
            2 * math.cos(radians),
            2 * math.sin(radians)
        ]
        assert not triangle.contains_point(outside_point)
