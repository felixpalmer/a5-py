"""
Tests for a5.core.math module
"""

import pytest
import math
from typing import cast, Tuple, List
from a5.core.coordinate_transforms import (
    deg_to_rad,
    rad_to_deg,
    to_cartesian,
    to_spherical,
    from_lonlat,
    to_lonlat,
    face_to_barycentric,
    barycentric_to_face,
    normalize_longitudes,
    Contour
)
from a5.core.coordinate_systems import (
    Degrees, Radians, Spherical, LonLat, Face, Barycentric, FaceTriangle
)

# Test points for coordinate conversions
TEST_POINTS_LONLAT: List[LonLat] = [
    cast(LonLat, (0.0, 0.0)),      # Equator
    cast(LonLat, (90.0, 0.0)),     # Equator
    cast(LonLat, (180.0, 0.0)),    # Equator
    cast(LonLat, (0.0, 45.0)),     # Mid latitude
    cast(LonLat, (0.0, -45.0)),    # Mid latitude
    cast(LonLat, (-90.0, -45.0)),  # West hemisphere mid-latitude
    cast(LonLat, (180.0, 45.0)),   # Date line mid-latitude
    cast(LonLat, (90.0, 45.0)),    # East hemisphere mid-latitude
    cast(LonLat, (0.0, 90.0)),     # North pole
    cast(LonLat, (0.0, -90.0)),    # South pole
    cast(LonLat, (123.0, 45.0)),   # Arbitrary point
]

# Test triangle for barycentric tests
TEST_TRIANGLE = cast(FaceTriangle, (
    cast(Face, (0.0, 0.0)),
    cast(Face, (1.0, 0.0)),
    cast(Face, (0.0, 1.0))
))

def test_angle_conversions():
    """Test degree to radian conversions and vice versa."""
    # Test degrees to radians
    assert deg_to_rad(cast(Degrees, 180.0)) == pytest.approx(math.pi)
    assert deg_to_rad(cast(Degrees, 90.0)) == pytest.approx(math.pi / 2)
    assert deg_to_rad(cast(Degrees, 0.0)) == pytest.approx(0.0)

    # Test radians to degrees
    assert rad_to_deg(cast(Radians, math.pi)) == pytest.approx(180.0)
    assert rad_to_deg(cast(Radians, math.pi / 2)) == pytest.approx(90.0)
    assert rad_to_deg(cast(Radians, 0.0)) == pytest.approx(0.0)

def test_barycentric_coordinate_functions():
    """Test barycentric coordinate conversions."""
    TOLERANCE = 1e-12

    # Test round-trip conversion for test points
    test_points = [
        cast(Face, (0.1, 0.1)),
        cast(Face, (0.5, 0.2)),
        cast(Face, (0.3, 0.3)),
        cast(Face, (0.1, 0.8)),
    ]

    for point in test_points:
        # Convert to barycentric coordinates
        bary = face_to_barycentric(point, TEST_TRIANGLE)
        
        # Convert back to face coordinates
        result = barycentric_to_face(bary, TEST_TRIANGLE)
        
        # Check round-trip accuracy
        assert all(abs(r - p) < TOLERANCE * max(abs(r), abs(p), 1.0) for r, p in zip(result, point))
        
        # Check that barycentric coordinates sum to 1
        assert abs(sum(bary) - 1.0) < TOLERANCE
        
        # Check that all barycentric coordinates are non-negative
        assert all(b >= 0 for b in bary)

def test_barycentric_specific_coordinates():
    """Test specific barycentric coordinate cases."""
    TOLERANCE = 1e-12
    
    # Test specific barycentric coordinates
    test_bary_coords = [
        cast(Barycentric, (0.043821975867140296, 0.9561208684797726, 0.00005715565308705983)),
        cast(Barycentric, (0.5, 0.3, 0.2)),
        cast(Barycentric, (0.1, 0.8, 0.1)),
        cast(Barycentric, (0.33, 0.33, 0.34)),
        cast(Barycentric, (0.9, 0.05, 0.05)),
        cast(Barycentric, (0.001, 0.999, 0.000)),
    ]
    
    for bary in test_bary_coords:
        # Convert barycentric to face coordinates
        face = barycentric_to_face(bary, TEST_TRIANGLE)
        
        # Convert back to barycentric
        result_bary = face_to_barycentric(face, TEST_TRIANGLE)
        
        # Check round-trip accuracy
        assert all(abs(r - b) < TOLERANCE * max(abs(r), abs(b), 1.0) for r, b in zip(result_bary, bary))
        
        # Check that barycentric coordinates sum to 1
        assert abs(sum(result_bary) - 1.0) < TOLERANCE

def test_barycentric_vertices():
    """Test barycentric coordinates at triangle vertices."""
    TOLERANCE = 1e-12
    
    # Test each vertex
    vertices = [TEST_TRIANGLE[0], TEST_TRIANGLE[1], TEST_TRIANGLE[2]]
    expected_bary = [
        (1.0, 0.0, 0.0),  # First vertex
        (0.0, 1.0, 0.0),  # Second vertex
        (0.0, 0.0, 1.0),  # Third vertex
    ]
    
    for vertex, expected in zip(vertices, expected_bary):
        bary = face_to_barycentric(vertex, TEST_TRIANGLE)
        
        # Check barycentric coordinates
        assert all(abs(b - e) < TOLERANCE * max(abs(b), abs(e), 1.0) for b, e in zip(bary, expected))
        
        # Round-trip test
        result = barycentric_to_face(bary, TEST_TRIANGLE)
        assert all(abs(r - v) < TOLERANCE * max(abs(r), abs(v), 1.0) for r, v in zip(result, vertex))

def test_barycentric_edge_midpoints():
    """Test barycentric coordinates at edge midpoints."""
    TOLERANCE = 1e-12
    
    edge_midpoints = [
        cast(Face, (0.5, 0.0)),    # Midpoint of first-second edge
        cast(Face, (0.0, 0.5)),    # Midpoint of first-third edge
        cast(Face, (0.5, 0.5)),    # Midpoint of second-third edge
    ]
    
    expected_bary = [
        (0.5, 0.5, 0.0),  # First-second edge midpoint
        (0.5, 0.0, 0.5),  # First-third edge midpoint
        (0.0, 0.5, 0.5),  # Second-third edge midpoint
    ]
    
    for midpoint, expected in zip(edge_midpoints, expected_bary):
        bary = face_to_barycentric(midpoint, TEST_TRIANGLE)
        
        # Check barycentric coordinates
        assert all(abs(b - e) < TOLERANCE * max(abs(b), abs(e), 1.0) for b, e in zip(bary, expected))
        
        # Round-trip test
        result = barycentric_to_face(bary, TEST_TRIANGLE)
        assert all(abs(r - m) < TOLERANCE * max(abs(r), abs(m), 1.0) for r, m in zip(result, midpoint))

def test_spherical_to_cartesian():
    """Test conversion from spherical to cartesian coordinates."""
    # Test north pole
    north_pole = to_cartesian(cast(Spherical, (0.0, 0.0)))
    assert all(abs(n - e) < 1e-15 for n, e in zip(north_pole, [0.0, 0.0, 1.0]))

    # Test equator at 0 longitude
    equator0 = to_cartesian(cast(Spherical, (0.0, math.pi/2)))
    assert all(abs(e - ex) < 1e-15 for e, ex in zip(equator0, [1.0, 0.0, 0.0]))

    # Test equator at 90° longitude
    equator90 = to_cartesian(cast(Spherical, (math.pi/2, math.pi/2)))
    assert all(abs(e - ex) < 1e-15 for e, ex in zip(equator90, [0.0, 1.0, 0.0]))

def test_cartesian_to_spherical():
    """Test conversion from cartesian to spherical coordinates."""
    # Test round trip conversion
    original = cast(Spherical, (math.pi/4, math.pi/6))
    cartesian = to_cartesian(original)
    spherical = to_spherical(cartesian)
    
    assert all(abs(s - o) < 1e-15 for s, o in zip(spherical, original))

def test_lonlat_to_spherical():
    """Test conversion from longitude/latitude to spherical coordinates."""
    # Test Greenwich equator
    greenwich = from_lonlat(cast(LonLat, (0.0, 0.0)))
    # Match OFFSET_LON: 93
    assert greenwich[0] == pytest.approx(deg_to_rad(cast(Degrees, 93.0)))
    assert greenwich[1] == pytest.approx(math.pi/2)  # 90° colatitude = equator

    # Test north pole
    north_pole = from_lonlat(cast(LonLat, (0.0, 90.0)))
    assert north_pole[1] == pytest.approx(0.0)  # 0° colatitude = north pole

    # Test south pole
    south_pole = from_lonlat(cast(LonLat, (0.0, -90.0)))
    assert south_pole[1] == pytest.approx(math.pi)  # 180° colatitude = south pole

def test_spherical_to_lonlat():
    """Test conversion from spherical to longitude/latitude coordinates."""
    # Test round trip conversion
    for lon_lat in TEST_POINTS_LONLAT:
        spherical = from_lonlat(lon_lat)
        result = to_lonlat(spherical)
        
        assert all(abs(r - l) < 1e-10 for r, l in zip([result[0], result[1]], [lon_lat[0], lon_lat[1]]))

def test_normalize_longitudes():
    """Test longitude normalization for contours."""
    # Test simple contour without wrapping
    contour: Contour = [
        cast(LonLat, (0.0, 0.0)),
        cast(LonLat, (10.0, 0.0)),
        cast(LonLat, (10.0, 10.0)),
        cast(LonLat, (0.0, 10.0)),
        cast(LonLat, (0.0, 0.0))
    ]
    normalized = normalize_longitudes(contour)
    assert normalized == contour

    # Test contour crossing antimeridian
    contour = [
        cast(LonLat, (170.0, 0.0)), # This should become -190
        cast(LonLat, (175.0, 0.0)), # This should become -185
        cast(LonLat, (-175.0, 0.0)),
        cast(LonLat, (-170.0, 0.0)),
    ]
    normalized = normalize_longitudes(contour)
    assert normalized[0][0] == pytest.approx(-190.0)
    assert normalized[1][0] == pytest.approx(-185.0)

    # Test contour crossing antimeridian in opposite direction
    contour = [
        cast(LonLat, (-170.0, 0.0)),
        cast(LonLat, (-175.0, 0.0)),
        cast(LonLat, (-180.0, 0.0)),
        cast(LonLat, (175.0, 0.0)),   # This should become -185
        cast(LonLat, (170.0, 0.0)),   # This should become -190
    ]
    normalized = normalize_longitudes(contour)
    assert normalized[3][0] == pytest.approx(-185.0)
    assert normalized[4][0] == pytest.approx(-190.0) 