"""
Tests for a5.core.math module
"""

import pytest
import numpy as np
from typing import cast, Tuple
from a5.core.coordinate_transforms import (
    deg_to_rad,
    rad_to_deg,
    to_cartesian,
    to_spherical,
    from_lonlat,
    to_lonlat,
)
from a5.core.coordinate_systems import Degrees, Radians, Spherical, LonLat

def test_angle_conversions():
    """Test degree to radian conversions and vice versa."""
    # Test degrees to radians
    assert deg_to_rad(cast(Degrees, 180.0)) == pytest.approx(np.pi)
    assert deg_to_rad(cast(Degrees, 90.0)) == pytest.approx(np.pi / 2)
    assert deg_to_rad(cast(Degrees, 0.0)) == pytest.approx(0.0)

    # Test radians to degrees
    assert rad_to_deg(cast(Radians, np.pi)) == pytest.approx(180.0)
    assert rad_to_deg(cast(Radians, np.pi / 2)) == pytest.approx(90.0)
    assert rad_to_deg(cast(Radians, 0.0)) == pytest.approx(0.0)

def test_spherical_to_cartesian():
    """Test conversion from spherical to cartesian coordinates."""
    # Test north pole
    north_pole = to_cartesian(cast(Spherical, (0.0, 0.0)))
    assert north_pole[0] == pytest.approx(0.0)
    assert north_pole[1] == pytest.approx(0.0)
    assert north_pole[2] == pytest.approx(1.0)

    # Test equator at 0 longitude
    equator0 = to_cartesian(cast(Spherical, (0.0, np.pi/2)))
    assert equator0[0] == pytest.approx(1.0)
    assert equator0[1] == pytest.approx(0.0)
    assert equator0[2] == pytest.approx(0.0)

    # Test equator at 90° longitude
    equator90 = to_cartesian(cast(Spherical, (np.pi/2, np.pi/2)))
    assert equator90[0] == pytest.approx(0.0)
    assert equator90[1] == pytest.approx(1.0)
    assert equator90[2] == pytest.approx(0.0)

def test_cartesian_to_spherical():
    """Test conversion from cartesian to spherical coordinates."""
    # Test round trip conversion
    original = cast(Spherical, (np.pi/4, np.pi/6))
    cartesian = to_cartesian(original)
    spherical = to_spherical(cartesian)
    
    assert spherical[0] == pytest.approx(original[0])
    assert spherical[1] == pytest.approx(original[1])

def test_lonlat_to_spherical():
    """Test conversion from longitude/latitude to spherical coordinates."""
    # Test Greenwich equator
    greenwich = from_lonlat(cast(LonLat, (0.0, 0.0)))
    # Match OFFSET_LON: 93
    assert greenwich[0] == pytest.approx(deg_to_rad(cast(Degrees, 93.0)))
    assert greenwich[1] == pytest.approx(np.pi/2)  # 90° colatitude = equator

    # Test north pole
    north_pole = from_lonlat(cast(LonLat, (0.0, 90.0)))
    assert north_pole[1] == pytest.approx(0.0)  # 0° colatitude = north pole

    # Test south pole
    south_pole = from_lonlat(cast(LonLat, (0.0, -90.0)))
    assert south_pole[1] == pytest.approx(np.pi)  # 180° colatitude = south pole

def test_spherical_to_lonlat():
    """Test conversion from spherical to longitude/latitude coordinates."""
    # Test round trip conversion
    test_points = [
        (0.0, 0.0),     # Greenwich equator
        (0.0, 90.0),    # North pole
        (0.0, -90.0),   # South pole
        (180.0, 45.0),  # Date line mid-latitude
        (-90.0, -45.0), # West hemisphere mid-latitude
    ]

    for lon, lat in test_points:
        spherical = from_lonlat(cast(LonLat, (lon, lat)))
        new_lon, new_lat = to_lonlat(spherical)
        
        assert new_lon == pytest.approx(lon)
        assert new_lat == pytest.approx(lat) 