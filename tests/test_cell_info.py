"""
Tests for cell-info functionality.
"""

import json
import math
import pytest
from pathlib import Path
from a5.core.cell_info import get_num_cells, cell_area, cell_edge_length_avg
from a5.core.cell import cell_to_boundary
from a5.core.serialization import get_resolution
from a5.core.hex import hex_to_u64
from a5.core.constants import AUTHALIC_RADIUS_EARTH

# Load test fixtures
FIXTURES_DIR = Path(__file__).parent / "fixtures"
with open(FIXTURES_DIR / "cell-info.json") as f:
    CELL_INFO_FIXTURES = json.load(f)

with open(Path(__file__).parent / "core" / "fixtures" / "serialization.json") as f:
    SERIALIZATION_FIXTURES = json.load(f)

def test_get_num_cells_returns_correct_count_for_all_resolutions():
    """Test that getNumCells returns correct number of cells for all resolutions."""
    for fixture in CELL_INFO_FIXTURES["numCells"]:
        result = get_num_cells(fixture["resolution"])
        expected = int(fixture["countBigInt"])  # Use the exact BigInt value
        assert result == expected, f"Resolution {fixture['resolution']}: got {result}, expected {expected}"

def test_cell_area_returns_correct_area_for_all_resolutions():
    """Test that cellArea returns correct area for all resolutions."""
    for fixture in CELL_INFO_FIXTURES["cellArea"]:
        assert cell_area(fixture["resolution"]) == fixture["areaM2"]

def test_cell_edge_length_avg_returns_correct_length_for_all_resolutions():
    """Test that cellEdgeLengthAvg returns correct edge length for all resolutions."""
    for fixture in CELL_INFO_FIXTURES["cellEdgeLengthAvg"]:
        assert cell_edge_length_avg(fixture["resolution"]) == pytest.approx(fixture["lengthM"], rel=1e-12)

def _geodesic(a, b):
    """Geodesic distance between two (lon, lat) points on the authalic sphere, in meters."""
    lat1 = math.radians(a[1])
    lat2 = math.radians(b[1])
    h = (
        math.sin((lat2 - lat1) / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(math.radians(b[0] - a[0]) / 2) ** 2
    )
    return 2 * AUTHALIC_RADIUS_EARTH * math.asin(math.sqrt(h))

def test_every_boundary_edge_of_test_cells_is_within_10_percent_of_average():
    """Every boundary edge of the test cells is within ±10% of the average."""
    # Sample each edge with multiple segments to measure its true curved length
    SEGMENTS = 10
    for hex_id in SERIALIZATION_FIXTURES["testIds"]:
        cell = hex_to_u64(hex_id)
        resolution = get_resolution(cell)
        avg = cell_edge_length_avg(resolution)
        boundary = cell_to_boundary(cell, {"closed_ring": True, "segments": SEGMENTS})
        num_edges = (len(boundary) - 1) // SEGMENTS
        for e in range(num_edges):
            length = 0.0
            for i in range(SEGMENTS):
                idx = e * SEGMENTS + i
                length += _geodesic(boundary[idx], boundary[idx + 1])
            ratio = length / avg
            assert 0.9 < ratio < 1.1, f"cell {hex_id} edge {e}: ratio {ratio}"