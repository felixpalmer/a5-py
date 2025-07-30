"""
Tests for cell-info functionality.
"""

import json
import pytest
from pathlib import Path
from a5.core.cell_info import get_num_cells, cell_area

# Load test fixtures  
FIXTURES_DIR = Path(__file__).parent / "fixtures"
with open(FIXTURES_DIR / "cell-info.json") as f:
    CELL_INFO_FIXTURES = json.load(f)

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