# SPDX-License-Identifier: Apache-2.0
import pytest
from a5.core.serialization import (
    serialize,
    deserialize,
    get_resolution,
    MAX_RESOLUTION,
    REMOVAL_MASK,
    FIRST_HILBERT_RESOLUTION,
    WORLD_CELL,
    cell_to_parent,
    cell_to_children,
    get_res0_cells,
)
from a5.core.utils import A5Cell
from a5.core.origin import origins
import json
from pathlib import Path
import copy

# Test data
origin0 = copy.deepcopy(origins[0])
TEST_IDS_PATH = Path(__file__).parent / "test-ids.json"

with open(TEST_IDS_PATH) as f:
    TEST_IDS = json.load(f)

# Resolution masks for testing bit encoding
RESOLUTION_MASKS = [
    # Non-Hilbert resolutions
    "0000001000000000000000000000000000000000000000000000000000000000",  # res0: Dodecahedron faces
    "0000000100000000000000000000000000000000000000000000000000000000",  # res1: Quintants
    # Hilbert resolutions (res2-29)
    "0000000010000000000000000000000000000000000000000000000000000000",
    "0000000000100000000000000000000000000000000000000000000000000000",
    "0000000000001000000000000000000000000000000000000000000000000000",
    "0000000000000010000000000000000000000000000000000000000000000000",
    "0000000000000000100000000000000000000000000000000000000000000000",
    "0000000000000000001000000000000000000000000000000000000000000000",
    "0000000000000000000010000000000000000000000000000000000000000000",
    "0000000000000000000000100000000000000000000000000000000000000000",
    "0000000000000000000000001000000000000000000000000000000000000000",
    "0000000000000000000000000010000000000000000000000000000000000000",
    "0000000000000000000000000000100000000000000000000000000000000000",
    "0000000000000000000000000000001000000000000000000000000000000000",
    "0000000000000000000000000000000010000000000000000000000000000000",
    "0000000000000000000000000000000000100000000000000000000000000000",
    "0000000000000000000000000000000000001000000000000000000000000000",
    "0000000000000000000000000000000000000010000000000000000000000000",
    "0000000000000000000000000000000000000000100000000000000000000000",
    "0000000000000000000000000000000000000000001000000000000000000000",
    "0000000000000000000000000000000000000000000010000000000000000000",
    "0000000000000000000000000000000000000000000000100000000000000000",
    "0000000000000000000000000000000000000000000000001000000000000000",
    "0000000000000000000000000000000000000000000000000010000000000000",
    "0000000000000000000000000000000000000000000000000000100000000000",
    "0000000000000000000000000000000000000000000000000000001000000000",
    "0000000000000000000000000000000000000000000000000000000010000000",
    "0000000000000000000000000000000000000000000000000000000000100000",
    "0000000000000000000000000000000000000000000000000000000000001000",
    "0000000000000000000000000000000000000000000000000000000000000010",
]


# Helper function to compare A5Cell objects
def assert_cells_equal(actual: A5Cell, expected: A5Cell):
    """Compare A5Cell objects properly."""
    assert actual["origin"].id == expected["origin"].id
    assert actual["origin"].axis == expected["origin"].axis
    assert actual["origin"].quat == expected["origin"].quat
    assert actual["origin"].angle == expected["origin"].angle
    assert actual["origin"].orientation == expected["origin"].orientation
    assert actual["origin"].first_quintant == expected["origin"].first_quintant
    assert actual["segment"] == expected["segment"]
    assert actual["S"] == expected["S"]
    assert actual["resolution"] == expected["resolution"]


# =============================================================================
# serialize tests
# =============================================================================

def test_correct_number_of_masks():
    """Test correct number of masks."""
    assert len(RESOLUTION_MASKS) == MAX_RESOLUTION


def test_removal_mask_is_correct():
    """Test removal mask is correct."""
    origin_segment_bits = "0" * 6
    remaining_bits = "1" * 58
    expected = int(f"0b{origin_segment_bits}{remaining_bits}", 2)
    assert REMOVAL_MASK == expected


def test_encodes_resolution_correctly_for_different_values():
    """Test resolution encoding."""
    for i in range(len(RESOLUTION_MASKS)):
        # Origin 0 has first quintant 4, so use segment 4 to obtain start of Hilbert curve
        input_cell = A5Cell(origin=origin0, segment=4, S=0, resolution=i)
        serialized = serialize(input_cell)
        actual_binary = format(serialized, '064b')
        expected_binary = RESOLUTION_MASKS[i]
        assert actual_binary == expected_binary


def test_correctly_extracts_resolution():
    """Test resolution extraction."""
    for i, binary in enumerate(RESOLUTION_MASKS):
        assert len(binary) == 64
        n = int(f"0b{binary}", 2)
        resolution = get_resolution(n)
        assert resolution == i


def test_encodes_origin_segment_and_s_correctly():
    """Test origin, segment and S encoding."""
    # Origin 0 has first quintant 4, so use segment 4 to obtain start of Hilbert curve
    cell = A5Cell(origin=origin0, segment=4, S=0, resolution=MAX_RESOLUTION - 1)
    serialized = serialize(cell)
    assert serialized == 0b10  # 2 in decimal


def test_throws_error_when_s_too_large_for_resolution():
    """Test S too large error."""
    cell = A5Cell(origin=origin0, segment=0, S=16, resolution=3)  # Too large for resolution 3 (max is 15)
    with pytest.raises(ValueError, match="S \\(16\\) is too large for resolution level 3"):
        serialize(cell)


def test_throws_error_when_resolution_exceeds_maximum():
    """Test resolution exceeds maximum error."""
    cell = A5Cell(origin=origin0, segment=0, S=0, resolution=MAX_RESOLUTION + 1)
    with pytest.raises(ValueError, match="Resolution .* is too large"):
        serialize(cell)


# Round trip tests
@pytest.mark.parametrize("id", TEST_IDS)
def test_round_trip_test_ids(id):
    """Test round trip with test IDs."""
    serialized = int(id, 16)
    deserialized = deserialize(serialized)
    reserialized = serialize(deserialized)
    assert reserialized == serialized


@pytest.mark.parametrize("origin_id", range(1, 12))
@pytest.mark.parametrize("binary", RESOLUTION_MASKS[FIRST_HILBERT_RESOLUTION:])
def test_round_trip_resolution_masks_with_origins(origin_id, binary):
    """Test round trip for resolution masks with different origins."""
    origin_segment_id = format(5 * origin_id, '06b')
    serialized = int(f"0b{origin_segment_id}{binary[6:]}", 2)
    deserialized = deserialize(serialized)
    reserialized = serialize(deserialized)
    assert reserialized == serialized


@pytest.mark.parametrize("resolution", range(MAX_RESOLUTION))
def test_serialize_deserialize_round_trip(resolution):
    """Test serialize/deserialize round trip for all resolutions."""
    input_cell = A5Cell(origin=origin0, segment=4, S=0, resolution=resolution)
    
    serialized = serialize(input_cell)
    deserialized = deserialize(serialized)
    
    # At resolution 0, segment is always normalized to 0
    expected_cell = copy.deepcopy(input_cell)
    if resolution == 0:
        expected_cell["segment"] = 0
    
    assert_cells_equal(deserialized, expected_cell)
    assert get_resolution(serialized) == resolution


# =============================================================================
# hierarchy tests
# =============================================================================

@pytest.mark.parametrize("id", TEST_IDS)
def test_cell_to_children_with_same_resolution_returns_original_cell(id):
    cell = int(id, 16)
    current_resolution = get_resolution(cell)
    children = cell_to_children(cell, current_resolution)
    assert len(children) == 1
    assert children[0] == cell

@pytest.mark.parametrize("id", TEST_IDS)
def test_cell_to_parent_with_same_resolution_returns_original_cell(id):
    cell = int(id, 16)
    current_resolution = get_resolution(cell)
    parent = cell_to_parent(cell, current_resolution)
    assert parent == cell

@pytest.mark.parametrize("id", TEST_IDS)
def test_round_trip_between_cell_to_parent_and_cell_to_children(id):
    """Test parent-child round trip."""
    cell = int(id, 16)
    
    children = cell_to_children(cell)
    assert children, "No children returned"
    
    # Test first child
    child = children[0]
    assert cell_to_parent(child) == cell
    
    # Test all children have same parent
    parents = [cell_to_parent(child) for child in children]
    assert all(p == cell for p in parents), "Not all children map to the same parent"


def test_non_hilbert_to_non_hilbert_hierarchy():
    """Test non-Hilbert to non-Hilbert transition."""
    # Test resolution 0 to 1 (both non-Hilbert)
    cell = serialize(A5Cell(origin=origin0, segment=0, S=0, resolution=0))
    children = cell_to_children(cell)
    assert len(children) == 5
    for child in children:
        assert cell_to_parent(child) == cell


def test_non_hilbert_to_hilbert_hierarchy():
    """Test non-Hilbert to Hilbert transition."""
    # Test resolution 1 to 2 (non-Hilbert to Hilbert)
    cell = serialize(A5Cell(origin=origin0, segment=0, S=0, resolution=1))
    children = cell_to_children(cell)
    assert len(children) == 4
    for child in children:
        assert cell_to_parent(child) == cell


def test_hilbert_to_non_hilbert_hierarchy():
    """Test Hilbert to non-Hilbert transition."""
    # Test resolution 2 to 1 (Hilbert to non-Hilbert)
    cell = serialize(A5Cell(origin=origin0, segment=0, S=0, resolution=2))
    parent = cell_to_parent(cell, 1)
    children = cell_to_children(parent)
    assert len(children) == 4
    assert cell in children


def test_low_resolution_hierarchy_chain():
    """Test hierarchy chain."""
    # Test a chain of resolutions from 0 to 4
    resolutions = [0, 1, 2, 3, 4]
    cells = [
        serialize(A5Cell(origin=origin0, segment=0, S=0, resolution=res))
        for res in resolutions
    ]

    # Test parent relationships
    for i in range(1, len(cells)):
        assert cell_to_parent(cells[i]) == cells[i-1]

    # Test children relationships
    for i in range(len(cells) - 1):
        children = cell_to_children(cells[i])
        assert cells[i+1] in children


def test_base_cell_division_counts():
    """Test base cell division."""
    # Start with the base cell (resolution -1)
    base_cell = serialize(A5Cell(origin=origin0, segment=0, S=0, resolution=-1))
    current_cells = [base_cell]
    expected_counts = [12, 60, 240, 960]  # 12, 12*5, 12*5*4, 12*5*4*4

    # Test each resolution level up to 4
    for resolution, expected_count in enumerate(expected_counts):
        # Get all children of current cells
        all_children = []
        for cell in current_cells:
            all_children.extend(cell_to_children(cell))
        
        # Verify the total number of cells matches expected
        assert len(all_children) == expected_count
        
        # Update current cells for next iteration
        current_cells = all_children


# =============================================================================
# getRes0Cells tests
# =============================================================================

def test_get_res0_cells_returns_12_resolution_0_cells():
    """Test getRes0Cells functionality."""
    res0_cells = get_res0_cells()
    assert len(res0_cells) == 12
    
    # Each cell should have resolution 0
    for cell in res0_cells:
        assert get_resolution(cell) == 0
