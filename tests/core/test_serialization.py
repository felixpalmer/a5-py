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
import numpy as np
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


# Helper function to compare A5Cell objects (handles numpy arrays)
def assert_cells_equal(actual: A5Cell, expected: A5Cell):
    """Compare A5Cell objects, handling numpy arrays properly."""
    assert actual["origin"].id == expected["origin"].id
    assert actual["origin"].axis == expected["origin"].axis
    assert np.array_equal(actual["origin"].quat, expected["origin"].quat)
    assert actual["origin"].angle == expected["origin"].angle
    assert actual["origin"].orientation == expected["origin"].orientation
    assert actual["origin"].first_quintant == expected["origin"].first_quintant
    assert actual["segment"] == expected["segment"]
    assert actual["S"] == expected["S"]
    assert actual["resolution"] == expected["resolution"]


# =============================================================================
# Basic Serialization Tests
# =============================================================================

def test_number_of_masks():
    """Verify we have the expected number of resolution masks."""
    assert len(RESOLUTION_MASKS) == MAX_RESOLUTION


def test_removal_mask():
    """Test the removal mask constant is correct."""
    origin_segment_bits = "0" * 6
    remaining_bits = "1" * 58
    expected = int(f"0b{origin_segment_bits}{remaining_bits}", 2)
    assert REMOVAL_MASK == expected


@pytest.mark.parametrize("resolution", range(MAX_RESOLUTION))
def test_serialize_deserialize_round_trip(resolution):
    """Test that serialize/deserialize round trip works for all resolutions."""
    input_cell = A5Cell(origin=origin0, segment=4, S=0, resolution=resolution)
    
    serialized = serialize(input_cell)
    deserialized = deserialize(serialized)
    
    # At resolution 0, segment is always normalized to 0
    expected_cell = copy.deepcopy(input_cell)
    if resolution == 0:
        expected_cell["segment"] = 0
    
    assert_cells_equal(deserialized, expected_cell)
    assert get_resolution(serialized) == resolution


@pytest.mark.parametrize("id", TEST_IDS)
def test_round_trip_test_ids(id):
    """Test round trip serialization for known test IDs."""
    serialized = int(id, 16)
    deserialized = deserialize(serialized)
    reserialized = serialize(deserialized)
    assert reserialized == serialized


@pytest.mark.parametrize("binary", RESOLUTION_MASKS[FIRST_HILBERT_RESOLUTION:])
@pytest.mark.parametrize("origin_id", range(1, 12))
def test_round_trip_hilbert_origins(origin_id, binary):
    """Test round trip for Hilbert resolutions with different origins."""
    origin_segment_id = bin(5 * origin_id)[2:].zfill(6)
    serialized = int(f"0b{origin_segment_id}{binary[6:]}", 2)
    deserialized = deserialize(serialized)
    reserialized = serialize(deserialized)
    assert reserialized == serialized


# =============================================================================
# Error Handling Tests
# =============================================================================

def test_serialize_s_too_large():
    """Test error when S is too large for resolution."""
    cell = A5Cell(origin=origin0, segment=0, S=16, resolution=3)  # max S for res3 is 15
    with pytest.raises(ValueError, match="S \\(16\\) is too large for resolution level"):
        serialize(cell)


def test_serialize_resolution_too_large():
    """Test error when resolution exceeds maximum."""
    cell = A5Cell(origin=origin0, segment=0, S=0, resolution=MAX_RESOLUTION + 1)
    with pytest.raises(ValueError, match="Resolution .* is too large"):
        serialize(cell)


def test_serialize_max_resolution():
    """Test serializing at maximum resolution works."""
    cell = A5Cell(origin=origin0, segment=4, S=0, resolution=MAX_RESOLUTION - 1)
    serialized = serialize(cell)
    assert get_resolution(serialized) == MAX_RESOLUTION - 1


# =============================================================================
# Hierarchy Tests
# =============================================================================

@pytest.mark.parametrize("id", TEST_IDS)
def test_parent_child_relationships(id):
    """Test that parent-child relationships are consistent."""
    cell = int(id, 16)
    
    children = cell_to_children(cell)
    assert children, "No children returned"
    
    # All children should have the same parent
    parents = [cell_to_parent(child) for child in children]
    assert all(p == cell for p in parents), "Not all children map to the same parent"


def test_resolution_transitions():
    """Test specific resolution transitions."""
    # res0 -> res1 (non-Hilbert to non-Hilbert)
    cell_res0 = serialize(A5Cell(origin=origin0, segment=0, S=0, resolution=0))
    children_res1 = cell_to_children(cell_res0)
    assert len(children_res1) == 5
    
    # res1 -> res2 (non-Hilbert to Hilbert)
    cell_res1 = serialize(A5Cell(origin=origin0, segment=0, S=0, resolution=1))
    children_res2 = cell_to_children(cell_res1)
    assert len(children_res2) == 4
    
    # res2 -> res3 (Hilbert to Hilbert)
    cell_res2 = serialize(A5Cell(origin=origin0, segment=0, S=0, resolution=2))
    children_res3 = cell_to_children(cell_res2)
    assert len(children_res3) == 4


def test_hierarchy_chain():
    """Test a chain of parent-child relationships."""
    cells = [
        serialize(A5Cell(origin=origin0, segment=0, S=0, resolution=res))
        for res in range(5)
    ]
    
    # Test parent relationships
    for i in range(1, len(cells)):
        parent = cell_to_parent(cells[i])
        assert parent == cells[i - 1]
    
    # Test children relationships
    for i in range(len(cells) - 1):
        children = cell_to_children(cells[i])
        assert cells[i + 1] in children


def test_world_cell_division():
    """Test division from world cell through multiple resolution levels."""
    base_cell = serialize(A5Cell(origin=origin0, segment=0, S=0, resolution=-1))
    current_cells = [base_cell]
    expected_counts = [12, 60, 240, 960]  # 12, 12*5, 12*5*4, 12*5*4*4
    
    for resolution, expected_count in enumerate(expected_counts):
        all_children = []
        for cell in current_cells:
            all_children.extend(cell_to_children(cell))
        
        assert len(all_children) == expected_count
        current_cells = all_children


# =============================================================================
# Special Functions Tests
# =============================================================================

def test_get_res0_cells():
    """Test that get_res0_cells returns 12 resolution 0 cells."""
    res0_cells = get_res0_cells()
    assert len(res0_cells) == 12
    
    for cell in res0_cells:
        assert get_resolution(cell) == 0


def test_world_cell_parent():
    """Test that resolution 0 cells have world cell as parent."""
    cell_res0 = serialize(A5Cell(origin=origin0, segment=0, S=0, resolution=0))
    parent = cell_to_parent(cell_res0)
    assert get_resolution(parent) == -1
    assert parent == WORLD_CELL
