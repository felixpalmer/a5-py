# SPDX-License-Identifier: Apache-2.0
import pytest
from a5.core.serialization import (
    serialize,
    deserialize,
    get_resolution,
    MAX_RESOLUTION,
    FIRST_HILBERT_RESOLUTION,
    WORLD_CELL,
    cell_to_parent,
    cell_to_children,
    get_res0_cells,
    is_first_child,
    get_stride,
)
from a5.core.utils import A5Cell
from a5.core.origin import origins
import json
from pathlib import Path
import copy

# Test data
origin0 = copy.deepcopy(origins[0])
FIXTURES_PATH = Path(__file__).parent / "fixtures" / "serialization.json"

with open(FIXTURES_PATH) as f:
    FIXTURES = json.load(f)

RESOLUTION_MASKS = FIXTURES["resolutionMasks"]
TEST_IDS = FIXTURES["testIds"]


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
    assert len(RESOLUTION_MASKS) == MAX_RESOLUTION + 1


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
@pytest.mark.parametrize("binary", RESOLUTION_MASKS[FIRST_HILBERT_RESOLUTION:MAX_RESOLUTION])
def test_round_trip_resolution_masks_with_origins(origin_id, binary):
    """Test round trip for resolution masks with different origins."""
    origin_segment_id = format(5 * origin_id, '06b')
    serialized = int(f"0b{origin_segment_id}{binary[6:]}", 2)
    deserialized = deserialize(serialized)
    reserialized = serialize(deserialized)
    assert reserialized == serialized


@pytest.mark.parametrize("resolution", range(MAX_RESOLUTION + 1))
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
    resolution = get_resolution(cell)
    # Skip res 30 (no children) and res 29 with out-of-bounds quintants
    # (res 30 children fall back to res 29)
    if resolution >= MAX_RESOLUTION:
        return
    child = cell_to_children(cell)[0]
    if get_resolution(child) != resolution + 1:
        return

    parent = cell_to_parent(child)
    assert parent == cell

    children = cell_to_children(cell)
    parents = [cell_to_parent(c) for c in children]
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


# =============================================================================
# resolution 30 tests
# =============================================================================

def test_res30_get_resolution_detects_from_lsb():
    """Any odd int (LSB=1) that isn't 0 is resolution 30."""
    assert get_resolution(1) == 30
    assert get_resolution(3) == 30
    assert get_resolution(0xFFFFFFFFFFFFFFFF) == 30


def test_res30_round_trip_valid_quintants():
    """Serialize/deserialize round trip for valid quintants (0-41)."""
    for q in range(42):
        origin_id = q // 5
        origin = origins[origin_id]
        segment_n = q % 5
        segment = (segment_n + origin.first_quintant) % 5

        cell = A5Cell(origin=origin, segment=segment, S=0, resolution=30)
        serialized = serialize(cell)
        assert get_resolution(serialized) == 30

        # Verify correct marker pattern
        if q <= 31:
            assert serialized & 1 == 1  # ...1 encoding
        elif q <= 39:
            assert serialized & 0b111 == 0b100  # ...100 encoding
        else:
            assert serialized & 0b11111 == 0b10000  # ...10000 encoding

        deserialized = deserialize(serialized)
        assert deserialized["origin"].id == origin_id
        assert deserialized["segment"] == segment
        assert deserialized["S"] == 0
        assert deserialized["resolution"] == 30

        # Round trip
        reserialized = serialize(deserialized)
        assert reserialized == serialized


def test_res30_round_trip_nonzero_s():
    """Serialize/deserialize round trip with non-zero S."""
    origin = origins[0]
    segment = (0 + origin.first_quintant) % 5

    test_s_values = [0, 1, 42, (1 << 58) - 1]
    for s in test_s_values:
        cell = A5Cell(origin=origin, segment=segment, S=s, resolution=30)
        serialized = serialize(cell)
        deserialized = deserialize(serialized)
        assert deserialized["S"] == s
        assert deserialized["resolution"] == 30
        assert serialize(deserialized) == serialized


def test_res30_bit_layout_1_encoding():
    """Bit layout: ...1 encoding (quintant 0-31)."""
    origin = origins[0]
    segment = (0 + origin.first_quintant) % 5

    # Quintant 0, S=0 -> just the marker bit
    cell0 = serialize(A5Cell(origin=origin, segment=segment, S=0, resolution=30))
    assert cell0 == 1

    # Quintant 0, S=1 -> marker + S shifted left by 1
    cell1 = serialize(A5Cell(origin=origin, segment=segment, S=1, resolution=30))
    assert cell1 == 0b11


def test_res30_bit_layout_10000_encoding():
    """Bit layout: ...10000 encoding (quintant 40-41)."""
    origin = origins[8]
    segment = (0 + origin.first_quintant) % 5

    # Quintant 40, S=0 -> just the marker
    cell0 = serialize(A5Cell(origin=origin, segment=segment, S=0, resolution=30))
    assert cell0 == 0b10000

    # Quintant 40, S=1 -> S shifted left by 5 + marker
    cell1 = serialize(A5Cell(origin=origin, segment=segment, S=1, resolution=30))
    assert cell1 == 0b110000


def test_res30_bit_layout_100_encoding():
    """Bit layout: ...100 encoding (quintant 32-39)."""
    # Origin 6 has quintants 30-34, segmentN=2 gives quintant 32
    origin = origins[6]
    segment_n = 2
    segment = (segment_n + origin.first_quintant) % 5

    # Quintant 32, S=0 -> just the marker
    cell0 = serialize(A5Cell(origin=origin, segment=segment, S=0, resolution=30))
    assert cell0 == 0b100

    # Quintant 32, S=1 -> S shifted left by 3 + marker
    cell1 = serialize(A5Cell(origin=origin, segment=segment, S=1, resolution=30))
    assert cell1 == 0b1100


def test_res30_round_trip_nonzero_s_100_encoding():
    """Round trip with non-zero S (...100 encoding)."""
    # Use quintant 35 (origin 7, segmentN=0)
    origin = origins[7]
    segment = (0 + origin.first_quintant) % 5

    test_s_values = [0, 1, 42, (1 << 58) - 1]
    for s in test_s_values:
        cell = A5Cell(origin=origin, segment=segment, S=s, resolution=30)
        serialized = serialize(cell)
        assert serialized & 0b111 == 0b100  # ...100 marker
        deserialized = deserialize(serialized)
        assert deserialized["S"] == s
        assert deserialized["resolution"] == 30
        assert serialize(deserialized) == serialized


def test_res30_round_trip_nonzero_s_10000_encoding():
    """Round trip with non-zero S (...10000 encoding)."""
    # Use quintant 40 (origin 8, segmentN=0)
    origin = origins[8]
    segment = (0 + origin.first_quintant) % 5

    test_s_values = [0, 1, 42, (1 << 58) - 1]
    for s in test_s_values:
        cell = A5Cell(origin=origin, segment=segment, S=s, resolution=30)
        serialized = serialize(cell)
        assert serialized & 0b11111 == 0b10000  # ...10000 marker
        deserialized = deserialize(serialized)
        assert deserialized["S"] == s
        assert deserialized["resolution"] == 30
        assert serialize(deserialized) == serialized


def test_res30_falls_back_to_res29_for_quintant_gt_41():
    """Falls back to res 29 for quintant > 41."""
    # Origin 9 has quintants 45-49, all > 41
    origin = origins[9]
    segment = (0 + origin.first_quintant) % 5
    cell = serialize(A5Cell(origin=origin, segment=segment, S=0, resolution=30))
    assert get_resolution(cell) == 29

    # With non-zero S, the parent S should be S >> 2
    cell2 = serialize(A5Cell(origin=origin, segment=segment, S=7, resolution=30))
    assert get_resolution(cell2) == 29
    assert deserialize(cell2)["S"] == 1  # 7 >> 2 = 1


def test_res30_falls_back_for_out_of_bounds_quintant_55():
    """Falls back to res 29 for out-of-bounds quintant (e.g. 55)."""
    # Origin 11 has quintants 55-59, all > 41
    origin = origins[11]
    segment_n = 0
    segment = (segment_n + origin.first_quintant) % 5
    cell = serialize(A5Cell(origin=origin, segment=segment, S=100, resolution=30))
    assert get_resolution(cell) == 29
    assert deserialize(cell)["S"] == 25  # 100 >> 2 = 25
    assert deserialize(cell)["origin"].id == 11


def test_res30_throws_for_s_too_large():
    """Throws for S too large."""
    origin = origins[0]
    segment = (0 + origin.first_quintant) % 5
    with pytest.raises(ValueError, match="too large for resolution level 30"):
        serialize(A5Cell(origin=origin, segment=segment, S=1 << 58, resolution=30))


def test_res30_cell_to_parent():
    """cellToParent from res 30 to res 29."""
    origin = origins[0]
    segment = (0 + origin.first_quintant) % 5

    for i in range(4):
        child = serialize(A5Cell(origin=origin, segment=segment, S=i, resolution=30))
        parent = cell_to_parent(child)
        assert get_resolution(parent) == 29
        assert deserialize(parent)["S"] == 0


def test_res30_cell_to_children():
    """cellToChildren from res 29 to res 30."""
    origin = origins[0]
    segment = (0 + origin.first_quintant) % 5
    parent = serialize(A5Cell(origin=origin, segment=segment, S=0, resolution=29))
    children = cell_to_children(parent, 30)

    assert len(children) == 4
    for i, child in enumerate(children):
        assert get_resolution(child) == 30
        assert deserialize(child)["S"] == i


def test_res30_children_parent_round_trip():
    """cellToChildren/cellToParent round trip."""
    origin = origins[0]
    segment = (0 + origin.first_quintant) % 5
    parent = serialize(A5Cell(origin=origin, segment=segment, S=42, resolution=29))
    children = cell_to_children(parent, 30)

    assert len(children) == 4
    for child in children:
        assert cell_to_parent(child) == parent


def test_res30_get_stride():
    """getStride returns 2 for res 30."""
    assert get_stride(30) == 2


def test_res30_is_first_child_1_encoding():
    """isFirstChild works for res 30 (...1 encoding)."""
    origin = origins[0]
    segment = (0 + origin.first_quintant) % 5

    assert is_first_child(serialize(A5Cell(origin=origin, segment=segment, S=0, resolution=30))) is True
    assert is_first_child(serialize(A5Cell(origin=origin, segment=segment, S=1, resolution=30))) is False
    assert is_first_child(serialize(A5Cell(origin=origin, segment=segment, S=4, resolution=30))) is True


def test_res30_is_first_child_100_encoding():
    """isFirstChild works for res 30 (...100 encoding)."""
    origin = origins[7]  # quintant 35, uses ...100
    segment = (0 + origin.first_quintant) % 5

    assert is_first_child(serialize(A5Cell(origin=origin, segment=segment, S=0, resolution=30))) is True
    assert is_first_child(serialize(A5Cell(origin=origin, segment=segment, S=1, resolution=30))) is False
    assert is_first_child(serialize(A5Cell(origin=origin, segment=segment, S=4, resolution=30))) is True


def test_res30_is_first_child_10000_encoding():
    """isFirstChild works for res 30 (...10000 encoding)."""
    origin = origins[8]  # quintant 40, uses ...10000
    segment = (0 + origin.first_quintant) % 5

    assert is_first_child(serialize(A5Cell(origin=origin, segment=segment, S=0, resolution=30))) is True
    assert is_first_child(serialize(A5Cell(origin=origin, segment=segment, S=1, resolution=30))) is False
    assert is_first_child(serialize(A5Cell(origin=origin, segment=segment, S=4, resolution=30))) is True


def test_res30_children_parent_round_trip_10000_encoding():
    """cellToChildren/cellToParent round trip (...10000 encoding)."""
    origin = origins[8]
    segment = (0 + origin.first_quintant) % 5
    parent = serialize(A5Cell(origin=origin, segment=segment, S=10, resolution=29))
    children = cell_to_children(parent, 30)

    assert len(children) == 4
    for child in children:
        assert get_resolution(child) == 30
        assert child & 0b11111 == 0b10000  # ...10000 marker
        assert cell_to_parent(child) == parent


def test_res30_children_parent_round_trip_100_encoding():
    """cellToChildren/cellToParent round trip (...100 encoding)."""
    origin = origins[7]
    segment = (0 + origin.first_quintant) % 5
    parent = serialize(A5Cell(origin=origin, segment=segment, S=10, resolution=29))
    children = cell_to_children(parent, 30)

    assert len(children) == 4
    for child in children:
        assert get_resolution(child) == 30
        assert child & 0b111 == 0b100  # ...100 marker
        assert cell_to_parent(child) == parent


def test_res30_cell_to_children_throws_at_max():
    """cellToChildren of res 30 throws (max resolution)."""
    origin = origins[0]
    segment = (0 + origin.first_quintant) % 5
    cell = serialize(A5Cell(origin=origin, segment=segment, S=0, resolution=30))
    with pytest.raises(ValueError, match="exceeds maximum resolution"):
        cell_to_children(cell)


# =============================================================================
# resolution 30 location tests
# =============================================================================

def test_res30_locations_round_trip():
    """Round trip for res 30 location cells."""
    res30_locations = FIXTURES["res30Locations"]
    for loc in res30_locations:
        cell = int(loc["hex"], 16)
        deserialized = deserialize(cell)
        reserialized = serialize(deserialized)
        assert reserialized == cell


def test_res30_locations_out_of_bounds_fall_back():
    """Out-of-bounds quintants fall back to res 29."""
    res30_locations = FIXTURES["res30Locations"]
    out_of_bounds = [l for l in res30_locations if l["resolution"] == 29]
    assert len(out_of_bounds) > 0
    for loc in out_of_bounds:
        cell = int(loc["hex"], 16)
        assert get_resolution(cell) == 29


def test_res30_locations_in_bounds_encode_at_res30():
    """In-bounds quintants encode at res 30."""
    res30_locations = FIXTURES["res30Locations"]
    in_bounds = [l for l in res30_locations if l["resolution"] == 30]
    assert len(in_bounds) > 0
    for loc in in_bounds:
        cell = int(loc["hex"], 16)
        assert get_resolution(cell) == 30
