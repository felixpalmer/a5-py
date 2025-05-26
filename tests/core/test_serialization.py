# SPDX-License-Identifier: Apache-2.0
import pytest
from a5.core.serialization import (
    serialize,
    deserialize,
    get_resolution,
    MAX_RESOLUTION,
    REMOVAL_MASK,
    FIRST_HILBERT_RESOLUTION,
    cell_to_parent,
    cell_to_children,
)
from a5.core.utils import origins, A5Cell
import json
import random
from pathlib import Path
import numpy as np
from types import SimpleNamespace
import copy

# Equivalent RESOLUTION_MASKS as list of strings
RESOLUTION_MASKS = [
    # Non-Hilbert resolutions
    "0000000000000000000000000000000000000000000000000000000000000000",  # Globe
    "0000001000000000000000000000000000000000000000000000000000000000",  # Dodecahedron faces
    "0000000100000000000000000000000000000000000000000000000000000000",  # Quintants
    # Hilbert resolutions
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
    # Point level - commented out in original code
    # "0000000000000000000000000000000000000000000000000000000000000001",
]


origin0 = copy.deepcopy(origins[0])# Use first origin for most tests

TEST_IDS_PATH = Path(__file__).parent / "test-ids.json"

with open(TEST_IDS_PATH) as f:
    TEST_IDS = json.load(f)

def test_number_of_masks():
    assert len(RESOLUTION_MASKS) == MAX_RESOLUTION # TODO add point level


def test_removal_mask():
    origin_segment_bits = "0" * 6
    remaining_bits = "1" * 58
    expected = int(f"0b{origin_segment_bits}{remaining_bits}", 2)
    assert REMOVAL_MASK == expected

def test_removal_mask_is_correct():
    origin_segment_bits = '0' * 6
    remaining_bits = '1' * 58
    expected = int(f'0b{origin_segment_bits}{remaining_bits}', 2)
    
    # Example value for REMOVAL_MASK; replace with your actual constant
    REMOVAL_MASK = int(f'0b{origin_segment_bits}{remaining_bits}', 2)
    
    assert REMOVAL_MASK == expected

@pytest.mark.parametrize("i", range(len(RESOLUTION_MASKS)))
def test_encodes_resolution_correctly(i):
    input_cell = A5Cell(origin = origin0,segment = 4, S = 0, resolution= i)

    serialized = serialize(input_cell)
    binary_str = bin(serialized)[2:].zfill(64)
    assert binary_str == RESOLUTION_MASKS[i]

@pytest.mark.parametrize("i,mask", list(enumerate(RESOLUTION_MASKS)))
def test_get_resolution(i, mask):
    N = int(f"0b{mask}", 2)
    assert get_resolution(N) == i


def test_encode_fields():
    cell = A5Cell(origin=origin0, segment=4, S=0, resolution=30)
    assert serialize(cell) == 0b10


def test_serialize_too_large_S():
    cell = A5Cell(origin = origin0,segment = 0,S = 16,resolution= 4) # Too large for resolution 2 (max is 15)
    
    with pytest.raises(ValueError, match="S \\(16\\) is too large for resolution level 4"):
        serialize(cell)


def test_serialize_too_large_resolution():
    cell = A5Cell(origin = origin0,segment = 0,S = 0,resolution= 32) # MAX_RESOLUTION is 31   
    with pytest.raises(ValueError, match="Resolution \\(32\\) is too large"):
        serialize(cell)


@pytest.mark.parametrize("binary", RESOLUTION_MASKS[FIRST_HILBERT_RESOLUTION:])
@pytest.mark.parametrize("n", range(1, 12))
def test_round_trip_hilbert_origins(n, binary):
    originSegmentId = bin(5 * n)[2:].zfill(6)
    serialized = int(f"0b{originSegmentId}{binary[6:]}", 2)
    deserialized = deserialize(serialized)
    reserialized = serialize(deserialized)
    assert reserialized == serialized


@pytest.mark.parametrize("id", TEST_IDS)
def test_round_trip_ids(id):
    serialized = int(id, 16)
    deserialized = deserialize(serialized)
    reserialized = serialize(deserialized)
    assert reserialized == serialized

@pytest.mark.parametrize("id", TEST_IDS)
def test_cell_hierarchy(id):
    assert isinstance(id, str), f"id is not a string: {id} (type={type(id)})"
    cell = int(id, 16)

    children = cell_to_children(cell)
    assert children, "No children returned"

    parent = cell_to_parent(children[0])
    assert parent == cell, "Parent of first child does not match"

    parents = [cell_to_parent(c) for c in children]
    assert all(p == cell for p in parents), "Not all children map to the same parent"


def test_non_hilbert_to_non_hilbert():
    # Test resolution 1 to 2 (both non-Hilbert)
    cell = serialize(A5Cell(origin = origin0,segment = 0,S = 0,resolution= 1))        
    children = cell_to_children(cell)
    assert len(children) == 5
    for child in children:
        assert cell_to_parent(child) == cell


def test_non_hilbert_to_hilbert():
    # Test resolution 2 to 3 (non-Hilbert to Hilbert)
    cell = serialize(A5Cell(origin = origin0,segment = 0,S = 0,resolution= 2))            
    children = cell_to_children(cell)
    assert len(children) == 4
    for child in children:
        assert cell_to_parent(child) == cell


def test_hilbert_to_non_hilbert():
    # Test resolution 3 to 2 (Hilbert to non-Hilbert)
    cell = serialize(A5Cell(origin = origin0,segment = 0,S = 0,resolution= 3))            
    parent = cell_to_parent(cell)
    children = cell_to_children(parent)
    assert cell in children
    assert len(children) == 4


# def test_hierarchy_chain():
#     # Test a chain of resolutions from 0 to 4
#     resolutions = [0, 1, 2, 3, 4]
#     cells = [
#         serialize(A5Cell(origin = origin0,segment = 0,S = 0,resolution= res))
#         for res in resolutions
#     ]
#     # Test parent relationships
#     for i in range(1, len(cells)):
#         assert cell_to_parent(cells[i]) == cells[i - 1]
    
#     # Test children relationships 
#     for i in range(len(cells) - 1):
#         children = cell_to_children(cells[i])
#         assert cells[i + 1] in children
        
# def test_base_cell_division_counts():
#     # Start with the base cell at resolution 0
#     base_cell = serialize(A5Cell(origin = origin0,segment = 0,S = 0,resolution= 0))
#     current_cells = [base_cell]
#     expected_counts = [1, 12, 60, 240, 960]  # 1, 12, 12*5, 12*5*4, 12*5*4*4

#     # Test each resolution level up to 4
#     for resolution in range(4):  
#         all_children = []
#         # Get all children of current cells
#         for cell in current_cells:
#             all_children.extend(cell_to_children(cell))
        
#         # Verify the total number of cells matches expected
#         assert len(all_children) == expected_counts[resolution + 1]
#         # Update current cells for next iteration
#         current_cells = all_children

def test_low_resolution_hierarchy_chain():
    resolutions = [0, 1, 2, 3, 4]
    cells = [
        serialize(A5Cell(origin=origin0, segment=0, S=0, resolution=res))
        for res in resolutions
    ]

    # Test parent relationships
    for i in range(1, len(cells)):
        parent = cell_to_parent(cells[i])
        assert parent == cells[i - 1]

    # Test children relationships
    for i in range(len(cells) - 1):
        children = cell_to_children(cells[i])
        assert cells[i + 1] in children

def test_base_cell_division_counts():
    base_cell = serialize(A5Cell(origin=origin0, segment=0, S=0, resolution=0))
    current_cells = [base_cell]
    expected_counts = [1, 12, 60, 240, 960]  # 1, 12, 12*5, 12*5*4, 12*5*4*4

    for resolution in range(0, 4):
        all_children = [
            child for cell in current_cells for child in cell_to_children(cell)
        ]
        assert len(all_children) == expected_counts[resolution + 1]
        current_cells = all_children

def random_id():
    origin_segment = format(random.randint(0, 59), '06b')
    resolution = random.randint(0, MAX_RESOLUTION - 2)
    S = random.randint(0, (1 << (2 * resolution)) - 1)
    S_bits = format(S, f'0{2 * resolution}b')
    id_bits = f'{origin_segment}{S_bits}10'.ljust(64, '0')
    return hex(int(id_bits, 2))[2:].rjust(16, '0')

# Example usage:
for _ in range(100):
    print(f'"{random_id()}",')
