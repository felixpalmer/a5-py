import pytest
import math
from a5.core.hilbert import (
    quaternary_to_kj,
    quaternary_to_flips,
    YES,
    NO,
    Anchor,
    ij_to_kj,
    kj_to_ij,
    get_required_digits,
    ij_to_s,
    s_to_anchor,
)

def test_hilbert_anchor_base_cases():
    # Test first corner (0)
    offset0 = quaternary_to_kj(0, (NO, NO))
    assert offset0 == (0, 0)
    flips0 = quaternary_to_flips(0)
    assert flips0 == (NO, NO)

    # Test second corner (1)
    offset1 = quaternary_to_kj(1, (NO, NO))
    assert offset1 == (1, 0)
    flips1 = quaternary_to_flips(1)
    assert flips1 == (NO, YES)

    # Test third corner (2)
    offset2 = quaternary_to_kj(2, (NO, NO))
    assert offset2 == (1, 1)
    flips2 = quaternary_to_flips(2)
    assert flips2 == (NO, NO)

    # Test fourth corner (3)
    offset3 = quaternary_to_kj(3, (NO, NO))
    assert offset3 == (2, 1)
    flips3 = quaternary_to_flips(3)
    assert flips3 == (YES, NO)

def test_hilbert_anchor_respects_flips():
    # Test with x-flip
    offset_x = quaternary_to_kj(1, (YES, NO))
    assert offset_x == (0, -1)

    # Test with y-flip
    offset_y = quaternary_to_kj(1, (NO, YES))
    assert offset_y == (0, 1)

    # Test with both flips
    offset_xy = quaternary_to_kj(1, (YES, YES))
    assert offset_xy == (-1, 0)

def test_output_flips_depend_only_on_input():
    EXPECTED_FLIPS = [
        (NO, NO),
        (NO, YES),
        (NO, NO),
        (YES, NO)
    ]
    for n in range(4):
        flips = quaternary_to_flips(n)
        assert flips == EXPECTED_FLIPS[n]

def test_generates_correct_sequence():
    # Test first few indices
    anchor0 = s_to_anchor(0, 1, 'uv')
    assert tuple(anchor0.offset) == (0, 0)
    assert anchor0.flips == (NO, NO)

    anchor1 = s_to_anchor(1, 1, 'uv')
    assert anchor1.flips[1] == YES

    anchor4 = s_to_anchor(4, 1, 'uv')
    assert math.sqrt(anchor4.offset[0]**2 + anchor4.offset[1]**2) > 1  # Should be scaled up

    # Test that sequence length grows exponentially
    anchors = [s_to_anchor(i, 1, 'uv') for i in range(16)]
    unique_offsets = {tuple(a.offset) for a in anchors}
    assert len(unique_offsets) == 12
    unique_anchors = {(tuple(a.offset), a.flips) for a in anchors}
    assert len(unique_anchors) == 15

def test_neighboring_anchors_are_adjacent():
    # Test that combining anchors preserves orientation rules
    anchor1 = s_to_anchor(0, 1, 'uv')
    anchor2 = s_to_anchor(1, 1, 'uv')
    anchor3 = s_to_anchor(2, 1, 'uv')
    
    # Check that relative positions make sense
    diff = (anchor2.offset[0] - anchor1.offset[0], anchor2.offset[1] - anchor1.offset[1])
    assert math.sqrt(diff[0]**2 + diff[1]**2) == 1  # Should be adjacent
    diff2 = (anchor3.offset[0] - anchor2.offset[0], anchor3.offset[1] - anchor2.offset[1])
    assert math.sqrt(diff2[0]**2 + diff2[1]**2) == math.sqrt(2)  # Should be adjacent

def test_generates_correct_anchors():
    EXPECTED_ANCHORS = [
        {'s': 0, 'offset': [0, 0], 'flips': (NO, NO)},
        {'s': 9, 'offset': [3, 1], 'flips': (YES, YES)},
        {'s': 16, 'offset': [2, 2], 'flips': (NO, NO)},
        {'s': 17, 'offset': [3, 2], 'flips': (NO, YES)},
        {'s': 31, 'offset': [1, 3], 'flips': (YES, NO)},
        {'s': 77, 'offset': [7, 5], 'flips': (NO, NO)},
        {'s': 100, 'offset': [3, 7], 'flips': (YES, YES)},
        {'s': 101, 'offset': [2, 7], 'flips': (YES, NO)},
        {'s': 170, 'offset': [10, 1], 'flips': (NO, NO)},
        {'s': 411, 'offset': [7, 13], 'flips': (YES, NO)},
        {'s': 1762, 'offset': [7, 31], 'flips': (YES, NO)},
        {'s': 481952, 'offset': [96, 356], 'flips': (YES, YES)},
        {'s': 192885192, 'offset': [13183, 1043], 'flips': (NO, NO)},
        {'s': 4719283155, 'offset': [37190, 46076], 'flips': (NO, YES)},
        {'s': 7123456789, 'offset': [29822, 40293], 'flips': (NO, YES)},
    ]

    for test_case in EXPECTED_ANCHORS:
        anchor = s_to_anchor(test_case['s'], 20, 'uv')
        assert list(anchor.offset) == test_case['offset']
        assert anchor.flips == test_case['flips']

def test_ij_to_kj_conversion():
    # Test some basic conversions
    test_cases = [
        ([0, 0], [0, 0]),    # Origin
        ([1, 0], [1, 0]),    # Unit i
        ([0, 1], [1, 1]),    # Unit j -> k=i+j=1, j=1
        ([1, 1], [2, 1]),    # i + j -> k=2, j=1
        ([2, 3], [5, 3])     # 2i + 3j -> k=5, j=3
    ]

    for input_ij, expected_kj in test_cases:
        result = ij_to_kj(input_ij)
        assert list(result) == expected_kj

def test_kj_to_ij_conversion():
    # Test some basic conversions
    test_cases = [
        ([0, 0], [0, 0]),     # Origin
        ([1, 0], [1, 0]),     # Pure k -> i=1, j=0
        ([1, 1], [0, 1]),     # k=1, j=1 -> i=0, j=1
        ([2, 1], [1, 1]),     # k=2, j=1 -> i=1, j=1
        ([5, 3], [2, 3])      # k=5, j=3 -> i=2, j=3
    ]

    for input_kj, expected_ij in test_cases:
        result = kj_to_ij(input_kj)
        assert list(result) == expected_ij

def test_ij_kj_inverse():
    # Test that converting back and forth gives the original coordinates
    test_points = [
        [0, 0],
        [1, 0],
        [0, 1],
        [1, 1],
        [2, 3],
        [-1, 2],
        [3, -2]
    ]

    for point in test_points:
        kj = ij_to_kj(point)
        ij = kj_to_ij(kj)
        assert list(ij) == point

def test_get_required_digits():
    test_cases = [
        ([0, 0], 1),
        ([1, 0], 1),
        ([2, 1], 2),
        ([4, 0], 3),
        ([8, 8], 5),
        ([16, 0], 5),
        ([32, 32], 7)
    ]

    for offset, expected in test_cases:
        assert get_required_digits(offset) == expected

def test_required_digits_matches_s_to_anchor():
    # Test that get_required_digits matches the number of digits 
    # actually used in sToAnchor's output
    test_values = [0, 1, 2, 3, 4, 9, 16, 17, 31, 77, 100]
    
    for s in test_values:
        anchor = s_to_anchor(s, 20, 'uv')
        required_digits = get_required_digits(anchor.offset)
        # Convert to base 4 manually
        temp_s = s
        digits = 0
        while temp_s > 0:
            digits += 1
            temp_s //= 4
        actual_digits = digits if digits > 0 else 1
        assert required_digits >= actual_digits
        assert required_digits <= actual_digits + 1

def test_ij_to_s():
    test_values = [
        # First quadrant
        {'s': 0, 'offset': [0, 0]},
        {'s': 0, 'offset': [0.999, 0]},
        {'s': 1, 'offset': [0.6, 0.6]},
        {'s': 7, 'offset': [0.000001, 1.1]},
        {'s': 2, 'offset': [1.2, 0.5]},
        {'s': 2, 'offset': [1.9999, 0]},

        # Recursive cases, 2nd quadrant, flipY
        {'s': 3, 'offset': [1.9999, 0.001]},
        {'s': 4, 'offset': [1.1, 1.1]},
        {'s': 5, 'offset': [1.999, 1.999]},
        {'s': 6, 'offset': [0.99, 1.99]},

        # 3rd quadrant, no flips
        {'s': 28, 'offset': [0.999, 2.000001]},
        {'s': 29, 'offset': [0.9, 2.5]},
        {'s': 30, 'offset': [0.5, 3.1]},
        {'s': 31, 'offset': [1.3, 2.5]},

        # 4th quadrant, flipX
        {'s': 8, 'offset': [2.00001, 1.001]},
        {'s': 9, 'offset': [2.8, 0.5]},
        {'s': 10, 'offset': [2.00001, 0.5]},
        {'s': 11, 'offset': [3.5, 0.2]},

        # Next level
        {'s': 15, 'offset': [2.5, 1.5]},
        {'s': 21, 'offset': [3.999, 3.999]},

        # Both flips
        {'s': 24, 'offset': [1.999, 3.999]},
        {'s': 25, 'offset': [1.2, 3.5]},
        {'s': 26, 'offset': [1.9, 2.2]},
        {'s': 27, 'offset': [0.1, 3.9]}
    ]

    for test_case in test_values:
        result = ij_to_s(test_case['offset'], 3, 'uv')
        assert result == test_case['s']

@pytest.mark.parametrize('orientation', ['uv', 'vu', 'uw', 'wu', 'vw', 'wv'])
@pytest.mark.parametrize('s', [0, 1, 2, 3, 4, 9, 16, 17, 31, 77, 100, 101, 170, 411, 1762, 4410, 12387, 41872])
def test_ij_to_s_inverse_of_s_to_anchor(s, orientation):
    resolution = 20
    anchor = s_to_anchor(s, resolution, orientation)

    # Nudge the offset away from the edge of the triangle
    flip_x, flip_y = anchor.flips
    if flip_x == NO and flip_y == NO:
        anchor.offset = [anchor.offset[0] + 0.1, anchor.offset[1] + 0.1]
    elif flip_x == YES and flip_y == NO:
        anchor.offset = [anchor.offset[0] + 0.1, anchor.offset[1] - 0.2]
    elif flip_x == NO and flip_y == YES:
        anchor.offset = [anchor.offset[0] - 0.1, anchor.offset[1] + 0.2]
    elif flip_x == YES and flip_y == YES:
        anchor.offset = [anchor.offset[0] - 0.1, anchor.offset[1] - 0.1]

    assert ij_to_s(anchor.offset, resolution, orientation) == s 