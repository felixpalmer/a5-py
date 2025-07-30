# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

# import numpy as np  # Removed
import math
from typing import Tuple, List, Union, Literal, Final
from .coordinate_systems import IJ, KJ

# Type aliases
Quaternary = Literal[0, 1, 2, 3]
YES: Literal[-1] = -1
NO: Literal[1] = 1
Flip = Literal[-1, 1]

class Anchor:
    def __init__(self, k: Quaternary, offset: IJ, flips: Tuple[Flip, Flip]):
        self.k = k
        self.offset = offset
        self.flips = flips

def reverse_pattern(pattern: List[int]) -> List[int]:
    """Reverse a pattern by finding the index of each position."""
    return [pattern.index(i) for i in range(len(pattern))]

# Patterns used to rearrange the cells when shifting
PATTERN = [0, 1, 3, 4, 5, 6, 7, 2]
PATTERN_FLIPPED = [0, 1, 2, 7, 3, 4, 5, 6]
PATTERN_REVERSED = reverse_pattern(PATTERN)
PATTERN_FLIPPED_REVERSED = reverse_pattern(PATTERN_FLIPPED)

# Anchor offset is specified in ij units, the eigenbasis of the Hilbert curve
# Define k as the vector i + j, as it means vectors u & v are of unit length
def ij_to_kj(ij: IJ) -> KJ:
    """Convert from IJ coordinates to KJ coordinates."""
    i, j = ij
    return (i + j, j)

def kj_to_ij(kj: KJ) -> IJ:
    """Convert from KJ coordinates to IJ coordinates."""
    k, j = kj
    return (k - j, j)

#  Orientation of the Hilbert curve. The curve fills a space defined by the triangle with vertices
#  u, v & w. The orientation describes which corner the curve starts and ends at, e.g. wv is a
#  curve that starts at w and ends at v.
Orientation = Literal['uv', 'vu', 'uw', 'wu', 'vw', 'wv']

# Using KJ allows simplification of definitions
k_pos = (1.0, 0.0)  # k
j_pos = (0.0, 1.0)  # j
k_neg = (-k_pos[0], -k_pos[1])
j_neg = (-j_pos[0], -j_pos[1])
ZERO = (0.0, 0.0)

def quaternary_to_kj(n: Quaternary, flips: Tuple[Flip, Flip]) -> KJ:
    """Indirection to allow for flips"""
    flip_x, flip_y = flips
    p = ZERO
    q = ZERO
    
    if flip_x == NO and flip_y == NO:
        p = k_pos
        q = j_pos
    elif flip_x == YES and flip_y == NO:
        # Swap and negate
        p = j_neg
        q = k_neg
    elif flip_x == NO and flip_y == YES:
        # Swap only
        p = j_pos
        q = k_pos
    elif flip_x == YES and flip_y == YES:
        # Negate only
        p = k_neg
        q = j_neg

    if n == 0:
        return ZERO
    elif n == 1:
        return p
    elif n == 2:
        return (p[0] + q[0], p[1] + q[1])
    elif n == 3:
        return (q[0] + 2 * p[0], q[1] + 2 * p[1])
    else:
        raise ValueError(f"Invalid Quaternary value: {n}")

def quaternary_to_flips(n: Quaternary) -> Tuple[Flip, Flip]:
    """Convert quaternary number to flip configuration."""
    flips = [(NO, NO), (NO, YES), (NO, NO), (YES, NO)]
    return flips[n]

FLIP_SHIFT = (-1, 1)

def _shift_digits(digits: List[Quaternary], i: int, flips: List[Flip], invert_j: bool, pattern: List[int]) -> None:
    """Shift digits based on pattern to adjust cell layout."""
    if i <= 0:
        return

    parent_k = digits[i] if i < len(digits) else 0
    child_k = digits[i - 1]
    F = flips[0] + flips[1]

    # Detect when cells need to be shifted
    needs_shift = True
    first = True

    # The value of F which cells need to be shifted
    # The rule is flipped depending on the orientation, specifically on the value of invert_j
    if invert_j != (F == 0):
        needs_shift = parent_k in (1, 2)  # Second & third pentagons only
        first = parent_k == 1  # Second pentagon is first
    else:
        needs_shift = parent_k < 2  # First two pentagons only
        first = parent_k == 0  # First pentagon is first

    if not needs_shift:
        return

    # Apply the pattern by setting the digits based on the value provided
    src = child_k if first else child_k + 4
    dst = pattern[src]
    digits[i - 1] = dst % 4
    digits[i] = (parent_k + 4 + (dst // 4) - (src // 4)) % 4

def s_to_anchor(s: Union[int, str], resolution: int, orientation: Orientation) -> Anchor:
    """Convert s-value to anchor with orientation."""
    input_val = int(s)
    reverse = orientation in ('vu', 'wu', 'vw')
    invert_j = orientation in ('wv', 'vw')
    flip_ij = orientation in ('wu', 'uw')
    
    if reverse:
        input_val = (1 << (2 * resolution)) - input_val - 1
        
    anchor = _s_to_anchor(input_val, resolution, invert_j, flip_ij)
    
    if flip_ij:
        i, j = anchor.offset
        anchor.offset = (j, i)
        
        # Compensate for origin shift
        if anchor.flips[0] == YES:
            anchor.offset = (anchor.offset[0] + FLIP_SHIFT[0], anchor.offset[1] + FLIP_SHIFT[1])
        if anchor.flips[1] == YES:
            anchor.offset = (anchor.offset[0] - FLIP_SHIFT[0], anchor.offset[1] - FLIP_SHIFT[1])
            
    if invert_j:
        i, j = anchor.offset
        new_j = (1 << resolution) - (i + j)
        anchor.flips = (-anchor.flips[0], anchor.flips[1])
        anchor.offset = (anchor.offset[0], new_j)
        
    return anchor

def _s_to_anchor(s: int, resolution: int, invert_j: bool, flip_ij: bool) -> Anchor:
    """Internal function to convert s-value to anchor."""
    offset = [0.0, 0.0]
    flips = [NO, NO]
    
    # Get quaternary digits
    digits = []
    while s > 0 or len(digits) < resolution:
        digits.append(s % 4)
        s >>= 2
    
    # Pad with zeros if needed
    while len(digits) < resolution:
        digits.append(0)
        
    pattern = PATTERN_FLIPPED if flip_ij else PATTERN

    # Process digits from left to right (most significant first)
    for i in range(len(digits) - 1, -1, -1):
        _shift_digits(digits, i, flips, invert_j, pattern)
        new_flips = quaternary_to_flips(digits[i])
        flips[0] *= new_flips[0]
        flips[1] *= new_flips[1]

    flips = [NO, NO]  # Reset flips for the next loop
    for i in range(len(digits) - 1, -1, -1):
        # Scale up existing anchor
        offset = [offset[0] * 2, offset[1] * 2]
        
        # Get child anchor and combine with current anchor
        child_offset = quaternary_to_kj(digits[i], tuple(flips))
        offset = [offset[0] + child_offset[0], offset[1] + child_offset[1]]
        
        new_flips = quaternary_to_flips(digits[i])
        flips[0] *= new_flips[0]
        flips[1] *= new_flips[1]
        
    k = digits[0] if digits else 0

    return Anchor(k, kj_to_ij(tuple(offset)), tuple(flips))

# Get the number of digits needed to represent the offset
# As we don't know the flips we need to add 2 to include the next row
def get_required_digits(offset: Tuple[float, float]) -> int:
    """Calculate required number of digits to represent the offset."""
    index_sum = math.ceil(offset[0]) + math.ceil(offset[1])
    if index_sum == 0:
        return 1
    return 1 + int(math.floor(math.log2(index_sum)))

# This function uses the ij basis, unlike its inverse!
def ij_to_quaternary(ij: IJ, flips: Tuple[Flip, Flip]) -> Quaternary:
    """Convert IJ coordinates to quaternary number with flips."""
    u, v = ij
    digit = 0
    
    # Boundaries to compare against
    a = -(u + v) if flips[0] == YES else u + v
    b = -u if flips[1] == YES else u
    c = -v if flips[0] == YES else v
    
    # Only one flip
    if flips[0] + flips[1] == 0:
        if c < 1:
            digit = 0
        elif b > 1:
            digit = 3
        elif a > 1:
            digit = 2
        else:
            digit = 1
    # No flips or both
    else:
        if a < 1:
            digit = 0
        elif b > 1:
            digit = 3
        elif c > 1:
            digit = 2
        else:
            digit = 1
            
    return digit

def ij_to_s(input_ij: IJ, resolution: int, orientation: str = 'uv') -> int:
    """Convert IJ coordinates to s-value with orientation."""
    reverse = orientation in ('vu', 'wu', 'vw')
    invert_j = orientation in ('wv', 'vw')
    flip_ij = orientation in ('wu', 'uw')
    
    # Convert tuple to list for modification, then back to tuple
    ij = list(input_ij)
    if flip_ij:
        ij[0], ij[1] = ij[1], ij[0]
    if invert_j:
        i, j = ij
        ij[1] = (1 << resolution) - (i + j)
    
    ij = tuple(ij)  # Convert back to tuple
        
    s = _ij_to_s(ij, invert_j, flip_ij, resolution)
    if reverse:
        s = (1 << (2 * resolution)) - s - 1
        
    return s

def _ij_to_s(input_ij: IJ, invert_j: bool, flip_ij: bool, resolution: int) -> int:
    """Internal function to convert IJ coordinates to s-value."""
    # Get number of digits we need to process
    num_digits = resolution
    digits = [0] * num_digits
    
    flips = [NO, NO]
    pivot = [0.0, 0.0]
    
    # Process digits from left to right (most significant first)
    for i in range(num_digits - 1, -1, -1):
        relative_offset = (input_ij[0] - pivot[0], input_ij[1] - pivot[1])
        scale = 1 << i
        scaled_offset = (relative_offset[0] / scale, relative_offset[1] / scale)
        
        digit = ij_to_quaternary(scaled_offset, tuple(flips))
        digits[i] = digit
        
        # Update running state
        child_offset = kj_to_ij(quaternary_to_kj(digit, tuple(flips)))
        upscaled_child_offset = (child_offset[0] * scale, child_offset[1] * scale)
        pivot = [pivot[0] + upscaled_child_offset[0], pivot[1] + upscaled_child_offset[1]]
        
        new_flips = quaternary_to_flips(digit)
        flips[0] *= new_flips[0]
        flips[1] *= new_flips[1]
        
    pattern = PATTERN_FLIPPED_REVERSED if flip_ij else PATTERN_REVERSED

    for i in range(num_digits):
        new_flips = quaternary_to_flips(digits[i])
        flips[0] *= new_flips[0]
        flips[1] *= new_flips[1]
        _shift_digits(digits, i, flips, invert_j, pattern)
        
    # Convert digits to s-value
    output = 0
    for i in range(num_digits - 1, -1, -1):
        scale = 1 << (2 * i)
        output += digits[i] * scale
        
    return output