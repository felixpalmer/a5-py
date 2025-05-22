# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

import numpy as np
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

# Anchor offset is specified in ij units, the eigenbasis of the Hilbert curve
# Define k as the vector i + j, as it means vectors u & v are of unit length
def ij_to_kj(ij: IJ) -> KJ:
    """Convert from IJ coordinates to KJ coordinates."""
    i, j = ij
    return np.array([i + j, j], dtype=np.float64)

def kj_to_ij(kj: KJ) -> IJ:
    """Convert from KJ coordinates to IJ coordinates."""
    k, j = kj
    return np.array([k - j, j], dtype=np.float64)

#  Orientation of the Hilbert curve. The curve fills a space defined by the triangle with vertices
#  u, v & w. The orientation describes which corner the curve starts and ends at, e.g. wv is a
#  curve that starts at w and ends at v.
Orientation = Literal['uv', 'vu', 'uw', 'wu', 'vw', 'wv']

# Using KJ allows simplification of definitions
k_pos = np.array([1.0, 0.0])  # k
j_pos = np.array([0.0, 1.0])  # j
k_neg = -k_pos
j_neg = -j_pos
ZERO = np.array([0.0, 0.0])

def quaternary_to_kj(n: Quaternary, flips: Tuple[Flip, Flip]) -> KJ:
    """Indirection to allow for flips"""
    flip_x, flip_y = flips
    p = ZERO.copy()
    q = ZERO.copy()
    
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
        return ZERO.copy()
    elif n == 1:
        return p.copy()
    elif n == 2:
        return p + q
    elif n == 3:
        return q + 2 * p
    else:
        raise ValueError(f"Invalid Quaternary value: {n}")

def quaternary_to_flips(n: Quaternary) -> Tuple[Flip, Flip]:
    """Convert quaternary number to flip configuration."""
    flips = [(NO, NO), (NO, YES), (NO, NO), (YES, NO)]
    return flips[n]

FLIP_SHIFT = np.array([-1, 1])

def s_to_anchor(s: Union[int, str], resolution: int, orientation: Orientation) -> Anchor:
    """Convert s-value to anchor with orientation."""
    input_val = int(s)
    reverse = orientation in ('vu', 'wu', 'vw')
    invert_j = orientation in ('wv', 'vw')
    flip_ij = orientation in ('wu', 'uw')
    
    if reverse:
        input_val = (1 << (2 * resolution)) - input_val - 1
        
    anchor = _s_to_anchor(input_val)
    
    if flip_ij:
        i, j = anchor.offset
        anchor.offset = np.array([j, i])
        
        # Compensate for origin shift
        if anchor.flips[0] == YES:
            anchor.offset += FLIP_SHIFT
        if anchor.flips[1] == YES:
            anchor.offset -= FLIP_SHIFT
            
    if invert_j:
        i, j = anchor.offset
        new_j = (1 << resolution) - (i + j)
        anchor.flips = (-anchor.flips[0], anchor.flips[1])
        anchor.offset[1] = new_j
        
    return anchor

def _s_to_anchor(s: int) -> Anchor:
    """Internal function to convert s-value to anchor."""
    k = s % 4
    offset = np.zeros(2)
    flips = [NO, NO]
    
    # Get quaternary digits
    digits = []
    while s > 0:
        digits.append(s % 4)
        s >>= 2
        
    # Process digits from left to right (most significant first)
    for digit in reversed(digits):
        # Scale up existing anchor
        offset *= 2
        
        # Get child anchor and combine with current anchor
        child_offset = quaternary_to_kj(digit, tuple(flips))
        offset += child_offset
        
        # Update flips
        new_flips = quaternary_to_flips(digit)
        flips[0] *= new_flips[0]
        flips[1] *= new_flips[1]
        
    return Anchor(k, kj_to_ij(offset), tuple(flips))

# Get the number of digits needed to represent the offset
# As we don't know the flips we need to add 2 to include the next row
def get_required_digits(offset: np.ndarray) -> int:
    """Calculate required number of digits to represent the offset."""
    index_sum = np.ceil(offset[0]) + np.ceil(offset[1])
    if index_sum == 0:
        return 1
    return 1 + int(np.floor(np.log2(index_sum)))

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
    
    ij = input_ij.copy()
    if flip_ij:
        ij[0], ij[1] = ij[1], ij[0]
    if invert_j:
        i, j = ij
        ij[1] = (1 << resolution) - (i + j)
        
    s = _ij_to_s(ij)
    if reverse:
        s = (1 << (2 * resolution)) - s - 1
        
    return s

def _ij_to_s(input_ij: IJ) -> int:
    # Get number of digits we need to process
    num_digits = get_required_digits(input_ij)
    digits = [0] * num_digits
    
    flips = [NO, NO]
    pivot = np.zeros(2)
    
    # Process digits from left to right (most significant first)
    for i in range(num_digits):
        relative_offset = input_ij - pivot
        scale = 1 << (num_digits - 1 - i)
        scaled_offset = relative_offset / scale
        
        digit = ij_to_quaternary(scaled_offset, tuple(flips))
        digits[i] = digit
        
        # Update running state
        child_offset = kj_to_ij(quaternary_to_kj(digit, tuple(flips)))
        upscaled_child_offset = child_offset * scale
        pivot += upscaled_child_offset
        
        new_flips = quaternary_to_flips(digit)
        flips[0] *= new_flips[0]
        flips[1] *= new_flips[1]
        
    # Convert digits to s-value
    output = 0
    for i, digit in enumerate(digits):
        scale = 1 << (2 * (num_digits - 1 - i))
        output += digit * scale
        
    return output