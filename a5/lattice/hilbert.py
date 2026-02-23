# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

import math
from typing import Tuple, List
from ..core.coordinate_systems import IJ, KJ
from .types import Anchor, Quaternary, Flip, Orientation, YES, NO
from .basis import kj_to_ij
from .quaternary import quaternary_to_kj, quaternary_to_flips, ij_to_quaternary
from .shift_digits import shift_digits, PATTERN, PATTERN_FLIPPED, PATTERN_REVERSED, PATTERN_FLIPPED_REVERSED

FLIP_SHIFT = (-1, 1)

SHIFTDIGITS = True


def s_to_anchor(s: int, resolution: int, orientation: Orientation, do_shift_digits: bool = SHIFTDIGITS) -> Anchor:
    """Convert s-value to anchor with orientation."""
    input_val = int(s)
    reverse = orientation in ('vu', 'wu', 'vw')
    invert_j = orientation in ('wv', 'vw')
    flip_ij = orientation in ('wu', 'uw')

    if reverse:
        input_val = (1 << (2 * resolution)) - input_val - 1

    anchor = _s_to_anchor(input_val, resolution, invert_j, flip_ij, do_shift_digits)

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


def _s_to_anchor(s: int, resolution: int, invert_j: bool, flip_ij: bool, do_shift_digits: bool = SHIFTDIGITS) -> Anchor:
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
        if do_shift_digits:
            shift_digits(digits, i, flips, invert_j, pattern)
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

    q = digits[0] if digits else 0

    return Anchor(q, kj_to_ij(tuple(offset)), tuple(flips))


def ij_to_s(input_ij: IJ, resolution: int, orientation: Orientation = 'uv', do_shift_digits: bool = SHIFTDIGITS) -> int:
    """Convert IJ coordinates to s-value with orientation."""
    reverse = orientation in ('vu', 'wu', 'vw')
    invert_j = orientation in ('wv', 'vw')
    flip_ij = orientation in ('wu', 'uw')

    ij = list(input_ij)
    if flip_ij:
        ij[0], ij[1] = ij[1], ij[0]
    if invert_j:
        i, j = ij
        ij[1] = (1 << resolution) - (i + j)

    ij = tuple(ij)

    s = _ij_to_s(ij, invert_j, flip_ij, resolution, do_shift_digits)
    if reverse:
        s = (1 << (2 * resolution)) - s - 1

    return s


def _ij_to_s(input_ij: IJ, invert_j: bool, flip_ij: bool, resolution: int, do_shift_digits: bool = SHIFTDIGITS) -> int:
    """Internal function to convert IJ coordinates to s-value."""
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
        if do_shift_digits:
            shift_digits(digits, i, flips, invert_j, pattern)

    # Convert digits to s-value
    output = 0
    for i in range(num_digits - 1, -1, -1):
        scale = 1 << (2 * i)
        output += digits[i] * scale

    return output


def ij_to_flips(input_ij: IJ, resolution: int) -> Tuple[Flip, Flip]:
    """Compute flip states from IJ coordinates."""
    num_digits = resolution

    flips = [NO, NO]
    pivot = [0.0, 0.0]

    # Process digits from left to right (most significant first)
    for i in range(num_digits - 1, -1, -1):
        relative_offset = (input_ij[0] - pivot[0], input_ij[1] - pivot[1])
        scale = 1 << i
        scaled_offset = (relative_offset[0] / scale, relative_offset[1] / scale)

        digit = ij_to_quaternary(scaled_offset, tuple(flips))

        # Update running state
        child_offset = kj_to_ij(quaternary_to_kj(digit, tuple(flips)))
        upscaled_child_offset = (child_offset[0] * scale, child_offset[1] * scale)
        pivot = [pivot[0] + upscaled_child_offset[0], pivot[1] + upscaled_child_offset[1]]

        new_flips = quaternary_to_flips(digit)
        flips[0] *= new_flips[0]
        flips[1] *= new_flips[1]

    return tuple(flips)


# Precomputed probe offsets for anchor_to_s(), indexed by flip combination.
# Index = (1 - flip0) + (1 - flip1) / 2
PROBE_R = 0.1
PROBE_OFFSETS = [
    (PROBE_R * math.cos(45 * math.pi / 180), PROBE_R * math.sin(45 * math.pi / 180)),
    (PROBE_R * math.cos(113 * math.pi / 180), PROBE_R * math.sin(113 * math.pi / 180)),
    (PROBE_R * math.cos(293 * math.pi / 180), PROBE_R * math.sin(293 * math.pi / 180)),
    (PROBE_R * math.cos(225 * math.pi / 180), PROBE_R * math.sin(225 * math.pi / 180)),
]


def anchor_to_s(anchor: Anchor, resolution: int, orientation: Orientation = 'uv') -> int:
    """Convert an anchor to an s-value using a single targeted fractional offset probe."""
    i, j = anchor.offset
    probe_idx = (1 - anchor.flips[0]) + (1 - anchor.flips[1]) // 2
    probe_offset = PROBE_OFFSETS[probe_idx]
    return ij_to_s(
        (i + probe_offset[0], j + probe_offset[1]),
        resolution,
        orientation
    )
