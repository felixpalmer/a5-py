# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

from typing import List, Tuple
from .types import Quaternary, Flip


def reverse_pattern(pattern: List[int]) -> List[int]:
    """Reverse a pattern by finding the index of each position."""
    return [pattern.index(i) for i in range(len(pattern))]


# Patterns used to rearrange the cells when shifting
PATTERN = [0, 1, 3, 4, 5, 6, 7, 2]
PATTERN_FLIPPED = [0, 1, 2, 7, 3, 4, 5, 6]
PATTERN_REVERSED = reverse_pattern(PATTERN)
PATTERN_FLIPPED_REVERSED = reverse_pattern(PATTERN_FLIPPED)


def shift_digits(digits: List[int], i: int, flips: List[int], invert_j: bool, pattern: List[int]) -> None:
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
