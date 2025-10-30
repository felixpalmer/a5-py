# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

"""
Optimized implementation of compact/uncompact functions for A5 DGGS.

This version uses cell_to_children for expansion and stride-based sibling detection
for compaction.
"""

from typing import List

from .serialization import (
    get_resolution,
    cell_to_children,
    cell_to_parent,
    get_stride,
    is_first_child,
    FIRST_HILBERT_RESOLUTION
)
from .cell_info import get_num_children


def uncompact(cells: List[int], target_resolution: int) -> List[int]:
    """
    Expands a set of A5 cells to a target resolution by generating all descendant cells.

    Args:
        cells: List of A5 cell identifiers to uncompact
        target_resolution: The target resolution level for all output cells

    Returns:
        List of cell identifiers, all at the target resolution
    """
    # First calculate how much space is needed
    n = 0
    resolutions = []
    for cell in cells:
        resolution = get_resolution(cell)
        resolution_diff = target_resolution - resolution
        if resolution_diff < 0:
            raise ValueError(
                f"Cannot uncompact cell at resolution {resolution} to lower resolution {target_resolution}"
            )

        resolutions.append(resolution)
        n += get_num_children(resolution, target_resolution)

    # Write directly into pre-allocated list
    result = [0] * n
    offset = 0
    for i, cell in enumerate(cells):
        resolution = resolutions[i]

        num_children = get_num_children(resolution, target_resolution)
        if num_children == 1:
            result[offset] = cell
        else:
            children = cell_to_children(cell, target_resolution)
            for j, child in enumerate(children):
                result[offset + j] = child

        offset += num_children

    return result


def compact(cells: List[int]) -> List[int]:
    """
    Compacts a set of A5 cells by replacing complete groups of sibling cells with their parent cells.

    Args:
        cells: List of A5 cell identifiers to compact

    Returns:
        List of compacted cell identifiers (typically smaller than input)
    """
    if len(cells) == 0:
        return []

    # Single sort and dedup
    current_cells = sorted(set(cells))

    # Compact until no more changes
    # No re-sorting needed - parents maintain sorted order!
    changed = True
    while changed:
        changed = False
        result = []
        i = 0

        while i < len(current_cells):
            cell = current_cells[i]
            resolution = get_resolution(cell)

            # Can't compact below resolution 0
            if resolution < 0:
                result.append(cell)
                i += 1
                continue

            # Check for complete sibling group using unified stride-based approach
            if resolution >= FIRST_HILBERT_RESOLUTION:
                expected_children = 4  # Hilbert levels have 4 siblings
            elif resolution == 0:
                expected_children = 12  # First two levels are exceptions, with 12 & 5 siblings
            else:
                expected_children = 5

            if i + expected_children <= len(current_cells):
                has_all_siblings = True

                # Use stride-based checking for all resolutions
                # First check if this cell is a first child (at a sibling group boundary)
                if is_first_child(cell, resolution):
                    stride = get_stride(resolution)

                    # Check that all expected siblings are present with correct stride
                    for j in range(1, expected_children):
                        expected_cell = cell + j * stride
                        if current_cells[i + j] != expected_cell:
                            has_all_siblings = False
                            break
                else:
                    # First cell is not at a sibling group boundary
                    has_all_siblings = False

                if has_all_siblings:
                    # Compute parent only once when needed
                    parent = cell_to_parent(cell)
                    result.append(parent)
                    i += expected_children
                    changed = True
                    continue

            result.append(cell)
            i += 1

        current_cells = result

    return current_cells
