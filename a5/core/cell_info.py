# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

import math

AUTHALIC_RADIUS = 6371007.2  # m
AUTHALIC_AREA = 4 * math.pi * AUTHALIC_RADIUS * AUTHALIC_RADIUS  # m^2
FIRST_HILBERT_RESOLUTION = 2

def get_num_cells(resolution: int) -> int:
    """
    Returns the number of cells at a given resolution.

    Args:
        resolution: The resolution level

    Returns:
        Number of cells at the given resolution
    """
    if resolution < 0:
        return 0
    if resolution == 0:
        return 12
    return 60 * (4 ** (resolution - 1))


def get_num_children(parent_resolution: int, child_resolution: int) -> int:
    """
    Returns the number of children between two resolutions.

    Args:
        parent_resolution: The parent resolution level
        child_resolution: The child resolution level

    Returns:
        Number of children
    """
    if child_resolution < parent_resolution:
        return 0
    if child_resolution == parent_resolution:
        return 1
    if parent_resolution >= FIRST_HILBERT_RESOLUTION:
        # Between levels of constant aperture of 4, relation simplifies
        return 4 ** (child_resolution - parent_resolution)

    parent_count = get_num_cells(parent_resolution) or 1
    child_count = get_num_cells(child_resolution)
    return child_count // parent_count


def cell_area(resolution: int) -> float:
    """
    Returns the area of a cell at a given resolution in square meters.

    Args:
        resolution: The resolution level

    Returns:
        Area of a cell in square meters
    """
    if resolution < 0:
        return AUTHALIC_AREA
    return AUTHALIC_AREA / get_num_cells(resolution)
