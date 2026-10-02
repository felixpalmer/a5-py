# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

import math
from .constants import AUTHALIC_AREA_EARTH

AUTHALIC_AREA = AUTHALIC_AREA_EARTH
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


# Mean cell edge length divided by sqrt(cell_area), measured exhaustively from the
# cell boundaries. Resolution 0 cells (dodecahedron faces) and resolution 1 cells
# (triangular quintants) have their own geometry; from resolution 2 the pentagonal
# tiling refines self-similarly and the ratio converges to ~0.8211, so a constant
# serves all higher resolutions.
EDGE_LENGTH_RATIOS = [0.7131, 1.4818, 0.8164, 0.8198, 0.8208, 0.821]
EDGE_LENGTH_RATIO = 0.8211


def cell_edge_length_avg(resolution: int) -> float:
    """
    Returns the average edge length of a cell at a given resolution in meters.
    Individual edge lengths vary from the average by roughly ±10%, depending
    on the cell's shape and its position on the globe.

    Args:
        resolution: The resolution level

    Returns:
        Average edge length of a cell in meters
    """
    if resolution < 0:
        resolution = 0
    ratio = EDGE_LENGTH_RATIOS[resolution] if resolution < len(EDGE_LENGTH_RATIOS) else EDGE_LENGTH_RATIO
    return ratio * math.sqrt(cell_area(resolution))
