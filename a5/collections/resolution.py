# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

from typing import Sequence

from ..core.compaction_marker import compaction_marker_resolution, is_compaction_marker
from ..core.serialization import RES30_TAG_BITS, RESOLUTION_TAGS, get_resolution, MAX_RESOLUTION


def covering_resolution(cells: Sequence[int]) -> int:
    """
    The resolution of a set of cells: the resolution of its compaction marker, or of
    its finest cell when it has none. A covering stands for all its
    cells at this resolution. Returns -1 for an empty set (or the world cell).

    Args:
        cells: Cells, as returned by `compact`, `polygon_to_cells` etc.

    Returns:
        Resolution (-1 to 30)
    """
    # A covering ends in its compaction marker, which records the resolution
    if len(cells) > 0 and is_compaction_marker(cells[-1]):
        return compaction_marker_resolution(cells[-1])

    # Otherwise the finest cell: finer cells have smaller resolution tags (the
    # lowest set bit), except at res 30, whose tags are recognised separately
    finest_tag = 0
    for cell in cells:
        if is_compaction_marker(cell):
            resolution = compaction_marker_resolution(cell)
            if resolution == MAX_RESOLUTION:
                return MAX_RESOLUTION
            tag = RESOLUTION_TAGS[resolution]
        else:
            tag = cell & -cell
            if tag & RES30_TAG_BITS:
                return MAX_RESOLUTION
        if tag != 0 and (finest_tag == 0 or tag < finest_tag):
            finest_tag = tag
    return -1 if finest_tag == 0 else get_resolution(finest_tag)
