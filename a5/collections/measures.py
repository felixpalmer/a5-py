# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

from typing import Dict, Sequence

from ..core.cell_info import cell_area, get_num_children
from ..core.compaction_marker import is_compaction_marker
from ..core.serialization import checked_resolution
from .resolution import covering_resolution


def count(cells: Sequence[int]) -> int:
    """
    The number of cells in a set at its resolution: the length of the
    uncompacted set. This differs from the length of a compacted list. Each
    cell given is counted, so overlapping cells are counted more than once; use
    `union` to merge them first.

    Args:
        cells: Set of cells (compacted or not)

    Returns:
        Number of cells at the set's resolution

    Raises:
        ValueError: If a value is neither an A5 cell ID nor a compaction marker
    """
    resolution = covering_resolution(cells)
    # Children per cell, by cell resolution
    children: Dict[int, int] = {}
    total = 0
    for cell in cells:
        if is_compaction_marker(cell):
            continue
        r = checked_resolution(cell)
        n = children.get(r)
        if n is None:
            n = children[r] = get_num_children(r, resolution)
        total += n
    return total


def area(cells: Sequence[int]) -> float:
    """
    The area of a set of cells, in square meters. Exact, as A5 cells are
    equal-area. Overlapping cells each add their area; use `union` to merge them
    first.

    Args:
        cells: Set of cells (compacted or not)

    Returns:
        Area in square meters

    Raises:
        ValueError: If a value is neither an A5 cell ID nor a compaction marker
    """
    total = 0.0
    for cell in cells:
        if is_compaction_marker(cell):
            continue
        total += cell_area(checked_resolution(cell))
    return total
