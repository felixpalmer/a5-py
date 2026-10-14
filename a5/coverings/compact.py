# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

"""
compact/uncompact for A5 DGGS. A compacted set of cells is a covering: its
cells sorted in curve order, then a compaction marker recording the resolution
they stand for (see ..core.compaction_marker).
"""

from typing import List, Sequence

from ..core.cell_info import get_num_children
from ..core.compaction_marker import is_compaction_marker
from ..core.serialization import checked_resolution, cell_to_children
from .slot_runs import to_covering
from .resolution import covering_resolution


def uncompact(cells: Sequence[int]) -> List[int]:
    """
    Expand a set of cells to all their cells at its resolution: the resolution
    of its compaction marker, or of its finest cell when it has none.

    **Ordering property**: If the input is sorted in curve order (as `compact`
    returns it), the output is too. All children of a cell form a contiguous,
    ordered block on the curve, so `children(A) < children(B)` whenever `A < B`.

    Raises:
        ValueError: If a value is neither an A5 cell ID nor a compaction marker
    """
    target_resolution = covering_resolution(cells)
    result: List[int] = []
    for cell in cells:
        if is_compaction_marker(cell):
            continue
        if get_num_children(checked_resolution(cell), target_resolution) == 1:
            result.append(cell)
        else:
            result.extend(cell_to_children(cell, target_resolution))
    return result


def compact(cells: Sequence[int]) -> List[int]:
    """
    Compact a set of cells: replace every complete group of siblings by their
    parent, recursively, and append a compaction marker recording the resolution of
    the input's finest cell, which `uncompact` expands back to.

    Args:
        cells: List of cell indices to compact

    Returns:
        Compacted cells sorted in curve order, then the compaction marker

    Raises:
        ValueError: If a value is neither an A5 cell ID nor a compaction marker
    """
    return to_covering(cells, covering_resolution(cells))
