# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

from typing import Callable, List

from ..core.compaction_marker import is_compaction_marker
from ..core.serialization import cell_first_slot, cell_first_slot_unchecked, cell_slot_count, checked_resolution
from .resolution import get_compaction_resolution
from .slot_runs import Cells, SlotRuns, append_slot_run, slot_runs_to_collection, to_slot_runs


def _union_slot_runs(a: SlotRuns, b: SlotRuns) -> SlotRuns:
    """Merge two lists of slot runs, keeping slots in either."""
    out: SlotRuns = []
    i = 0
    j = 0
    while i < len(a) or j < len(b):
        if j >= len(b) or (i < len(a) and a[i] <= b[j]):
            append_slot_run(out, a[i], a[i + 1])
            i += 2
        else:
            append_slot_run(out, b[j], b[j + 1])
            j += 2
    return out


def _intersect_slot_runs(a: SlotRuns, b: SlotRuns) -> SlotRuns:
    """Slots in both lists of slot runs."""
    out: SlotRuns = []
    i = 0
    j = 0
    while i < len(a) and j < len(b):
        lo = a[i] if a[i] > b[j] else b[j]
        hi = a[i + 1] if a[i + 1] < b[j + 1] else b[j + 1]
        if lo < hi:
            out.append(lo)
            out.append(hi)
        if a[i + 1] < b[j + 1]:
            i += 2
        else:
            j += 2
    return out


def _difference_slot_runs(a: SlotRuns, b: SlotRuns) -> SlotRuns:
    """Slots in the first list of slot runs but not the second."""
    out: SlotRuns = []
    j = 0
    for i in range(0, len(a), 2):
        lo = a[i]
        hi = a[i + 1]
        while j < len(b) and b[j + 1] <= lo:
            j += 2
        k = j
        while k < len(b) and b[k] < hi:
            if b[k] > lo:
                out.append(lo)
                out.append(b[k])
            if b[k + 1] > lo:
                lo = b[k + 1]
            k += 2
        if lo < hi:
            out.append(lo)
            out.append(hi)
    return out


def _same_resolution(a: Cells, b: Cells) -> int:
    """
    The resolution of two sets of cells, which must be the same: A5 resolutions
    don't nest geometrically, so combining sets at different ones has no meaning.
    """
    resolution_a = get_compaction_resolution(a)
    resolution_b = get_compaction_resolution(b)
    if resolution_a != resolution_b:
        raise ValueError(f"Cannot combine cells at resolution {resolution_a} with cells at resolution {resolution_b}")
    return resolution_a


def _combine(a: Cells, b: Cells, operation: Callable[[SlotRuns, SlotRuns], SlotRuns]) -> List[int]:
    """Combine two sets of cells as slot runs, compacted at their resolution."""
    resolution = _same_resolution(a, b)
    return slot_runs_to_collection(operation(to_slot_runs(a), to_slot_runs(b)), resolution)


def union(a: Cells, b: Cells) -> List[int]:
    """
    The union of two sets of cells: cells in either. Both sets must be at the
    same resolution, and the result is compacted at it.

    Args:
        a: First set of cells (compacted or not)
        b: Second set of cells (compacted or not)

    Returns:
        Compacted cells, with a compaction marker recording the resolution

    Raises:
        ValueError: If the sets are at different resolutions, or a value is neither an A5 cell ID nor a
            compaction marker
    """
    return _combine(a, b, _union_slot_runs)


def intersect(a: Cells, b: Cells) -> List[int]:
    """
    The intersection of two sets of cells: cells in both. Both sets must be at
    the same resolution, and the result is compacted at it.

    Args:
        a: First set of cells (compacted or not)
        b: Second set of cells (compacted or not)

    Returns:
        Compacted cells, with a compaction marker recording the resolution

    Raises:
        ValueError: If the sets are at different resolutions, or a value is neither an A5 cell ID nor a
            compaction marker
    """
    return _combine(a, b, _intersect_slot_runs)


def difference(a: Cells, b: Cells) -> List[int]:
    """
    The difference of two sets of cells: cells in `a` but not in `b`. Both sets
    must be at the same resolution, and the result is compacted at it.

    Args:
        a: Set of cells to subtract from (compacted or not)
        b: Set of cells to subtract (compacted or not)

    Returns:
        Compacted cells, with a compaction marker recording the resolution

    Raises:
        ValueError: If the sets are at different resolutions, or a value is neither an A5 cell ID nor a
            compaction marker
    """
    return _combine(a, b, _difference_slot_runs)


def overlaps(a: Cells, b: Cells) -> bool:
    """
    Check whether two sets of cells share any cell. Both sets must be at the same
    resolution.

    Args:
        a: First set of cells (compacted or not)
        b: Second set of cells (compacted or not)

    Returns:
        Whether some cell is in both sets

    Raises:
        ValueError: If the sets are at different resolutions, or a value is neither an A5 cell ID nor a
            compaction marker
    """
    _same_resolution(a, b)
    return len(_intersect_slot_runs(to_slot_runs(a), to_slot_runs(b))) > 0


def contains(cells: Cells, cell: int) -> bool:
    """
    Check whether a cell is in a set of cells. The cell must be at the set's
    resolution: A5 cells don't nest geometrically across resolutions, so for a
    point-in-polygon test pass `lonlat_to_cell(point, resolution)` with the
    resolution of the set. Uses a binary search, so `cells` must be sorted in
    curve order, as returned by `compact` and the other A5 functions.

    Args:
        cells: Sorted set of cells (compacted or not)
        cell: Cell to test, at the set's resolution

    Returns:
        Whether the cell is in the set

    Raises:
        ValueError: If the cell is at a different resolution from the set, or it, or the set's cell the
            search lands on, is not an A5 cell ID
    """
    resolution = get_compaction_resolution(cells)
    cell_resolution = checked_resolution(cell)
    if cell_resolution != resolution:
        raise ValueError(f"Cannot test a cell at resolution {cell_resolution} against cells at resolution {resolution}")
    slot = cell_first_slot(cell)
    n = len(cells)
    if n > 0 and is_compaction_marker(cells[n - 1]):
        n -= 1

    # The last cell starting at or before the cell's first slot is the only one
    # that can hold it. Only that cell is checked to be a cell: the search steps
    # just need an order
    low = 0
    high = n - 1
    found = -1
    while low <= high:
        mid = (low + high) >> 1
        if cell_first_slot_unchecked(cells[mid]) <= slot:
            found = mid
            low = mid + 1
        else:
            high = mid - 1
    return found >= 0 and slot < cell_first_slot(cells[found]) + cell_slot_count(cells[found])
