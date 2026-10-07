# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

# The collection engine. Each cell covers a block of leaf slots (see
# core/serialization), so a set of cells is a list of sorted, disjoint slot runs:
# cells are turned into slot runs, set operations merge runs, and runs are
# turned back into the coarsest cells covering them. Nothing is uncompacted.

import math
from typing import List, Sequence

from ..core.compaction_marker import compaction_marker, is_compaction_marker
from ..core.serialization import (
    cell_first_slot,
    cell_slot_count,
    slot_to_cell,
    SLOT_COUNTS,
    ORIGIN_SLOTS,
    QUINTANT_SHIFT,
    QUINTANT_SLOTS,
    RES30_TAG_BITS,
    RESOLUTION_TAGS,
    S_MASK,
    WORLD_SLOTS,
    WORLD_CELL,
)

# A set of cells, compacted or not, possibly ending in a compaction marker
Cells = Sequence[int]
# Sorted, disjoint, half-open runs of leaf slots [lo, hi), flattened as
# [lo0, hi0, lo1, hi1, ...]: the form set operations work on
SlotRuns = List[int]


def _merge_cells(cells: Cells, out: SlotRuns, check_order: bool) -> bool:
    """
    Merge cells, in the given order, into sorted and disjoint slot runs in `out`.
    With `check_order`, stop and return False at the first cell starting before
    the one before it.
    """
    previous_lo = 0
    for cell in cells:
        if is_compaction_marker(cell):
            continue
        lo = cell_first_slot(cell)
        hi = lo + cell_slot_count(cell)
        if check_order and lo < previous_lo:
            return False
        previous_lo = lo
        # Drop earlier runs this one contains, then merge with the one before it
        top = len(out) - 2
        while top >= 0 and out[top] >= lo and out[top + 1] <= hi:
            del out[top:]
            top -= 2
        append_slot_run(out, lo, hi)
    return True


def _id_inside_run(cell: int) -> bool:
    """Whether a cell's ID lies inside its own slot run, so IDs sort like runs: res 1-29."""
    tag = cell & -cell
    return cell != WORLD_CELL and tag != RESOLUTION_TAGS[0] and (tag & RES30_TAG_BITS) == 0


def _sort_cells(cells: Cells) -> List[int]:
    """
    Cells sorted by a slot inside each cell's run: then disjoint runs come out
    in order, and a run can only be preceded by runs it contains or that
    contain it. Below res 30 and above res 0 the ID itself is such a slot, so a
    native sort of the IDs does it.
    """
    if all(_id_inside_run(cell) for cell in cells):
        return sorted(cells)
    return sorted(cells, key=lambda cell: WORLD_SLOTS if is_compaction_marker(cell) else cell_first_slot(cell))


def to_slot_runs(cells: Cells) -> SlotRuns:
    """
    The slot runs covered by a set of cells, sorted and merged. Compaction
    markers are skipped.

    Collections come sorted in curve order, so the cells are first merged as
    given, checking the order as they go; only input found out of order is
    sorted, and merged again.
    """
    runs: SlotRuns = []
    if not _merge_cells(cells, runs, True):
        runs.clear()
        _merge_cells(_sort_cells(cells), runs, False)
    return runs


def append_slot_run(runs: SlotRuns, lo: int, hi: int) -> None:
    """Append a slot run [lo, hi) to sorted runs starting at or before lo, merging if they touch or overlap."""
    last = len(runs) - 1
    if last > 0 and runs[last] >= lo:
        if hi > runs[last]:
            runs[last] = hi
    else:
        runs.append(lo)
        runs.append(hi)


def slot_runs_to_cells(runs: SlotRuns) -> List[int]:
    """
    The coarsest cells covering the slot runs, in curve order. Runs built from
    cells at resolution r or coarser are aligned to res-r cells, so no cell finer
    than r is needed.
    """
    out: List[int] = []
    for i in range(0, len(runs), 2):
        lo = runs[i]
        hi = runs[i + 1]
        while lo < hi:
            if lo == 0 and hi == WORLD_SLOTS:
                out.append(WORLD_CELL)
                break
            if (lo & S_MASK) == 0:
                # Whole origins, then whole quintants
                if (lo >> QUINTANT_SHIFT) % 5 == 0 and lo + ORIGIN_SLOTS <= hi:
                    out.append(slot_to_cell(lo, 0))
                    lo += ORIGIN_SLOTS
                    continue
                if lo + QUINTANT_SLOTS <= hi:
                    out.append(slot_to_cell(lo, 1))
                    lo += QUINTANT_SLOTS
                    continue
            # The coarsest Hilbert-level cell that starts at lo and fits: its span is
            # 4^k slots, at most the alignment of lo and the length of the run
            offset = lo & S_MASK
            bits = 56 if offset == 0 else (offset & -offset).bit_length() - 1
            fit = (hi - lo).bit_length() - 1
            if fit < bits:
                bits = fit
            r = max(2, math.ceil((60 - bits) / 2))
            while lo + SLOT_COUNTS[r] > hi:
                r += 1
            out.append(slot_to_cell(lo, r))
            lo += SLOT_COUNTS[r]
    return out


def slot_runs_to_collection(runs: SlotRuns, resolution: int) -> List[int]:
    """The coarsest cells covering slot runs, in curve order, then the compaction marker for `resolution`."""
    cells = slot_runs_to_cells(runs)
    if resolution >= 0:
        cells.append(compaction_marker(resolution))
    return cells


def compact_cells(cells: Cells) -> List[int]:
    """
    Compact cells without appending a compaction marker: the coarsest cells covering
    them, sorted in curve order. For internal use on intermediate results.
    """
    return slot_runs_to_cells(to_slot_runs(cells))


def to_collection(cells: Cells, resolution: int) -> List[int]:
    """
    Compact cells, at resolution `resolution` or coarser, into a collection: the
    coarsest cells covering them, sorted in curve order, then the compaction
    marker for `resolution`. The resolution is given, so an empty fill still
    records it.
    """
    return slot_runs_to_collection(to_slot_runs(cells), resolution)
