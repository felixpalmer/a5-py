# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

# A region found by descending the cell hierarchy: a cell wholly inside is kept
# whole, one wholly outside is dropped, and the rest split, so the work follows
# the region's boundary rather than its area. The descent runs in curve order,
# stepping the curve one digit per level (see lattice curve_child), so the cells
# come out as sorted slot runs, with no cell IDs to encode and nothing to sort.

from typing import Callable, List

from ..core.coordinate_systems import Face
from ..core.serialization import FIRST_HILBERT_RESOLUTION, SLOT_COUNTS
from ..core.tiling import get_pentagon_center
from ..coverings.slot_runs import SlotRuns, append_slot_run
from ..lattice import CurveNode, Triple, curve_child, triple_to_curve_node
from .triple_cells import QUINTANT_ORIENTATION, QUINTANT_PREFIX

# How a cell lies relative to the region: see `CurveDescentClassifier`.
OUTSIDE = 0
INSIDE = 1
SPLIT = 2

# Classifies a cell of the descent by its center, given in its face's frame:
# (origin_id, resolution, center, slot) -> INSIDE keeps the cell whole (all its
# cells at the target resolution are in the region), OUTSIDE drops it (none
# are), and SPLIT descends into its children. At the target resolution it must
# decide, INSIDE or OUTSIDE.
CurveDescentClassifier = Callable[[int, int, Face, int], int]


def descend_in_curve_order(
    starts: List[int],
    start_level: int,
    resolution: int,
    classify: CurveDescentClassifier,
    runs: SlotRuns,
) -> None:
    """
    Descend from `starts` -- non-overlapping cells at one Hilbert level, as flat
    triples (origin_id, quintant, x, y, z), in any order -- to `resolution`,
    appending the cells of the region `classify` describes to `runs`, as slot
    runs in curve order.
    """
    shift = 58 - 2 * start_level
    entries = []
    for c in range(0, len(starts), 5):
        q = starts[c] * 5 + starts[c + 1]
        triple = Triple(starts[c + 2], starts[c + 3], starts[c + 4])
        state = triple_to_curve_node(triple, start_level, QUINTANT_ORIENTATION[q])
        entries.append((QUINTANT_PREFIX[q] | (state.s << shift), c, triple, state))
    # Slots are distinct, so the sort never compares past them
    entries.sort(key=lambda e: e[0])

    target_level = resolution - FIRST_HILBERT_RESOLUTION + 1
    # The descent state below the cell being visited at each level (a level's is
    # only overwritten once its cell's children are done)
    nodes = [CurveNode() for _ in range(target_level + 1)]

    def descend(level: int, triple: Triple, flavor: int, slot: int) -> None:
        """Visit the cell at Hilbert `level` (its descent state in `nodes`) with the given triple, flavor and first slot."""
        res = level + FIRST_HILBERT_RESOLUTION - 1
        center = get_pentagon_center(level, quintant, triple, flavor)
        kind = classify(origin_id, res, center, slot)
        if kind == INSIDE:
            append_slot_run(runs, slot, slot + SLOT_COUNTS[res])
            return
        if kind == OUTSIDE or level == target_level:
            return
        # The children's slots follow one another in curve order
        node = nodes[level]
        child_node = nodes[level + 1]
        child_slots = SLOT_COUNTS[res + 1]
        child_level = level + 1
        for digit in range(4):
            child_triple, child_flavor = curve_child(node, digit, child_level, orientation, child_node)
            descend(child_level, child_triple, child_flavor, slot)
            slot += child_slots

    for slot, c, triple, state in entries:
        origin_id = starts[c]
        quintant = starts[c + 1]
        orientation = QUINTANT_ORIENTATION[origin_id * 5 + quintant]
        node = nodes[start_level]
        start_node = state.node
        node.motif = start_node.motif
        node.flip = start_node.flip
        node.pos_a = start_node.pos_a
        node.pos_b = start_node.pos_b
        descend(start_level, triple, state.flavor, slot)
