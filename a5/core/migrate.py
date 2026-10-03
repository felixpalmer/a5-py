# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

from .origin import face_step, quintant_to_segment
from .serialization import deserialize, serialize, FIRST_HILBERT_RESOLUTION
from .utils import A5Cell
from ..lattice import compat_s_to_triple, triple_to_s

# The v0 face layouts, by origin id (curve order): the quintant orientations
# and first quintant of every face. Windings are the same as in v1.
FAN = ['vu', 'uw', 'vw', 'vw', 'vw']
COUNTER_STEP = ['wu', 'uv', 'wv', 'wu', 'uw']
COUNTER_JUMP = ['vu', 'uv', 'wv', 'wu', 'uw']
CLOCKWISE_STEP = ['wu', 'uw', 'vw', 'vu', 'uw']
V0_LAYOUTS = [
    (FAN, 4),
    (COUNTER_JUMP, 2),
    (COUNTER_STEP, 3),
    (COUNTER_STEP, 0),
    (CLOCKWISE_STEP, 2),
    (COUNTER_JUMP, 4),
    (CLOCKWISE_STEP, 2),
    (CLOCKWISE_STEP, 2),
    (COUNTER_STEP, 3),
    (COUNTER_JUMP, 0),
    (COUNTER_JUMP, 3),
    (CLOCKWISE_STEP, 0),
]


def migrate(cell: int) -> int:
    """
    Migrates a cell id from the v0 index (a5 <= 0.10) to the v1 index. Both
    versions share the same cells; the v1 index threads the curve differently:
    a new curve within each quintant, and a new quintant order on some faces.
    So the cell is located in v0 terms (quintant + lattice triple, via the
    original curve) and re-encoded in v1 terms.

    Args:
        cell: A cell id in the v0 index

    Returns:
        The id of the same cell in the v1 index
    """
    data = deserialize(cell)
    origin = data["origin"]
    resolution = data["resolution"]
    if resolution < FIRST_HILBERT_RESOLUTION - 1:
        return cell

    # Locate the cell's quintant in the v0 layout. Windings are unchanged, so
    # the v0 face shares the v1 face's direction of travel.
    v0_orientation, v0_first_quintant = V0_LAYOUTS[origin.id]
    step = face_step(origin)
    face_relative_quintant = (data["segment"] - origin.first_quintant + 5) % 5
    quintant = (v0_first_quintant + step * face_relative_quintant + 5) % 5
    segment, orientation = quintant_to_segment(quintant, origin)
    if resolution == FIRST_HILBERT_RESOLUTION - 1:
        return serialize(A5Cell(origin=origin, segment=segment, S=0, resolution=resolution))

    hilbert_resolution = 1 + resolution - FIRST_HILBERT_RESOLUTION
    triple = compat_s_to_triple(data["S"], hilbert_resolution, v0_orientation[face_relative_quintant])
    new_s = triple_to_s(triple, hilbert_resolution, orientation)
    return serialize(A5Cell(origin=origin, segment=segment, S=new_s, resolution=resolution))
