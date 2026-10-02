# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

from .origin import segment_to_quintant
from .serialization import deserialize, serialize, FIRST_HILBERT_RESOLUTION
from .utils import A5Cell
from ..lattice import compat_s_to_triple, triple_to_s


def migrate(cell: int) -> int:
    """
    Migrates a cell id from the v0 index (a5 <= 0.10, original curve) to the v1
    index (non-self-intersecting curve). Both versions share the same cells and
    the same origin/segment/resolution bits; only the curve position S within
    the quintant differs, so the old S is decoded to its lattice triple and
    re-encoded along the new curve.

    Args:
        cell: A cell id in the v0 index

    Returns:
        The id of the same cell in the v1 index
    """
    data = deserialize(cell)
    resolution = data["resolution"]
    if resolution < FIRST_HILBERT_RESOLUTION:
        return cell

    _, orientation = segment_to_quintant(data["segment"], data["origin"])
    hilbert_resolution = 1 + resolution - FIRST_HILBERT_RESOLUTION
    triple = compat_s_to_triple(data["S"], hilbert_resolution, orientation)
    new_s = triple_to_s(triple, hilbert_resolution, orientation)
    return serialize(A5Cell(origin=data["origin"], segment=data["segment"], S=new_s, resolution=resolution))
