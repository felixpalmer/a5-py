# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

# The compaction marker: the value a covering carries as its last
# element, recording the resolution its cells stand for. It is a value no cell
# can take: quintant 60 (only 0-59 exist), the resolution in bits 55-48, and the
# marker tag 1000000 in bits 6-0. Cell IDs end in a 1 followed by an odd number
# of zeros, or in one of the res-30 patterns ...1, ...100, ...10000, never in a 1
# followed by 6 zeros. The other bits carry no meaning yet: they are written as
# 0 and ignored when read.
#
#   63-58    57-56   55-48        47-7          6-0
#   111100   00      resolution   reserved (0)  1000000

from .serialization import QUINTANT_SHIFT, MAX_RESOLUTION

_COMPACTION_MARKER_PREFIX = 60 << QUINTANT_SHIFT
_COMPACTION_MARKER_END = 61 << QUINTANT_SHIFT
_COMPACTION_MARKER_RESOLUTION_SHIFT = 48
_COMPACTION_MARKER_TAG = 0b1000000
_LOW_7_BITS = 0b1111111


def compaction_marker(resolution: int) -> int:
    """The compaction marker recording `resolution`."""
    return _COMPACTION_MARKER_PREFIX | (resolution << _COMPACTION_MARKER_RESOLUTION_SHIFT) | _COMPACTION_MARKER_TAG


def compaction_marker_resolution(value: int) -> int:
    """The resolution a compaction marker records."""
    return (value >> _COMPACTION_MARKER_RESOLUTION_SHIFT) & 0xFF


def is_compaction_marker(value: int) -> bool:
    """
    Check whether a value is a compaction marker: the value a covering
    carries, as its last element, to record its resolution. It is not a cell.
    """
    return (
        _COMPACTION_MARKER_PREFIX <= value < _COMPACTION_MARKER_END
        and (value & _LOW_7_BITS) == _COMPACTION_MARKER_TAG
        and compaction_marker_resolution(value) <= MAX_RESOLUTION
    )
