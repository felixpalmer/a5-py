# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

from typing import List, Optional
from .utils import A5Cell, Origin, origins

FIRST_HILBERT_RESOLUTION = 3
MAX_RESOLUTION = 31
HILBERT_START_BIT = 58  # 64 - 6 bits for origin & segment

REMOVAL_MASK = 0x3ffffffffffffff
ORIGIN_SEGMENT_MASK = 0xfc00000000000000
ALL_ONES = 0xffffffffffffffff


def get_resolution(index: int) -> int:
    #  Find resolution from position of first non-00 bits from the right
    resolution = MAX_RESOLUTION - 1
    shifted = index >> 1 # TODO check if non-zero for point level
    while resolution > 0 and (shifted & 1) == 0:
        resolution -= 1
        # For non-Hilbert resolutions, resolution marker moves by 1 bit per resolution
        # For Hilbert resolutions, resolution marker moves by 2 bits per resolution
        shifted >>= 1 if resolution < FIRST_HILBERT_RESOLUTION else 2
    return resolution


def deserialize(index: int) -> A5Cell:
    resolution = get_resolution(index)

    if resolution == 0:
        return A5Cell(origin=origins[0], segment=0, S=0, resolution=resolution)
  
    # Extract origin*segment from top 6 bits
    top6_bits = index >> 58

    # Find origin and segment that multiply to give this product
    origin_id = top6_bits
    segment = index % 5

    if resolution == 1:
        origin_id = top6_bits
        origin = origins[origin_id]
        segment = 0
    else:
        origin_id = top6_bits // 5
        origin = origins[origin_id]
        segment = (top6_bits + origin.first_quintant) % 5

    if origin is None:
        raise ValueError(f"Could not parse origin: {top6_bits}")

    if resolution < FIRST_HILBERT_RESOLUTION:
        return A5Cell(origin=origin, segment=segment, S=0, resolution=resolution)

    # Mask away origin & segment and shift away resolution and 00 bits
    hilbert_levels = resolution - FIRST_HILBERT_RESOLUTION + 1
    hilbert_bits = 2 * hilbert_levels
    shift = HILBERT_START_BIT - hilbert_bits
    S = (index & REMOVAL_MASK) >> shift

    return A5Cell(origin=origin, segment=segment, S=S, resolution=resolution)


def serialize(cell: A5Cell) -> int:
    origin = cell.origin
    segment = cell.segment
    S = cell.S
    resolution = cell.resolution

    if resolution > MAX_RESOLUTION:
        raise ValueError(f"Resolution ({resolution}) is too large")

    if resolution == 0:
        return 0
    # Position of resolution marker as bit shift from LSB
    if resolution < FIRST_HILBERT_RESOLUTION:
        # For non-Hilbert resolutions, resolution marker moves by 1 bit per resolution
        R = resolution
    else:
        # For Hilbert resolutions, resolution marker moves by 2 bits per resolution
        hilbert_resolution = 1 + resolution - FIRST_HILBERT_RESOLUTION
        R = 2 * hilbert_resolution + 1

    # First 6 bits are the origin id and the segment
    segment_n = (segment - origin.first_quintant + 5) % 5

    if resolution == 1:
        index = origin.id << 58
    else:
        index = (5 * origin.id + segment_n) << 58

    if resolution >= FIRST_HILBERT_RESOLUTION:
        # Number of bits required for S Hilbert curve
        hilbert_levels = resolution - FIRST_HILBERT_RESOLUTION + 1
        hilbert_bits = 2 * hilbert_levels
        if S >= (1 << hilbert_bits):
            raise ValueError(f"S ({S}) is too large for resolution level {resolution}")
        # Next (2 * hilbertResolution) bits are S (hilbert index within segment)
        index += S << (HILBERT_START_BIT - hilbert_bits)
  
    # Resolution is encoded by position of the least significant 1
    index |= 1 << (HILBERT_START_BIT - R)

    return index

def cell_to_children(index: int, child_resolution: Optional[int] = None) -> List[int]:
    cell = deserialize(index)
    origin, segment, S, current_resolution = cell.origin, cell.segment, cell.S, cell.resolution
    new_resolution = child_resolution if child_resolution is not None else current_resolution + 1

    if new_resolution <= current_resolution:
        raise ValueError(f"Target resolution ({new_resolution}) must be greater than current resolution ({current_resolution})")

    if new_resolution > MAX_RESOLUTION:
        raise ValueError(f"Target resolution ({new_resolution}) exceeds maximum resolution ({MAX_RESOLUTION})")

    new_origins = [origin]
    new_segments = [segment]
    if current_resolution == 0:
        new_origins = origins
    if (current_resolution == 0 and new_resolution > 1) or current_resolution == 1:
        new_segments = list(range(5))

    resolution_diff = new_resolution - max(current_resolution, FIRST_HILBERT_RESOLUTION - 1)
    if resolution_diff < 0:
        resolution_diff = 0

    children_count = 4 ** resolution_diff
    shifted_S = S << (2 * resolution_diff)

    children = []
    for new_origin in new_origins:
        for new_segment in new_segments:
            for i in range(children_count):
                new_S = shifted_S + i
                children.append(serialize(A5Cell(origin=new_origin, segment=new_segment, S=new_S, resolution=new_resolution)))

    return children

def cell_to_parent(index: int, parent_resolution: Optional[int] = None) -> int:
    cell = deserialize(index)
    origin, segment, S, current_resolution = cell.origin, cell.segment, cell.S, cell.resolution

    if current_resolution < 0:
        raise ValueError(f"Deserialized resolution is negative: {current_resolution}")

    new_resolution = parent_resolution if parent_resolution is not None else current_resolution - 1

    if new_resolution < 0:
        raise ValueError(f"Target resolution ({new_resolution}) cannot be negative")

    if new_resolution >= current_resolution:
        raise ValueError(
            f"Target resolution ({new_resolution}) must be less than current resolution ({current_resolution})"
        )

    resolution_diff = current_resolution - new_resolution

    # Extra protection against negative shift
    if resolution_diff < 0:
        raise ValueError(f"Resolution diff ({resolution_diff}) is negative")

    shifted_S = S >> (2 * resolution_diff)
    return serialize(A5Cell(origin=origin, segment=segment, S=shifted_S, resolution=new_resolution))
