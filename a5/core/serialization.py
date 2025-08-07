# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

from typing import List, Optional
from .utils import A5Cell, Origin
from .origin import origins

FIRST_HILBERT_RESOLUTION = 2
MAX_RESOLUTION = 30
HILBERT_START_BIT = 58  # 64 - 6 bits for origin & segment

# First 6 bits 0, remaining 58 bits 1
REMOVAL_MASK = 0x3ffffffffffffff

# First 6 bits 1, remaining 58 bits 0
ORIGIN_SEGMENT_MASK = 0xfc00000000000000

# All 64 bits 1
ALL_ONES = 0xffffffffffffffff

# Abstract cell that contains the whole world, has resolution -1 and 12 children,
# which are the res0 cells.
WORLD_CELL = 0


def get_resolution(index: int) -> int:
    """Find resolution from position of first non-00 bits from the right."""
    resolution = MAX_RESOLUTION - 1
    shifted = index >> 1  # TODO check if non-zero for point level
    while resolution > -1 and (shifted & 1) == 0:
        resolution -= 1
        # For non-Hilbert resolutions, resolution marker moves by 1 bit per resolution
        # For Hilbert resolutions, resolution marker moves by 2 bits per resolution
        shifted >>= 1 if resolution < FIRST_HILBERT_RESOLUTION else 2
    return resolution


def deserialize(index: int) -> A5Cell:
    """Deserialize a cell index into an A5Cell."""
    resolution = get_resolution(index)

    # Technically not a resolution, but can be useful to think of as an
    # abstract cell that contains the whole world
    if resolution == -1:
        return A5Cell(origin=origins[0], segment=0, S=0, resolution=resolution)

    # Extract origin*segment from top 6 bits
    top6_bits = index >> 58
    
    # Find origin and segment that multiply to give this product
    if resolution == 0:
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
    """Serialize an A5Cell into a cell index."""
    origin = cell["origin"]
    segment = cell["segment"]
    S = cell["S"]
    resolution = cell["resolution"]

    if resolution > MAX_RESOLUTION:
        raise ValueError(f"Resolution ({resolution}) is too large")

    if resolution == -1:
        return WORLD_CELL

    # Position of resolution marker as bit shift from LSB
    if resolution < FIRST_HILBERT_RESOLUTION:
        # For non-Hilbert resolutions, resolution marker moves by 1 bit per resolution
        R = resolution + 1
    else:
        # For Hilbert resolutions, resolution marker moves by 2 bits per resolution
        hilbert_resolution = 1 + resolution - FIRST_HILBERT_RESOLUTION
        R = 2 * hilbert_resolution + 1

    # First 6 bits are the origin id and the segment
    segment_n = (segment - origin.first_quintant + 5) % 5

    if resolution == 0:
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
    """Get the children of a cell at a specific resolution."""
    cell = deserialize(index)
    origin, segment, S, current_resolution = cell["origin"], cell["segment"], cell["S"], cell["resolution"]
    new_resolution = child_resolution if child_resolution is not None else current_resolution + 1

    if new_resolution < current_resolution:
        raise ValueError(f"Target resolution ({new_resolution}) must be equal to or greater than current resolution ({current_resolution})")

    if new_resolution > MAX_RESOLUTION:
        raise ValueError(f"Target resolution ({new_resolution}) exceeds maximum resolution ({MAX_RESOLUTION})")

    if new_resolution == current_resolution:
        return [index]

    new_origins = [origin]
    new_segments = [segment]
    if current_resolution == -1:
        new_origins = origins
    if (current_resolution == -1 and new_resolution > 0) or current_resolution == 0:
        new_segments = list(range(5))

    resolution_diff = new_resolution - max(current_resolution, FIRST_HILBERT_RESOLUTION - 1)
    children_count = 4 ** max(0, resolution_diff)
    shifted_S = S << (2 * max(0, resolution_diff))

    children = []
    for new_origin in new_origins:
        for new_segment in new_segments:
            for i in range(children_count):
                new_S = shifted_S + i
                children.append(serialize(A5Cell(origin=new_origin, segment=new_segment, S=new_S, resolution=new_resolution)))

    return children

def cell_to_parent(index: int, parent_resolution: Optional[int] = None) -> int:
    """Get the parent of a cell at a specific resolution."""
    cell = deserialize(index)
    origin, segment, S, current_resolution = cell["origin"], cell["segment"], cell["S"], cell["resolution"]

    new_resolution = parent_resolution if parent_resolution is not None else current_resolution - 1

    if new_resolution < -1:
        raise ValueError(f"Target resolution ({new_resolution}) cannot be less than -1")

    if new_resolution > current_resolution:
        raise ValueError(
            f"Target resolution ({new_resolution}) must be equal to or less than current resolution ({current_resolution})"
        )

    if new_resolution == current_resolution:
        return index

    resolution_diff = current_resolution - new_resolution
    shifted_S = S >> (2 * resolution_diff)

    return serialize(A5Cell(
        origin=origin,
        segment=segment,
        S=shifted_S,
        resolution=new_resolution
    ))


def get_res0_cells() -> List[int]:
    """
    Returns resolution 0 cells of the A5 system, which serve as a starting point
    for all higher-resolution subdivisions in the hierarchy.
    
    Returns:
        List of 12 cell indices
    """
    return cell_to_children(WORLD_CELL, 0)  