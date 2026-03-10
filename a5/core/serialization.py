# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

from typing import List, Optional
from .utils import A5Cell, Origin
from .origin import origins

FIRST_HILBERT_RESOLUTION = 2
MAX_RESOLUTION = 30
HILBERT_START_BIT = 58  # 64 - 6 bits for origin & segment

# Abstract cell that contains the whole world, has resolution -1 and 12 children,
# which are the res0 cells.
WORLD_CELL = 0


def get_resolution(index: int) -> int:
    """Find resolution from position of first non-00 bits from the right."""
    if index == 0:
        return -1

    # Resolution 30 uses three encoding patterns:
    #   ...1     -> 5-bit quintant (0-31),  58-bit S
    #   ...100   -> 3-bit quintant (32-39), 58-bit S
    #   ...10000 -> 1-bit quintant (40-41), 58-bit S
    if (index & 1) or (index & 0b111) == 0b100 or (index & 0b11111) == 0b10000:
        return MAX_RESOLUTION

    resolution = MAX_RESOLUTION - 1
    shifted = index >> 1
    if shifted == 0:
        return -1

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

    # For res 30, quintant bits are fewer to make room for S:
    #   ...1     marker (1 bit)  -> 5-bit quintant (0-31)
    #   ...100   marker (3 bits) -> 3-bit quintant + 32 (32-39)
    #   ...10000 marker (5 bits) -> 1-bit quintant + 40 (40-41)
    quintant_shift = HILBERT_START_BIT
    quintant_offset = 0
    if resolution == MAX_RESOLUTION:
        marker_bits = 1 if (index & 1) else (3 if (index & 0b100) else 5)
        quintant_shift = HILBERT_START_BIT + marker_bits
        quintant_offset = 0 if marker_bits == 1 else (32 if marker_bits == 3 else 40)

    # Extract origin*segment from top bits
    top_bits = (index >> quintant_shift) + quintant_offset

    # Find origin and segment
    if resolution == 0:
        origin = origins[top_bits]
        segment = 0
    else:
        origin_id = top_bits // 5
        origin = origins[origin_id]
        segment = (top_bits + origin.first_quintant) % 5

    if origin is None:
        raise ValueError(f"Could not parse origin: {top_bits}")

    if resolution < FIRST_HILBERT_RESOLUTION:
        return A5Cell(origin=origin, segment=segment, S=0, resolution=resolution)

    # Mask away origin & segment and shift away resolution and marker bits
    hilbert_levels = resolution - FIRST_HILBERT_RESOLUTION + 1
    hilbert_bits = 2 * hilbert_levels
    removal_mask = (1 << quintant_shift) - 1
    S = (index & removal_mask) >> (quintant_shift - hilbert_bits)

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

    # For res 30, quintant bits are fewer to make room for S:
    #   quintant 0-31:  ...1     marker -> 5-bit quintant
    #   quintant 32-39: ...100   marker -> 3-bit quintant + 32
    #   quintant 40-41: ...10000 marker -> 1-bit quintant + 40
    #   quintant 42+:   fall back to res 29
    quintant_shift = HILBERT_START_BIT

    # Position of resolution marker as bit shift from LSB
    if resolution < FIRST_HILBERT_RESOLUTION:
        R = resolution + 1
    else:
        hilbert_resolution = 1 + resolution - FIRST_HILBERT_RESOLUTION
        R = 2 * hilbert_resolution + 1

    # Top bits encode the origin id and segment
    segment_n = (segment - origin.first_quintant + 5) % 5

    if resolution == 0:
        index = origin.id << quintant_shift
    else:
        quintant = 5 * origin.id + segment_n
        if resolution == MAX_RESOLUTION:
            if quintant <= 31:
                quintant_shift = HILBERT_START_BIT + 1
                quintant_value = quintant
            elif quintant <= 39:
                quintant_shift = HILBERT_START_BIT + 3
                quintant_value = quintant - 32
            elif quintant <= 41:
                quintant_shift = HILBERT_START_BIT + 5
                quintant_value = quintant - 40
            else:
                return serialize(A5Cell(origin=origin, segment=segment, S=S >> 2, resolution=MAX_RESOLUTION - 1))
            index = quintant_value << quintant_shift
        else:
            index = quintant << quintant_shift

    if resolution >= FIRST_HILBERT_RESOLUTION:
        hilbert_levels = resolution - FIRST_HILBERT_RESOLUTION + 1
        hilbert_bits = 2 * hilbert_levels
        if S >= (1 << hilbert_bits):
            raise ValueError(f"S ({S}) is too large for resolution level {resolution}")
        index += S << (quintant_shift - hilbert_bits)

    # Resolution is encoded by position of the least significant 1
    index |= 1 << (quintant_shift - R)

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

    # Special case: parent of resolution 0 cells is the world cell
    if new_resolution == -1:
        return WORLD_CELL

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


def is_first_child(index: int, resolution: Optional[int] = None) -> bool:
    """Check whether index corresponds to first child of its parent."""
    if resolution is None:
        resolution = get_resolution(index)

    if resolution < 2:
        # For resolution 0: first child is origin 0 (child count = 12)
        # For resolution 1: first children are at multiples of 5 (child count = 5)
        top6_bits = index >> HILBERT_START_BIT
        child_count = 12 if resolution == 0 else 5
        return top6_bits % child_count == 0

    if resolution == MAX_RESOLUTION:
        # S's 2 LSBs sit just above the marker bits
        marker_bits = 1 if (index & 1) else (3 if (index & 0b100) else 5)
        return (index & (3 << marker_bits)) == 0

    s_position = 2 * (MAX_RESOLUTION - resolution)
    s_mask = 3 << s_position  # Mask for the 2 LSBs of S
    return (index & s_mask) == 0


def get_stride(resolution: int) -> int:
    """Difference between two neighbouring sibling cells at a given resolution."""
    # Both level 0 & 1 just write values 0-11 or 0-59 to the first 6 bits
    if resolution < 2:
        return 1 << HILBERT_START_BIT

    # For res 30, S is shifted left by 1 (marker bit at position 0)
    if resolution == MAX_RESOLUTION:
        return 2

    # For hilbert levels, the position shifts by 2 bits per resolution level
    s_position = 2 * (MAX_RESOLUTION - resolution)
    return 1 << s_position