# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

from typing import List, Optional
from .utils import A5Cell, Origin
from .origin import origins

FIRST_HILBERT_RESOLUTION = 2
MAX_RESOLUTION = 30
# IDs below res 30 start with a 6-bit origin (res 0) or quintant, above S
QUINTANT_SHIFT = 58
S_MASK = (1 << QUINTANT_SHIFT) - 1

# Abstract cell that contains the whole world, has resolution -1 and 12 children,
# which are the res0 cells.
WORLD_CELL = 0

# Resolution 30 IDs have no room for a 6-bit quintant: its field is 5, 3 or 1
# bits wide, marked by the tag (lowest bits) ...1, ...100 or ...10000:
#   ...1     -> 5-bit quintant (0-31),  58-bit S
#   ...100   -> 3-bit quintant (32-39), 58-bit S
#   ...10000 -> 1-bit quintant (40-41), 58-bit S
# Quintants 42-59 have no res-30 IDs.
# The number of quintants (in ID order) with resolution 30 IDs.
RES30_QUINTANTS = 42


def _res30_to_slot(index: int) -> int:
    """The leaf slot of a res-30 ID: its quintant, then its 58-bit S (see Leaf slots below)."""
    if index & 1:
        return ((index >> 59) << QUINTANT_SHIFT) | ((index >> 1) & S_MASK)
    if index & 0b100:
        return (((index >> 61) + 32) << QUINTANT_SHIFT) | ((index >> 3) & S_MASK)
    return (((index >> 63) + 40) << QUINTANT_SHIFT) | ((index >> 5) & S_MASK)


def _slot_to_res30(slot: int) -> int:
    """The res-30 ID of a leaf slot in quintants 0-41."""
    q = slot >> QUINTANT_SHIFT
    s = slot & S_MASK
    if q < 32:
        return (q << 59) | (s << 1) | 1
    if q < 40:
        return ((q - 32) << 61) | (s << 3) | 0b100
    return ((q - 40) << 63) | (s << 5) | 0b10000


# Leaf slots. The A5 curve, at resolution 30, passes through every leaf
# (res-30) cell of the globe once: picture it as a line of slots, one per leaf
# cell, numbered in curve order from 0 to 60 * 4^29 - 1. A leaf slot is the
# 6-bit quintant (0-59) then the leaf's S, left-aligned below it.
#
# A cell at resolution r occupies 4^(30-r) consecutive slots, an aligned block
# starting at its first slot, and the cells of a resolution step along the
# slots in strides of that size, as the Hilbert curve does. Unlike cell IDs,
# whose layout differs at resolutions 0, 1 and 30, slots put every cell on one
# integer line in curve order, including all 60 quintants at resolution 30.
# A leaf slot is not a cell ID.

QUINTANT_SLOTS = 1 << QUINTANT_SHIFT
ORIGIN_SLOTS = 5 * QUINTANT_SLOTS
WORLD_SLOTS = 60 * QUINTANT_SLOTS

# By resolution 0..30: the resolution tag (lowest set bit) of a cell below
# res 30, and the number of slots a cell occupies.
RESOLUTION_TAGS: List[int] = [
    1 << 57 if r == 0 else 1 << 56 if r == 1 else 1 << max(59 - 2 * r, 0) for r in range(MAX_RESOLUTION + 1)
]
SLOT_COUNTS: List[int] = [
    ORIGIN_SLOTS if r == 0 else QUINTANT_SLOTS if r == 1 else 1 << (60 - 2 * r) for r in range(MAX_RESOLUTION + 1)
]

# Res-30 IDs end in ...1, ...100 or ...10000: their tag has one of these bits
RES30_TAG_BITS = 0b10101


# The tags of resolutions 2-29: the odd bits 55 down to 1
_HILBERT_TAG_BITS = 0
for _tag in RESOLUTION_TAGS[FIRST_HILBERT_RESOLUTION:MAX_RESOLUTION]:
    _HILBERT_TAG_BITS |= _tag


def cell_first_slot(cell: int) -> int:
    """
    The first slot a cell occupies. Raises ValueError if the value is not an A5
    cell ID: its tag (lowest set bit) must be a resolution tag, and its origin
    (res 0) or quintant (res 1-29) must exist. Every res-30 pattern decodes to an
    existing quintant (0-41).
    """
    # The resolution tag: the lowest set bit (0 for the world cell)
    tag = cell & -cell
    if tag < RESOLUTION_TAGS[1]:
        if tag & RES30_TAG_BITS:
            return _res30_to_slot(cell)
        # Resolutions 2-29, in quintants 0-59: the first slot is the ID without its tag
        if (tag & _HILBERT_TAG_BITS) and cell < WORLD_SLOTS:
            return cell - tag
        if tag == 0:
            return 0
    else:
        # Resolution 0 (tag bit 57) starts its origin's 5 quintants, 1 (bit 56) its quintant
        top = cell >> QUINTANT_SHIFT
        if tag == RESOLUTION_TAGS[0] and top < 12:
            return (5 * top) << QUINTANT_SHIFT
        if tag == RESOLUTION_TAGS[1] and top < 60:
            return top << QUINTANT_SHIFT
    raise _invalid_cell(cell)


def cell_first_slot_unchecked(cell: int) -> int:
    """
    The first slot of a cell, without checking that the value is an A5 cell ID:
    for searches, which check the cell they land on. A value that is not a cell
    gives a meaningless slot.
    """
    tag = cell & -cell
    if tag == 0:
        return 0
    if tag >= RESOLUTION_TAGS[1]:
        top = cell >> QUINTANT_SHIFT
        return (5 * top if tag == RESOLUTION_TAGS[0] else top) << QUINTANT_SHIFT
    if tag & RES30_TAG_BITS:
        return _res30_to_slot(cell)
    return cell - tag


def checked_resolution(cell: int) -> int:
    """
    The resolution of a cell, as `get_resolution` gives it, but raising ValueError
    if the value is not an A5 cell ID (see `cell_first_slot` for what that requires).
    """
    tag = cell & -cell
    if tag == 0:
        return -1
    bit = tag.bit_length() - 1
    if bit < 56:
        if bit % 2 == 1 and cell < WORLD_SLOTS:
            return (59 - bit) >> 1
        if bit <= 4 and bit % 2 == 0:
            return MAX_RESOLUTION
    else:
        top = cell >> QUINTANT_SHIFT
        if bit == 57 and top < 12:
            return 0
        if bit == 56 and top < 60:
            return 1
    raise _invalid_cell(cell)


def _invalid_cell(cell: int) -> ValueError:
    return ValueError(f"Invalid cell: {cell:#x}")


def cell_slot_count(cell: int) -> int:
    """The number of slots a cell occupies."""
    # The resolution tag: the lowest set bit (0 for the world cell)
    tag = cell & -cell
    if tag == 0:
        return WORLD_SLOTS
    if tag == RESOLUTION_TAGS[0]:
        return ORIGIN_SLOTS
    if tag == RESOLUTION_TAGS[1]:
        return QUINTANT_SLOTS
    if tag & RES30_TAG_BITS:
        return 1
    # Resolutions 2-29: the slots are symmetric about the ID
    return tag << 1


def slot_to_cell(slot: int, resolution: int) -> int:
    """The res-r cell whose block of slots starts at `slot`."""
    if resolution < 0:
        return WORLD_CELL
    if 0 < resolution < MAX_RESOLUTION:
        return slot + RESOLUTION_TAGS[resolution]
    if resolution == 0:
        return (((slot >> QUINTANT_SHIFT) // 5) << QUINTANT_SHIFT) | RESOLUTION_TAGS[0]
    return _slot_to_res30(slot)


def get_resolution(index: int) -> int:
    """The resolution of a cell, from its resolution tag (lowest set bit)."""
    # The tag's position gives the resolution: bit 57 is res 0, 56 res 1,
    # 59 - 2r res r (2-29), and res 30 uses the patterns ...1, ...100 and
    # ...10000 (bits 0, 2 and 4). The world cell has no tag.
    tag = index & -index
    if tag == 0:
        return -1
    bit = tag.bit_length() - 1
    if bit == 57:
        return 0
    if bit == 56:
        return 1
    if bit <= 4 and bit % 2 == 0:
        return MAX_RESOLUTION
    return (59 - bit) >> 1


def deserialize(index: int) -> A5Cell:
    """Deserialize a cell index into an A5Cell."""
    resolution = get_resolution(index)

    # Technically not a resolution, but can be useful to think of as an
    # abstract cell that contains the whole world
    if resolution == -1:
        return A5Cell(origin=origins[0], segment=0, S=0, resolution=resolution)

    # The cell's first slot holds its quintant, then its S above the slots of one cell
    slot = cell_first_slot(index)
    quintant = slot >> QUINTANT_SHIFT
    origin = origins[quintant // 5]
    if resolution == 0:
        return A5Cell(origin=origin, segment=0, S=0, resolution=resolution)

    segment = (quintant + origin.first_quintant) % 5
    S = 0 if resolution < FIRST_HILBERT_RESOLUTION else (slot & S_MASK) // SLOT_COUNTS[resolution]
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
    if resolution == 0:
        return slot_to_cell(5 * origin.id * QUINTANT_SLOTS, 0)

    # The cell's first slot: its quintant, then S cells of this resolution into it
    offset = S * SLOT_COUNTS[resolution] if resolution >= FIRST_HILBERT_RESOLUTION else 0
    if offset >= QUINTANT_SLOTS:
        raise ValueError(f"S ({S}) is too large for resolution level {resolution}")

    quintant = 5 * origin.id + (segment - origin.first_quintant + 5) % 5
    # Quintants past RES30_QUINTANTS have no res-30 IDs: fall back to res 29
    if resolution == MAX_RESOLUTION and quintant >= RES30_QUINTANTS:
        return serialize(A5Cell(origin=origin, segment=segment, S=S >> 2, resolution=MAX_RESOLUTION - 1))
    return slot_to_cell((quintant << QUINTANT_SHIFT) + offset, resolution)


# The segments of an origin in ID (quintant) order, by its first_quintant
_QUINTANT_SEGMENTS = [[(n + first) % 5 for n in range(5)] for first in range(5)]


def cell_to_children(index: int, child_resolution: Optional[int] = None) -> List[int]:
    """Get the children of a cell at a specific resolution, in ascending ID order."""
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
    if current_resolution == -1:
        new_origins = origins
    all_segments = (current_resolution == -1 and new_resolution > 0) or current_resolution == 0

    resolution_diff = new_resolution - max(current_resolution, FIRST_HILBERT_RESOLUTION - 1)
    children_count = 4 ** max(0, resolution_diff)
    shifted_S = S << (2 * max(0, resolution_diff))

    children = []
    for new_origin in new_origins:
        # An origin's quintants in ID order: the n-th is segment (n + first_quintant) % 5
        new_segments = _QUINTANT_SEGMENTS[new_origin.first_quintant] if all_segments else [segment]
        for new_segment in new_segments:
            for i in range(children_count):
                new_S = shifted_S + i
                children.append(serialize(A5Cell(origin=new_origin, segment=new_segment, S=new_S, resolution=new_resolution)))

    return children

def _is_max_resolution(index: int) -> bool:
    """Whether a cell is at resolution 30: its tag is one of ...1, ...100 or ...10000."""
    return (index & -index & RES30_TAG_BITS) != 0


def _normalize_res30(index: int) -> int:
    """Re-pack a res-30 cell into the standard res-29 bit layout (6-bit quintant
    in [63..58], 56-bit S in [57..2], tag at bit 1). The 58-bit res-30 S is
    truncated by 2 bits, exactly as cell_to_parent(_, 29) would.
    """
    # The res-29 parent starts at the same slot, rounded down to its 4 children
    return (_res30_to_slot(index) & ~3) | 0b10


def cell_to_parent(index: int, parent_resolution: Optional[int] = None) -> int:
    """Walk a cell up the hierarchy to a coarser resolution.

    Implemented as pure bit ops over the encoded index — no deserialize /
    serialize round-trip. The three encoding regimes (non-Hilbert res 0/1,
    Hilbert res 2..29, variable-width res 30) all reduce to the same shape
    after a small amount of normalization.
    """
    if parent_resolution is None:
        parent_resolution = get_resolution(index) - 1

    # Special case: parent of resolution 0 cells is the world cell
    if parent_resolution == -1:
        return WORLD_CELL
    if parent_resolution < -1 or parent_resolution > MAX_RESOLUTION:
        raise ValueError(f"Target resolution ({parent_resolution}) is out of range")
    if index == WORLD_CELL:
        raise ValueError(
            f"Target resolution ({parent_resolution}) must be equal to or less than current resolution (-1)"
        )

    # Normalize res-30 children to the standard res-29 layout. After this,
    # the fast paths below treat the cell as a Hilbert-range cell.
    c = index
    if _is_max_resolution(index):
        if parent_resolution == MAX_RESOLUTION:
            return index  # identity (already res 30)
        c = _normalize_res30(index)
        if parent_resolution == MAX_RESOLUTION - 1:
            return c

    if parent_resolution >= FIRST_HILBERT_RESOLUTION:
        # Hilbert-range parent: clear bits below the parent tag, set the tag.
        # Identity (parent res === child res) falls out for free: the tag lands
        # in the same position and bits below the keep cut are already zero.
        keep_shift = 60 - 2 * parent_resolution
        return ((c >> keep_shift) << keep_shift) | (1 << (59 - 2 * parent_resolution))

    if parent_resolution == 1:
        # Top 6 bits already encode 5*originId + segmentN; only the tag moves.
        # Identity (cell already at res 1) is preserved.
        return ((c >> 58) << 58) | (1 << 56)

    # parent_resolution == 0: top 6 bits change from quintant (0-59) to originId (0-11).
    # Identity (cell already at res 0) needs an explicit guard since dividing
    # an originId by 5 would corrupt it. A res-0 cell has bit 57 set with all
    # lower bits zero — equivalently, all bottom 57 bits are zero.
    if (c & ((1 << 57) - 1)) == 0:
        return c
    return (((c >> 58) // 5) << 58) | (1 << 57)


# The 12 resolution-0 cells (dodecahedron faces) are a constant — compute once.
_RES0_CELLS: Optional[List[int]] = None


def get_res0_cells() -> List[int]:
    """
    Returns resolution 0 cells of the A5 system, which serve as a starting point
    for all higher-resolution subdivisions in the hierarchy.

    Returns:
        List of 12 cell indices
    """
    global _RES0_CELLS
    if _RES0_CELLS is None:
        _RES0_CELLS = cell_to_children(WORLD_CELL, 0)
    return list(_RES0_CELLS)


def is_child_of(child: int, parent: int, parent_resolution: int) -> bool:
    """Bit-level descendant test: is child the same cell as parent, or one of
    its descendants at any deeper resolution? Compares the high (quintant +
    parent's Hilbert) bits in a single shift, no deserialize needed.

    Restricted to the Hilbert range: parent_resolution must be in
    [FIRST_HILBERT_RESOLUTION .. MAX_RESOLUTION - 1], and child must not be
    a resolution-30 cell (whose encoding uses a variable quintant shift).
    Callers handling those cases should fall back to cell_to_parent equality.
    """
    # Parent's identifying bits occupy positions 63..(60-2P): 6 quintant bits
    # + 2(P-1) Hilbert bits. Bit (59-2P) is the tag, below that is zero.
    # Shifting both right by (60-2P) keeps exactly those identifying bits and
    # discards the tag, so a descendant matches iff the high bits match.
    shift = 60 - 2 * parent_resolution
    return (child >> shift) == (parent >> shift)
