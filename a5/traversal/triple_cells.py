# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

# Cells handled in triple space -- (origin_id, quintant, x, y, z) -- by the
# traversal algorithms that walk many neighboring cells: they key and dedup
# cells as plain integers and encode a cell to its ID only when it is output.

from typing import Callable, Iterable, List, Optional

from ..core.coordinate_systems import Spherical
from ..lattice import Triple, s_to_triple, triple_flavor, triple_in_bounds, triple_to_s
from ..core.serialization import deserialize, serialize, FIRST_HILBERT_RESOLUTION
from ..core.origin import origins, quintant_to_segment, segment_to_quintant
from ..core.tiling import get_pentagon_center
from ..projections.dodecahedron import DodecahedronProjection
from .lattice_boundary import get_boundary_neighbor_triples
from .neighbors import NEIGHBOR_DELTAS

_dodecahedron = DodecahedronProjection()

# A cell's key packs the quintant (origin.id * 5 + quintant, < 60), parity,
# and the low _KEY_BITS bits of -x and -z (y follows). Up to Hilbert resolution
# 21 the coordinates fit whole; above it, two cells of one quintant share a key
# only if they are 2^22 rows apart, and a walk holding both would need ~2^21
# steps (~10^12 cells for a disk) -- far past what fits in memory.
_KEY_BITS = 22
_KEY_MASK = (1 << _KEY_BITS) - 1
_KEY_SIDE = 1 << _KEY_BITS

# Segment and curve orientation of each of the 60 quintants, by origin.id * 5 + quintant
_QUINTANT_SEGMENTS = [quintant_to_segment(q, origin) for origin in origins for q in range(5)]


def triple_cell_key(origin_id: int, quintant: int, x: int, y: int, z: int) -> int:
    """The integer key of a cell, unique among the cells of any one traversal."""
    return ((((-x) & _KEY_MASK) * _KEY_SIDE + ((-z) & _KEY_MASK)) * 2 + x + y + z
            + (origin_id * 5 + quintant) * 2 * _KEY_SIDE * _KEY_SIDE)


def triple_cell_to_id(origin_id: int, quintant: int, x: int, y: int, z: int,
                      hilbert_res: int, resolution: int) -> int:
    """The cell ID of a cell given in triple space."""
    segment, orientation = _QUINTANT_SEGMENTS[origin_id * 5 + quintant]
    s = triple_to_s(Triple(x, y, z), hilbert_res, orientation)
    return serialize({'origin': origins[origin_id], 'segment': segment, 'S': s, 'resolution': resolution})



def cell_ids_to_triples(cell_ids: Iterable[int], out: Optional[List[int]] = None) -> List[int]:
    """
    Decode cell IDs (each at resolution 1 or above) into triple space, appending
    them to `out` as flat (origin_id, quintant, x, y, z).
    """
    if out is None:
        out = []
    for cell_id in cell_ids:
        cell = deserialize(cell_id)
        origin = cell['origin']
        quintant, orientation = segment_to_quintant(cell['segment'], origin)
        t = s_to_triple(cell['S'], cell['resolution'] - FIRST_HILBERT_RESOLUTION + 1, orientation)
        out.extend((origin.id, quintant, t.x, t.y, t.z))
    return out


def triple_cell_center(origin_id: int, quintant: int, x: int, y: int, z: int,
                       hilbert_res: int, max_row: int) -> Spherical:
    """The center of a cell given in triple space, on the sphere."""
    triple = Triple(x, y, z)
    face = get_pentagon_center(hilbert_res, quintant, triple, triple_flavor(triple, max_row))
    return _dodecahedron.inverse(face, origin_id)


# Receives a cell given in triple space: (origin_id, quintant, x, y, z)
TripleCellVisitor = Callable[[int, int, int, int, int], None]


def for_each_triple_neighbor(origin_id: int, quintant: int, x: int, y: int, z: int,
                             max_row: int, edge_only: bool, visit: TripleCellVisitor) -> None:
    """
    Visit every neighbor of a cell given in triple space: within its quintant the
    fixed per-flavor triple deltas, and, for a cell on a quintant edge (x = 0,
    z = 0 or y = max_row), the boundary delta tables. `edge_only` restricts to the
    5 edge-sharing neighbors; otherwise the vertex-only neighbors come too. A
    neighbor may be visited more than once; visitors deduplicate.
    """
    triple = Triple(x, y, z)

    # Within the quintant: the fixed per-flavor deltas
    flavor = triple_flavor(triple, max_row)
    deltas = NEIGHBOR_DELTAS[flavor].edge if edge_only else NEIGHBOR_DELTAS[flavor].all
    for d in deltas:
        neighbor = Triple(x + d.x, y + d.y, z + d.z)
        if triple_in_bounds(neighbor, max_row):
            visit(origin_id, quintant, neighbor.x, neighbor.y, neighbor.z)

    # Across a quintant edge: the boundary delta tables
    if x == 0 or z == 0 or y == max_row:
        _visit_boundary(origin_id, quintant, triple, max_row, edge_only, False, visit)


def walk_triple_cells(seeds: List[int], max_row: int, expand: Callable[[int, int, int, int, int], bool]) -> None:
    """
    Breadth-first walk from `seeds` (flat triples) through neighbors (edge and
    vertex, across quintant edges too): each cell reached is passed to `expand`
    once, and the walk continues from those it returns True for. The seeds count
    as reached but are not passed to `expand`.
    """
    visited = {triple_cell_key(*seeds[c:c + 5]) for c in range(0, len(seeds), 5)}
    frontier = seeds
    while frontier:
        next_frontier: List[int] = []

        def visit(origin_id: int, quintant: int, x: int, y: int, z: int) -> None:
            key = triple_cell_key(origin_id, quintant, x, y, z)
            if key in visited:
                return
            visited.add(key)
            if expand(origin_id, quintant, x, y, z):
                next_frontier.extend((origin_id, quintant, x, y, z))

        for c in range(0, len(frontier), 5):
            for_each_triple_neighbor(frontier[c], frontier[c + 1], frontier[c + 2], frontier[c + 3],
                                     frontier[c + 4], max_row, False, visit)
        frontier = next_frontier


def for_each_lattice_neighbor(origin_id: int, quintant: int, x: int, y: int, z: int,
                              max_row: int, visit: TripleCellVisitor) -> None:
    """
    Visit every lattice neighbor of a cell given in triple space: the 3
    parity-valid single-axis moves within its quintant (the connectivity
    `triple_space_flood_fill` floods by), and, for a cell on a quintant edge, its
    edge-sharing boundary neighbors -- but not the vertex corner, which the
    lattice moves don't traverse either. A neighbor may be visited more than
    once; visitors deduplicate.
    """
    # Within the quintant: +1 on one axis from a parity 0 triple, -1 from parity 1
    step = 1 if x + y + z == 0 else -1
    if triple_in_bounds(Triple(x + step, y, z), max_row):
        visit(origin_id, quintant, x + step, y, z)
    if triple_in_bounds(Triple(x, y + step, z), max_row):
        visit(origin_id, quintant, x, y + step, z)
    if triple_in_bounds(Triple(x, y, z + step), max_row):
        visit(origin_id, quintant, x, y, z + step)

    # Across a quintant edge: the boundary delta tables
    if x == 0 or z == 0 or y == max_row:
        _visit_boundary(origin_id, quintant, Triple(x, y, z), max_row, True, True, visit)


def _visit_boundary(origin_id: int, quintant: int, triple: Triple, max_row: int,
                    edge_only: bool, skip_corners: bool, visit: TripleCellVisitor) -> None:
    """Visit the neighbors of a cell on a quintant edge that lie across it."""
    boundary: List[int] = []
    get_boundary_neighbor_triples(triple, triple.x + triple.y + triple.z, quintant, origins[origin_id],
                                  max_row, edge_only, skip_corners, boundary)
    for i in range(0, len(boundary), 5):
        visit(boundary[i], boundary[i + 1], boundary[i + 2], boundary[i + 3], boundary[i + 4])


# The cell hierarchy in triple space. A cell's 4 children are 2*triple + the
# offsets for its flavor (each level of A5 refines the square grid R of
# g o^r D into 4); only their curve order depends on the orientation.
_CHILD_OFFSETS = [
    [(0, 0, 0), (0, 1, -1), (0, 1, 0), (0, 2, -1)],  # flavor 0
    [(-1, -1, 0), (-1, 0, -1), (-1, 0, 0), (-1, 1, -1)],  # flavor 1
    [(-1, 1, 0), (0, 0, 0), (0, 1, -1), (0, 1, 0)],  # flavor 2
    [(-1, 0, -1), (-1, 0, 0), (-1, 1, -1), (0, 0, -1)],  # flavor 3
]


def triple_children(origin_id: int, quintant: int, x: int, y: int, z: int,
                    max_row: int, out: List[int]) -> None:
    """The 4 children of a cell given in triple space (`max_row` is its own), appended to `out`."""
    for dx, dy, dz in _CHILD_OFFSETS[triple_flavor(Triple(x, y, z), max_row)]:
        out.extend((origin_id, quintant, 2 * x + dx, 2 * y + dy, 2 * z + dz))


def triple_parent(origin_id: int, quintant: int, x: int, y: int, z: int,
                  parent_max_row: int, out: List[int]) -> None:
    """
    The parent of a cell given in triple space (`parent_max_row` is the
    parent's), appended to `out`. The child's coordinates mod 2 fix
    child - 2*parent, but for two classes, where the two candidate parents
    differ in flavor -- and so, sharing x and z, in apex colour (see
    triple_flavor).

    Not used by the library: kept for completeness, as the inverse of
    `triple_children`, for traversals that coarsen in triple space.
    """
    dx = -(x & 1)
    dz = -(z & 1)
    dy = y & 1
    px = (x - dx) >> 1
    pz = (z - dz) >> 1
    colour = (parent_max_row + 1 + px + pz) & 1
    if dx == 0 and dy == 0 and dz == -1:
        dy = 2 if colour == 0 else 0  # flavor 0 or 3 parent
    if dx == -1 and dy == 1 and dz == 0:
        dy = 1 if colour == 1 else -1  # flavor 2 or 1 parent
    out.extend((origin_id, quintant, px, (y - dy) >> 1, pz))
