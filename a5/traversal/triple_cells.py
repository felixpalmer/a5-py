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
