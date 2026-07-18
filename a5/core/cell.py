# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

import math
from typing import List, Tuple, Optional, Dict, TypedDict, Union
from .coordinate_systems import Cartesian, Face, LonLat, Spherical
from .coordinate_transforms import (
    face_to_ij, from_lonlat, to_cartesian, to_face, to_lonlat,
    to_spherical, to_polar, normalize_longitudes
)
from .origin import (
    find_nearest_origin,
    find_nearest_origins,
    quintant_to_segment,
    segment_to_quintant,
)
from ..projections.dodecahedron import DodecahedronProjection
from .utils import A5Cell, Origin, OriginId
from ..geometry.pentagon import PentagonShape
from .tiling import cell_margin_scaled, get_face_vertices, get_pentagon_center, get_pentagon_vertices, get_quintant_polar, get_quintant_vertices
from .constants import PI_OVER_5
from ..lattice import s_to_cell, triple_flavor, triple_in_bounds
from ..lattice.triple import triple_to_s
from ..lattice.curve import round_to_triple
from ..lattice.types import Triple
from .serialization import deserialize, serialize, FIRST_HILBERT_RESOLUTION, MAX_RESOLUTION, WORLD_CELL
from ..geometry.spherical_polygon import SphericalPolygonShape

# Reuse this object to avoid allocation
_dodecahedron = DodecahedronProjection()

# Single-entry cache of the most recent successful lookup. Speeds up
# dense-sample workloads (polygon boundary tracing, line tracing) where
# consecutive calls often land in the same cell. The cache stores the
# pre-computed pentagon + origin so the hit-test is just one projection
# + one pentagon containment check.
_last_result: Optional[Dict] = None


class CellToBoundaryOptions(TypedDict, total=False):
    """Options for cell_to_boundary function."""
    closed_ring: bool
    segments: Union[int, str]

def lonlat_to_cell(lon_lat: LonLat, resolution: int) -> int:
    """
    Convert longitude/latitude coordinates to a cell ID.

    Args:
        lon_lat: Tuple of (longitude, latitude) in degrees
        resolution: Resolution level of the cell

    Returns:
        Cell ID as a big integer
    """
    return spherical_to_cell(from_lonlat(lon_lat), resolution)


def spherical_to_cell(spherical: Spherical, resolution: int) -> int:
    """
    Like `lonlat_to_cell`, but accepts a point already in A5's internal spherical
    representation (rotated authalic frame, as produced by `from_lonlat` or
    `to_spherical(authalic_cartesian)`). Skips the redundant authalic
    inverse/forward round-trip in dense-sample loops where the input already
    comes from the authalic Cartesian space (e.g. polygon-fill boundary slerp).
    """
    # Resolution -1 represents WORLD_CELL, which covers the entire world
    if resolution == -1:
        return WORLD_CELL

    if resolution < FIRST_HILBERT_RESOLUTION:
        # For low resolutions there is no Hilbert curve: the cell is determined
        # by the face (and quintant) alone, so the lookup is exact.
        origin = find_nearest_origin(spherical)
        dodec_point = _dodecahedron.forward(spherical, origin.id)
        quintant = get_quintant_polar(to_polar(dodec_point))
        segment, _ = quintant_to_segment(quintant, origin)
        return serialize({'origin': origin, 'segment': segment, 'S': 0, 'resolution': resolution})

    # Try the cached pentagon first -- skips the full lookup when consecutive
    # calls land in the same cell (common in dense-sample loops).
    global _last_result
    if _last_result is not None and _last_result['resolution'] == resolution:
        projected = _dodecahedron.forward(spherical, _last_result['origin_id'])
        if _last_result['pentagon'].contains_point(projected) > 0:
            return _last_result['cell_id']

    # Fast path: locate the containing pentagon directly. Round to the leaf
    # triangle, get the closed-form flavor, and test the pentagon geometrically
    # in the scaled quintant frame; the triangular and pentagonal lattices are
    # not congruent, but the containing pentagon is always the triangle's cell
    # or one of its fixed neighbor deltas (verified exhaustively), so at most
    # one 7-candidate walk resolves it -- then a single curve encode.
    origin = find_nearest_origin(spherical)
    dodec_point = _dodecahedron.forward(spherical, origin.id)
    quintant = get_quintant_polar(to_polar(dodec_point))
    best = _lookup_in_quintant(dodec_point, origin, quintant, resolution)
    if best is not None and best[0] > 0:
        return _accept_candidate(best)
    # No strictly-containing pentagon in the assigned frame: the point sits on
    # a cell boundary or within float noise of a quintant/face seam.
    return _spherical_to_cell_boundary(spherical, resolution, origin.id, quintant, best)


# A candidate is the tuple (margin, cell_id, triple, flavor, quintant,
# hilbert_resolution, origin_id, resolution); margin > 0 iff the unique
# strictly-containing pentagon.
def _lookup_in_quintant(dodec_point, origin, quintant: int, resolution: int):
    """The best cell for `dodec_point` (face frame of `origin`) within one
    quintant: round to the leaf triangle, closed-form flavor, geometric
    margin, and -- when the triangle's cell doesn't strictly contain the
    point -- the best of its fixed neighbor deltas."""
    segment, orientation = quintant_to_segment(quintant, origin)

    # Res-30 ids can only encode quintants 0-41 (by design: 64 bits cannot fit
    # res 30 globally, so A5 covers the populous region). In the unsupported
    # quintants, answer at the finest representable resolution instead -- the
    # res-29 cell CONTAINING the point. (Previously the cap lived only in
    # serialize, which swapped in the res-29 parent of a res-30 search result --
    # a cell that fails to contain the query point ~44% of the time there.)
    if resolution == MAX_RESOLUTION and 5 * origin.id + (segment - origin.first_quintant + 5) % 5 > 41:
        resolution = MAX_RESOLUTION - 1

    px, py = dodec_point
    if quintant != 0:
        extra = 2 * PI_OVER_5 * quintant
        c, s = math.cos(-extra), math.sin(-extra)
        px, py = c * px - s * py, s * px + c * py
    hilbert_resolution = 1 + resolution - FIRST_HILBERT_RESOLUTION
    scale = 1 << hilbert_resolution
    px *= scale
    py *= scale
    ij = face_to_ij((px, py))

    base = round_to_triple(ij, hilbert_resolution)
    triple = base
    flavor = triple_flavor(base)
    margin = cell_margin_scaled(px, py, base.x, base.y, flavor)
    if margin <= 0:
        # All deltas are relative to the ROUNDED triple (the containing
        # pentagon is always among its fixed neighbors), not to intermediate
        # best cells.
        max_row = scale - 1
        for d in NEIGHBOR_DELTAS[flavor].all:
            neighbor = Triple(base.x + d.x, base.y + d.y, base.z + d.z)
            if not triple_in_bounds(neighbor, max_row):
                continue
            neighbor_flavor = triple_flavor(neighbor)
            neighbor_margin = cell_margin_scaled(px, py, neighbor.x, neighbor.y, neighbor_flavor)
            if neighbor_margin > margin:
                triple = neighbor
                flavor = neighbor_flavor
                margin = neighbor_margin
                if margin > 0:
                    break
    S = triple_to_s(triple, hilbert_resolution, orientation)
    if S is None:
        return None
    cell_id = serialize({'origin': origin, 'segment': segment, 'S': S, 'resolution': resolution})
    return (margin, cell_id, triple, flavor, quintant, hilbert_resolution, origin.id, resolution)


def _accept_candidate(c) -> int:
    """Cache the winning pentagon for the dense-sample fast accept and return its id."""
    global _last_result
    margin, cell_id, triple, flavor, quintant, hilbert_resolution, origin_id, resolution = c
    _last_result = {
        'cell_id': cell_id,
        'pentagon': get_pentagon_vertices(hilbert_resolution, quintant, triple, flavor),
        'origin_id': origin_id,
        'resolution': resolution,
    }
    return cell_id


# Tie margin tolerance: containment margins are cross products of unit-scale
# pentagon edges against coordinates of magnitude up to 2^hilbert_resolution,
# so their float noise is ~2^(hilbert_resolution - 52); 2^-44 gives a wide
# safety factor while staying geometrically negligible (cells are unit-size
# in the scaled frame).
_TIE_EPS = 2.0 ** -44


def _spherical_to_cell_boundary(spherical, resolution: int, first_origin_id, first_quintant: int, first) -> int:
    """Boundary resolution: the point has no strictly-containing pentagon in
    its assigned frame -- it lies on a cell edge, or within float noise of a
    quintant or face seam (where the containing cell belongs to a neighboring
    frame). Deterministically rerun the same lookup in every frame that could
    own the point -- all 5 quintants of the 3 nearest faces (a dodecahedron
    vertex joins 3 faces; a face center joins 5 quintants). A strictly-
    containing pentagon is unique, so the first strict hit wins; if none
    exists the point is exactly on a boundary shared by the near-best
    candidates, and the tie-break is the cell that comes FIRST ALONG THE CURVE
    -- the lowest cell id (origin/segment occupy the top id bits in curve
    order, so numeric order is curve order globally)."""
    candidates = [first] if first is not None else []
    for origin in find_nearest_origins(spherical, 3):
        dodec_point = _dodecahedron.forward(spherical, origin.id)
        # Try this origin's assigned quintant first, then its gamma-adjacent
        # neighbors: seam points resolve in the adjacent frame, so this order
        # finds the strict container in 1-2 lookups instead of scanning all 5.
        q0 = get_quintant_polar(to_polar(dodec_point))
        for dq in (0, 1, 4, 2, 3):
            quintant = (q0 + dq) % 5
            if origin.id == first_origin_id and quintant == first_quintant:
                continue
            c = _lookup_in_quintant(dodec_point, origin, quintant, resolution)
            if c is None:
                continue
            if c[0] > 0:
                return _accept_candidate(c)
            candidates.append(c)
    if not candidates:
        raise ValueError('spherical_to_cell: no candidate cell found')
    best = max(c[0] for c in candidates)
    eps = _TIE_EPS * (1 << (1 + resolution - FIRST_HILBERT_RESOLUTION))
    winner = min((c for c in candidates if c[0] >= best - eps), key=lambda c: c[1])
    return _accept_candidate(winner)


def _get_pentagon(cell: A5Cell) -> PentagonShape:
    """
    Get the pentagon shape for a given cell.
    
    Args:
        cell: A5Cell object
        
    Returns:
        PentagonShape object
    """
    quintant, orientation = segment_to_quintant(cell["segment"], cell["origin"])
    if cell["resolution"] == (FIRST_HILBERT_RESOLUTION - 1):
        out = get_quintant_vertices(quintant)
        return out
    elif cell["resolution"] == (FIRST_HILBERT_RESOLUTION - 2):
        return get_face_vertices()

    hilbert_resolution = cell["resolution"] - FIRST_HILBERT_RESOLUTION + 1
    cell_geom = s_to_cell(cell["S"], hilbert_resolution, orientation)
    return get_pentagon_vertices(hilbert_resolution, quintant, cell_geom.triple, cell_geom.flavor)

def cell_to_spherical(cell_id: int) -> Spherical:
    """
    Convert a cell ID to spherical coordinates.

    Args:
        cell_id: Cell ID as a big integer

    Returns:
        Spherical coordinates (theta, phi)
    """
    cell = deserialize(cell_id)
    if cell["resolution"] >= FIRST_HILBERT_RESOLUTION:
        # Fast path: the pentagon center is O(1) from (triple, flavor) -- no need
        # to construct the pentagon itself.
        quintant, orientation = segment_to_quintant(cell["segment"], cell["origin"])
        hilbert_resolution = cell["resolution"] - FIRST_HILBERT_RESOLUTION + 1
        cell_geom = s_to_cell(cell["S"], hilbert_resolution, orientation)
        center = get_pentagon_center(hilbert_resolution, quintant, cell_geom.triple, cell_geom.flavor)
        return _dodecahedron.inverse(center, cell["origin"].id)
    pentagon = _get_pentagon(cell)
    return _dodecahedron.inverse(pentagon.get_center(), cell["origin"].id)


def cell_to_lonlat(cell_id: int) -> LonLat:
    """
    Convert a cell ID to longitude/latitude coordinates.

    Args:
        cell_id: Cell ID as a big integer

    Returns:
        Tuple of (longitude, latitude) in degrees
    """
    # WORLD_CELL represents the entire world, return (0, 0) as a reasonable default
    if cell_id == WORLD_CELL:
        return (0.0, 0.0)

    return to_lonlat(cell_to_spherical(cell_id))

def cell_to_boundary(
    cell_id: int,
    options: Optional[CellToBoundaryOptions] = None
) -> List[LonLat]:
    """
    Get the boundary coordinates of a cell.

    Args:
        cell_id: Cell ID as a big integer
        options: Dictionary with optional parameters:
            - closed_ring: Pass True to close the ring with the first point (default True)
            - segments: Number of segments to use for each edge. Pass 'auto' to use the resolution of the cell (default 'auto')

    Returns:
        List of (longitude, latitude) coordinates forming the cell boundary
    """
    # WORLD_CELL represents the entire world and is unbounded
    if cell_id == WORLD_CELL:
        return []

    if options is None:
        options = {}

    closed_ring = options.get('closed_ring', True)
    segments = options.get('segments', 'auto')

    cell = deserialize(cell_id)
    if segments == 'auto' or segments is None:
        segments = max(1, 2 ** (6 - cell["resolution"]))

    pentagon = _get_pentagon(cell)

    # Split each edge into segments before projection
    # Important to do before projection to obtain equal area cells
    split_pentagon = pentagon.split_edges(segments)
    vertices = split_pentagon.get_vertices()

    # Unproject to obtain lon/lat coordinates. Fused loop avoids the
    # intermediate unprojected_vertices allocation.
    boundary: List[LonLat] = [None] * len(vertices)
    for i in range(len(vertices)):
        boundary[i] = to_lonlat(_dodecahedron.inverse(vertices[i], cell["origin"].id))

    # Normalize longitudes to handle antimeridian crossing
    normalized_boundary = normalize_longitudes(boundary)

    if closed_ring:
        normalized_boundary.append(normalized_boundary[0])
    
    # TODO: This is a patch to make the boundary CCW, but we should fix the winding order of the pentagon
    # throughout the whole codebase
    normalized_boundary.reverse()
    return normalized_boundary

def a5cell_contains_point(cell: A5Cell, spherical: Spherical) -> float:
    """
    Check if a spherical point is contained within a cell.

    Args:
        cell: A5Cell object
        spherical: Spherical coordinates in A5's internal rotated authalic frame

    Returns:
        Positive number if the point is contained within the cell, negative otherwise
    """
    pentagon = _get_pentagon(cell)
    projected_point = _dodecahedron.forward(spherical, cell['origin'].id)

    return pentagon.contains_point(projected_point)


def cell_intersects_segment(cell_id: int, a: LonLat, b: LonLat) -> bool:
    """
    Tests whether the segment between two LonLat points intersects a cell.

    The test runs entirely in the cell's Face coordinate system: both endpoints
    are projected via the dodecahedron projection (with face-plane extension for
    points beyond the face's edge), then checked against the pentagon's straight
    2D edges. The segment is treated as a 2D straight line in Face coords --
    accurate when the segment is short relative to the face (DSEA distortion is
    negligible at sub-cell scales).
    """
    if cell_id == WORLD_CELL:
        return True
    cell = deserialize(cell_id)
    pentagon = _get_pentagon(cell)
    a_face = _dodecahedron.forward(from_lonlat(a), cell['origin'].id)
    b_face = _dodecahedron.forward(from_lonlat(b), cell['origin'].id)
    return pentagon.intersects_segment(a_face, b_face)


# Imported last to break the circular import with the traversal package
# (traversal.cap imports cell_to_spherical from this module).
from ..traversal.neighbors import NEIGHBOR_DELTAS  # noqa: E402
