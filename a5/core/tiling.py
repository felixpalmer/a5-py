# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

import math
from typing import List, Tuple
from ..geometry.pentagon import PentagonShape, Pentagon
from .pentagon import a, BASIS, PENTAGON, TRIANGLE, v, V, w
from .constants import TWO_PI, TWO_PI_OVER_5
from ..lattice import Triple
from ..math import vec2

TRIANGLE_MODE = False

# Define transforms for each pentagon in the primitive unit
# Using pentagon vertices and angle as the basis for the transform
QUINTANT_ROTATIONS = [
    (
        (math.cos(TWO_PI_OVER_5 * quintant), -math.sin(TWO_PI_OVER_5 * quintant)),
        (math.sin(TWO_PI_OVER_5 * quintant), math.cos(TWO_PI_OVER_5 * quintant))
    )
    for quintant in range(5)
]

# Center of the base PENTAGON under each flavor's orientation ops. The vertex
# mean is linear, so an oriented pentagon's center is the transformed base
# center -- no need to construct the five vertices when only the center is
# wanted (see get_pentagon_center).
def _flavor_centers() -> List[Tuple[float, float]]:
    centers = []
    for flavor in range(4):
        p = PENTAGON.clone()
        if flavor & 1:
            p.rotate180()
        if flavor & 2:
            p.reflect_y()
        centers.append(p.get_center())
    return centers


FLAVOR_CENTERS = _flavor_centers()


def _basis_translation(ref_ij: Tuple[float, float]) -> Tuple[float, float]:
    """BASIS @ ref_ij (gl-matrix column-major convention)."""
    basis_flat = [BASIS[0][0], BASIS[1][0], BASIS[0][1], BASIS[1][1]]
    out = vec2.create()
    vec2.transformMat2(out, ref_ij, basis_flat)
    return (out[0], out[1])


# The base PENTAGON under each flavor's orientation ops, flattened to
# (x0, y0, ..., x4, y4) for the allocation-free containment test below.
def _flavor_pentagons_flat():
    out = []
    for flavor in range(4):
        p = PENTAGON.clone()
        if flavor & 1:
            p.rotate180()
        if flavor & 2:
            p.reflect_y()
        verts = p.get_vertices()
        out.append(tuple(c for v in verts for c in (v[0], v[1])))
    return out


_FLAVOR_PENTAGONS = _flavor_pentagons_flat()
_B00, _B01, _B10, _B11 = BASIS[0][0], BASIS[0][1], BASIS[1][0], BASIS[1][1]


def cell_contains_scaled(px: float, py: float, x: int, y: int, flavor: int) -> bool:
    """
    Strict containment of a point in the pentagon of (triple, flavor), tested
    in the SCALED quintant-0 frame (face coords rotated into quintant 0 and
    scaled by 2^resolution -- the frame `_face_to_estimate` works in). In this
    frame the cell's pentagon is the flavor-oriented base pentagon translated
    by BASIS @ (x+y, -x+(flavor&1)), so the test needs no curve decode, no
    re-projection, and -- pentagons being unit-size here -- stays
    well-conditioned at every resolution.
    """
    rx = x + y
    ry = -x + (flavor & 1)
    tx = _B00 * rx + _B01 * ry
    ty = _B10 * rx + _B11 * ry
    pent = _FLAVOR_PENTAGONS[flavor]
    for i in range(5):
        j = 0 if i == 4 else i + 1
        v1x = pent[i * 2] + tx
        v1y = pent[i * 2 + 1] + ty
        v2x = pent[j * 2] + tx
        v2y = pent[j * 2 + 1] + ty
        # (v1 - v2) x (p - v1) < 0 => strictly outside this edge
        if (v1x - v2x) * (py - v1y) - (v1y - v2y) * (px - v1x) < 0:
            return False
    return True


def get_pentagon_vertices(resolution: int, quintant: int, triple: Triple, flavor: int) -> PentagonShape:
    """
    Get pentagon vertices for a cell.

    A cell's pentagon is one of exactly FOUR orientations of the base PENTAGON
    (the Cairo-like metatile): flavor bit 0 is a 180 deg rotation, bit 1 a Y
    reflection. The oriented pentagon sits at the triple-derived lattice point
    ref = (x+y, -x) in IJ, shifted by one j unit for the rotated flavors.
    The flavor is a 1:1 function of the cell's L-system jigsaw piece and is
    produced by the descent (s_to_cell); the placement was derived and verified
    exhaustively against the pentagon geometry.

    Args:
        resolution: The resolution level
        quintant: The quintant index (0-4)
        triple: The cell's triple coordinates
        flavor: The cell's pentagon flavor (0-3)

    Returns:
        A pentagon shape with transformed vertices
    """
    pentagon = (TRIANGLE if TRIANGLE_MODE else PENTAGON).clone()

    if flavor & 1:
        pentagon.rotate180()
    if flavor & 2:
        pentagon.reflect_y()

    # Position within quintant: ref(triple), plus (0, 1) for the rotated flavors
    translation = _basis_translation((triple.x + triple.y, -triple.x + (flavor & 1)))
    pentagon.translate(translation)
    pentagon.scale(1 / (2 ** resolution))
    pentagon.transform(QUINTANT_ROTATIONS[quintant])

    return pentagon


def get_pentagon_center(resolution: int, quintant: int, triple: Triple, flavor: int) -> Tuple[float, float]:
    """
    The center of a cell's pentagon, without constructing the pentagon --
    O(1) via the precomputed flavor centers. Equivalent to
    get_pentagon_vertices(...).get_center() (up to float associativity).
    """
    c = FLAVOR_CENTERS[flavor]
    translation = _basis_translation((triple.x + triple.y, -triple.x + (flavor & 1)))
    scale = 2 ** resolution
    ox = (c[0] + translation[0]) / scale
    oy = (c[1] + translation[1]) / scale
    rot = QUINTANT_ROTATIONS[quintant]
    return (rot[0][0] * ox + rot[0][1] * oy, rot[1][0] * ox + rot[1][1] * oy)


def get_quintant_vertices(quintant: int) -> PentagonShape:
    triangle = TRIANGLE.clone()
    triangle.transform(QUINTANT_ROTATIONS[quintant])
    return triangle

def get_face_vertices() -> PentagonShape:
    vertices = []
    for rotation in QUINTANT_ROTATIONS:
        # Matrix-vector multiplication using gl-matrix style: rotation @ v
        # Convert 2x2 matrix from ((a,b),(c,d)) to [a,c,b,d] (column-major)
        rotation_flat = [rotation[0][0], rotation[1][0], rotation[0][1], rotation[1][1]]
        vertex_vec = vec2.create()
        vec2.transformMat2(vertex_vec, v, rotation_flat)
        new_vertex = (vertex_vec[0], vertex_vec[1])
        vertices.append(new_vertex)
    return PentagonShape(vertices)

def get_quintant_polar(polar: Tuple[float, float]) -> int:
    """
    Determines which quintant a polar coordinate belongs to.
    
    Args:
        polar: Polar coordinates (r, theta)
        
    Returns:
        Quintant index (0-4)
    """
    rho, gamma = polar
    return (round(gamma / TWO_PI_OVER_5) + 5) % 5