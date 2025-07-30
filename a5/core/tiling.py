# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

import math
from typing import List, Tuple
from ..geometry.pentagon import PentagonShape, Pentagon
from .pentagon import a, BASIS, PENTAGON, TRIANGLE, v, V, w
from .constants import TWO_PI, TWO_PI_OVER_5
from .hilbert import NO, Anchor, YES
from ..math import vec2

TRIANGLE_MODE = False

shift_right = w  # No need to copy, just reference
shift_left = (-w[0], -w[1])

# Define transforms for each pentagon in the primitive unit
# Using pentagon vertices and angle as the basis for the transform
QUINTANT_ROTATIONS = [
    (
        (math.cos(TWO_PI_OVER_5 * quintant), -math.sin(TWO_PI_OVER_5 * quintant)),
        (math.sin(TWO_PI_OVER_5 * quintant), math.cos(TWO_PI_OVER_5 * quintant))
    )
    for quintant in range(5)
]

def get_pentagon_vertices(resolution: int, quintant: int, anchor: Anchor) -> PentagonShape:
    """
    Get pentagon vertices
    
    Args:
        resolution: The resolution level
        quintant: The quintant index (0-4)
        anchor: The anchor information
        
    Returns:
        A pentagon shape with transformed vertices
    """
    pentagon = (TRIANGLE if TRIANGLE_MODE else PENTAGON).clone()
    
    # Matrix-vector multiplication using gl-matrix style: BASIS @ anchor.offset
    # Convert 2x2 matrix from ((a,b),(c,d)) to [a,c,b,d] (column-major)
    basis_flat = [BASIS[0][0], BASIS[1][0], BASIS[0][1], BASIS[1][1]]
    translation_vec = vec2.create()
    vec2.transformMat2(translation_vec, anchor.offset, basis_flat)
    translation = (translation_vec[0], translation_vec[1])

    # Apply transformations based on anchor properties
    if anchor.flips[0] == NO and anchor.flips[1] == YES:
        pentagon.rotate180()

    k = anchor.k
    F = anchor.flips[0] + anchor.flips[1]
    if (
        # Orient last two pentagons when both or neither flips are YES
        ((F == -2 or F == 2) and k > 1) or
        # Orient first & last pentagons when only one of flips is YES
        (F == 0 and (k == 0 or k == 3))
    ):
        pentagon.reflect_y()

    if anchor.flips[0] == YES and anchor.flips[1] == YES:
        pentagon.rotate180()
    elif anchor.flips[0] == YES:
        pentagon.translate(shift_left)
    elif anchor.flips[1] == YES:
        pentagon.translate(shift_right)

    # Position within quintant
    pentagon.translate(translation)
    pentagon.scale(1 / (2 ** resolution))
    pentagon.transform(QUINTANT_ROTATIONS[quintant])

    return pentagon

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