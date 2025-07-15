# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

import numpy as np
from typing import List, Tuple
from .utils import Pentagon
from ..geometry.pentagon import PentagonShape
from .pentagon import a, BASIS, PENTAGON, TRIANGLE, v, V, w
from .constants import TWO_PI, TWO_PI_OVER_5
from .hilbert import NO, Anchor, YES

TRIANGLE_MODE = False

shift_right = w.copy()
shift_left = -w

# Define transforms for each pentagon in the primitive unit
# Using pentagon vertices and angle as the basis for the transform
QUINTANT_ROTATIONS = [
    np.array([[np.cos(TWO_PI_OVER_5 * quintant), -np.sin(TWO_PI_OVER_5 * quintant)],
              [np.sin(TWO_PI_OVER_5 * quintant), np.cos(TWO_PI_OVER_5 * quintant)]])
    for quintant in range(5)
]

translation = np.zeros(2)

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
    
    translation[:] = np.dot(BASIS, anchor.offset)

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
        pentagon.reflectY()

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
        vertices.append(np.dot(rotation, v))
    return PentagonShape(vertices)

def get_quintant(point: np.ndarray) -> int:
    # TODO perhaps quicker way without trigonometry
    angle = np.arctan2(point[1], point[0])
    normalized_angle = (angle - V + TWO_PI) % TWO_PI
    return int(np.ceil(normalized_angle / TWO_PI_OVER_5) % 5) 