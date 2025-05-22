"""
A5
SPDX-License-Identifier: Apache-2.0
Copyright (c) A5 contributors
"""

import math
from typing import cast
from .coordinate_systems import Radians

# Golden ratio
PHI = (1 + math.sqrt(5)) / 2

# Basic angle constants
TWO_PI = cast(Radians, 2 * math.pi)
TWO_PI_OVER_5 = cast(Radians, 2 * math.pi / 5)
PI_OVER_5 = cast(Radians, math.pi / 5)
PI_OVER_10 = cast(Radians, math.pi / 10)

# Angles between faces
dihedral_angle = cast(Radians, 2 * math.atan(PHI))  # Angle between pentagon faces (radians) = 116.565°
interhedral_angle = cast(Radians, math.pi - dihedral_angle)  # Angle between pentagon faces (radians) = 63.435°
face_edge_angle = cast(Radians, -0.5 * math.pi + math.acos(-1 / math.sqrt(3 - PHI)))  # = 58.28252558853899

# Distance from center to edge of pentagon face
distance_to_edge = PHI - 1
# TODO cleaner derivation?
distance_to_vertex = distance_to_edge / math.cos(PI_OVER_5)

# Warping parameters
WARP_FACTOR = 0.515  # Set to near 0 to disable warping

# Dodecahedron sphere radii (normalized to unit radius for inscribed sphere)
"""
Radius of the inscribed sphere in dodecahedron
"""
R_INSCRIBED = 1.0

"""
Radius of the sphere that touches the dodecahedron's edge midpoints
"""
R_MIDEDGE = math.sqrt(3 - PHI)

"""
Radius of the circumscribed sphere for dodecahedron
"""
R_CIRCUMSCRIBED = math.sqrt(3) * R_MIDEDGE / PHI

__all__ = [
    'PHI',
    'TWO_PI',
    'TWO_PI_OVER_5',
    'PI_OVER_5',
    'PI_OVER_10',
    'dihedral_angle',
    'interhedral_angle',
    'face_edge_angle',
    'distance_to_edge',
    'distance_to_vertex',
    'WARP_FACTOR',
    'R_INSCRIBED',
    'R_MIDEDGE',
    'R_CIRCUMSCRIBED'
] 