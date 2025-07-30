"""
A5
SPDX-License-Identifier: Apache-2.0
Copyright (c) A5 contributors
"""

import math
from typing import List, cast
from .coordinate_systems import Degrees, Face, Vec2
from .constants import distance_to_edge, PI_OVER_10, PI_OVER_5
from ..geometry.pentagon import PentagonShape

# Pentagon vertex angles
A = cast(Degrees, 72.0)
B = cast(Degrees, 127.94543761193603)
C = cast(Degrees, 108.0)
D = cast(Degrees, 82.29202980963508)
E = cast(Degrees, 149.7625318412527)

# Initialize vertices
a = cast(Face, (0.0, 0.0))
b = cast(Face, (0.0, 1.0))
# c & d calculated by circle intersections. Perhaps can obtain geometrically.
c = cast(Face, (0.7885966681787006, 1.6149108024237764))
d = cast(Face, (1.6171013659387945, 1.054928690397459))
e = cast(Face, (math.cos(PI_OVER_10), math.sin(PI_OVER_10)))

# Distance to edge midpoint
c_norm = math.sqrt(c[0] * c[0] + c[1] * c[1])
edge_midpoint_d = 2 * c_norm * math.cos(PI_OVER_5)

# Lattice growth direction is AC, want to rotate it so that it is parallel to x-axis
BASIS_ROTATION = PI_OVER_5 - math.atan2(c[1], c[0])  # -27.97 degrees

# Scale to match unit sphere
scale = 2 * distance_to_edge / edge_midpoint_d

# Apply transformations to vertices
def transform_vertex(vertex: Face, scale: float, rotation: float) -> Face:
    """Apply scale and rotation to a vertex."""
    # Scale
    scaled_x = vertex[0] * scale
    scaled_y = vertex[1] * scale
    
    # Rotate around origin
    cos_rot = math.cos(rotation)
    sin_rot = math.sin(rotation)
    
    return cast(Face, (
        scaled_x * cos_rot - scaled_y * sin_rot,
        scaled_x * sin_rot + scaled_y * cos_rot
    ))

# Apply transformations
a = transform_vertex(a, scale, BASIS_ROTATION)
b = transform_vertex(b, scale, BASIS_ROTATION)
c = transform_vertex(c, scale, BASIS_ROTATION)
d = transform_vertex(d, scale, BASIS_ROTATION)
e = transform_vertex(e, scale, BASIS_ROTATION)

"""
Definition of pentagon used for tiling the plane.
While this pentagon is not equilateral, it forms a tiling with 5 fold
rotational symmetry and thus can be used to tile a regular pentagon.
"""
PENTAGON = PentagonShape([a, b, c, d, e])

bisector_angle = math.atan2(c[1], c[0]) - PI_OVER_5

# Define triangle also, as UVW
u = cast(Face, (0.0, 0.0))
L = distance_to_edge / math.cos(PI_OVER_5)

V = bisector_angle + PI_OVER_5
v = cast(Face, (L * math.cos(V), L * math.sin(V)))

W = bisector_angle - PI_OVER_5
w = cast(Face, (L * math.cos(W), L * math.sin(W)))
TRIANGLE = PentagonShape([u, v, w])

"""
Basis vectors used to layout primitive unit
"""
# Basis matrix represented as nested tuples
BASIS = (
    (v[0], w[0]),
    (v[1], w[1])
)

# Calculate matrix inverse manually for 2x2 matrix
# For matrix [[a, b], [c, d]], inverse is [[d, -b], [-c, a]] / (ad - bc)
det = BASIS[0][0] * BASIS[1][1] - BASIS[0][1] * BASIS[1][0]
BASIS_INVERSE = (
    (BASIS[1][1] / det, -BASIS[0][1] / det),
    (-BASIS[1][0] / det, BASIS[0][0] / det)
)

__all__ = [
    'A', 'B', 'C', 'D', 'E',
    'a', 'b', 'c', 'd', 'e',
    'PENTAGON',
    'u', 'v', 'w', 'V',
    'TRIANGLE',
    'BASIS', 'BASIS_INVERSE'
]