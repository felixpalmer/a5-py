"""
A5
SPDX-License-Identifier: Apache-2.0
Copyright (c) A5 contributors
"""

import numpy as np
from typing import List, cast
from .types import Degrees, Face, Vec2
from .constants import distance_to_edge, PI_OVER_10, PI_OVER_5

class PentagonShape:
    """A pentagon shape defined by its vertices."""
    def __init__(self, vertices: List[Face]):
        self.vertices = vertices

# Pentagon vertex angles
A = cast(Degrees, 72.0)
B = cast(Degrees, 127.94543761193603)
C = cast(Degrees, 108.0)
D = cast(Degrees, 82.29202980963508)
E = cast(Degrees, 149.7625318412527)

# Initialize vertices
a = cast(Face, np.array([0.0, 0.0], dtype=np.float64))
b = cast(Face, np.array([0.0, 1.0], dtype=np.float64))
# c & d calculated by circle intersections. Perhaps can obtain geometrically.
c = cast(Face, np.array([0.7885966681787006, 1.6149108024237764], dtype=np.float64))
d = cast(Face, np.array([1.6171013659387945, 1.054928690397459], dtype=np.float64))
e = cast(Face, np.array([np.cos(PI_OVER_10), np.sin(PI_OVER_10)], dtype=np.float64))

# Distance to edge midpoint
edge_midpoint_d = 2 * np.linalg.norm(c) * np.cos(PI_OVER_5)

# Lattice growth direction is AC, want to rotate it so that it is parallel to x-axis
BASIS_ROTATION = PI_OVER_5 - np.arctan2(c[1], c[0])  # -27.97 degrees

# Scale to match unit sphere
scale = 2 * distance_to_edge / edge_midpoint_d

# Apply transformations to vertices
for v in [a, b, c, d, e]:
    v *= scale
    # Rotate around origin
    cos_rot = np.cos(BASIS_ROTATION)
    sin_rot = np.sin(BASIS_ROTATION)
    x, y = v
    v[0] = x * cos_rot - y * sin_rot
    v[1] = x * sin_rot + y * cos_rot

"""
Definition of pentagon used for tiling the plane.
While this pentagon is not equilateral, it forms a tiling with 5 fold
rotational symmetry and thus can be used to tile a regular pentagon.
"""
PENTAGON = PentagonShape([a, b, c, d, e])

bisector_angle = np.arctan2(c[1], c[0]) - PI_OVER_5

# Define triangle also, as UVW
u = cast(Face, np.array([0.0, 0.0], dtype=np.float64))
L = distance_to_edge / np.cos(PI_OVER_5)

V = bisector_angle + PI_OVER_5
v = cast(Face, np.array([L * np.cos(V), L * np.sin(V)], dtype=np.float64))

W = bisector_angle - PI_OVER_5
w = cast(Face, np.array([L * np.cos(W), L * np.sin(W)], dtype=np.float64))
TRIANGLE = PentagonShape([u, v, w, w, w])  # TODO hacky, don't pretend this is pentagon

"""
Basis vectors used to layout primitive unit
"""
BASIS = np.array([
    [v[0], v[1]],
    [w[0], w[1]]
], dtype=np.float64)

BASIS_INVERSE = np.linalg.inv(BASIS)

__all__ = [
    'A', 'B', 'C', 'D', 'E',
    'a', 'b', 'c', 'd', 'e',
    'PENTAGON',
    'u', 'v', 'w', 'V',
    'TRIANGLE',
    'BASIS', 'BASIS_INVERSE'
] 