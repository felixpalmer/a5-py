"""
A5
SPDX-License-Identifier: Apache-2.0
Copyright (c) A5 contributors
"""

import math
from typing import List, Tuple

SQRT5 = math.sqrt(5)
INV_SQRT5 = math.sqrt(0.2)

# Dodecahedron face centers (origins) can be defined exactly using trigonometry
# The north and south poles are just at z=1 and z=-1
# Then there are two rings at z = ±INV_SQRT5, with radius 2 * INV_SQRT5

# Exact values for defining a regular pentagon (with radius 1). It is correct to use a radius
# of 1 as we want to obtain the axes of rotations, so the vectors need to be normalized.
# cos0 = 0;
# cos36 = (SQRT5 + 1) / 4;
# cos72 = (SQRT5 - 1) / 4;
# sin0 = 0;
# sin36 = Math.sqrt(10 - 2 * SQRT5) / 4;
# sin72 = Math.sqrt(10 + 2 * SQRT5) / 4;
#
# To compute the quaternion use the equation:
# q = [...sin(alpha) * axis, cos(alpha)]
# where alpha is the half-angle of rotation from the pole to the face center.

# Sin/cosine of half angle (alpha) of rotation from pole to first ring
# For the second ring sin -> cos and cos -> -sin by (pi / 2 - x) identities
sin_alpha = math.sqrt((1 - INV_SQRT5) / 2)
cos_alpha = math.sqrt((1 + INV_SQRT5) / 2)

# The resulting value simplify a set of expressions. It is much better to compute
# these directly than using trigonometry
A = 0.5  # sin72 * sinAlpha or sin36 * cosAlpha 
B = math.sqrt((2.5 - SQRT5) / 10)  # cos72 * sinAlpha 
C = math.sqrt((2.5 + SQRT5) / 10)  # cos36 * cosAlpha
D = math.sqrt((1 + INV_SQRT5) / 8)  # cos36 * sinAlpha
E = math.sqrt((1 - INV_SQRT5) / 8)  # cos72 * cosAlpha
F = math.sqrt((3 - SQRT5) / 8)  # sin36 * sinAlpha
G = math.sqrt((3 + SQRT5) / 8)  # sin72 * cosAlpha

# Face centers projected onto the z=0 plane & normalized
# 0: North pole,
# 1-5: First pentagon ring
# 6-10: Second pentagon ring
# 11: South pole
face_centers = [
    (0, 0),  # Doesn't actually matter as rotation is 0

    # First ring: five vertices, CCW, multiplied by sinAlpha
    (sin_alpha, 0),  # [cos0, sin0]
    (B, A),  # [cos72, sin72]
    (-D, F),  # [-cos36, sin36]
    (-D, -F),  # [-cos36, -sin36]
    (B, -A),  # [cos72, -sin72]

    # Second ring: the same five vertices but negated (180deg rotation), multiplied by cosAlpha
    (-cos_alpha, 0),  # [-cos0, -sin0]
    (-E, -G),  # [-cos72, -sin72]
    (C, -A),  # [cos36, -sin36]
    (C, A),  # [cos36, sin36]
    (-E, G),  # [-cos72, sin72]

    (0, 0)
]

# Obtain by cross product with the z-axis
axes = [(-y, x) for x, y in face_centers]

# Quaternions are obtained from axis of rotation & angle of rotation
quaternions: List[Tuple[float, float, float, float]] = []
for i, axis in enumerate(axes):
    if i == 0:
        quaternions.append((0, 0, 0, 1))
    elif i == 11:
        quaternions.append((0, -1, 0, 0))  # TODO better to use 1, 0, 0, 0?
    else:
        x, y = axis
        w = cos_alpha if i < 6 else sin_alpha
        quaternions.append((x, y, 0, w))