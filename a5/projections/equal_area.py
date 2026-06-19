# Adaptation of icoVertexGreatCircle.ec from DGGAL project
# BSD 3-Clause License
# 
# Copyright (c) 2014-2025, Ecere Corporation
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# BSD 3-Clause License
# Copyright (c) 2024, A5 Project Contributors
# All rights reserved.

"""
A5
SPDX-License-Identifier: Apache-2.0
Copyright (c) A5 contributors
"""

import math
from typing import cast, List
from ..core.coordinate_systems import Cartesian, Face, Barycentric, FaceTriangle, SphericalTriangle
from ..core.coordinate_transforms import face_to_barycentric, barycentric_to_face
from ..geometry.spherical_polygon import spherical_triangle_area
from ..math import vec3


# Equal area projection originally described by:
# Snyder92 (AN EQUAL-AREA MAP PROJECTION FOR POLYHEDRAL GLOBES)
# Closed form equations due to Brenton R. S. Recht
#
# The projection maps a point V within a spherical triangle ABC onto a planar
# point F (within a planar triangle), in an equal-area-preserving manner.
#
# The first point of the triangle (A) is known as the radiating vertex and the
# choice of this vertex subtly modifies how the projection behaves. All three
# choices will yield an equal area projection, but the cusps will vary.
#
# The transformation is done via an intermediate point P, which is obtained by
# intersecting the two great circles formed by A&V and B&C (hence why A is special)
#
# The equal-area transformation is then done by computing the ratio of areas between
# triangles ABP & ABC
#
# The inverse follows the reverse procedure of obtaining P from the face triangle
# by inverting the equal-area transformation, before slerping between A&P to obtain V
class EqualAreaProjection:
    def __init__(self, canonical_triangle: SphericalTriangle):
        # By assuming that the geometry of the spherical triangle used is constant
        # (up to rotations on a sphere), a number of constants can be precomputed
        # and reused.
        self._constants = EqualAreaProjection.compute_constants(canonical_triangle)

    @staticmethod
    def compute_constants(spherical_triangle: SphericalTriangle) -> dict:
        A, B, C = spherical_triangle
        BxC = vec3.create()
        vec3.cross(BxC, B, C)
        AdotB = vec3.dot(A, B)
        AdotC = vec3.dot(A, C)
        BdotC = vec3.dot(B, C)

        V = vec3.dot(A, BxC)
        P = AdotC + BdotC
        Q = AdotB + 1
        R = AdotB * BdotC - AdotC
        F = P * P - Q * Q
        G = 2 * Q * R
        # Affine transform [N1, N2] of [cos(alpha), sin(alpha)]. Stored in
        # gl-matrix mat2d order [a, b, c, d, tx, ty] = [c1c, c2c, c1s, c2s, c1o, c2o]:
        # the cos/sin coefficients form the 2x2 block, the constants the translation.
        alpha_transform = [V * V - F, -G, -2 * V * P, 2 * V * Q, V * V + F, G]

        return {
            'AdotB': AdotB,  # A · B — the canonical ("even") B-C orientation
            'AdotC': AdotC,  # A · C — the mirror ("odd") orientation, B and C swapped
            'alphaTransform': alpha_transform,
            'area_abc': spherical_triangle_area(A, B, C),
            'volumeABC': V,  # A · (B × C) — signed triple product
        }

    def forward(self, V: Cartesian, spherical_triangle: SphericalTriangle, face_triangle: FaceTriangle) -> Face:
        """
        Forward projection: converts a spherical point to face coordinates

        Args:
            V: The spherical point to project
            spherical_triangle: The spherical triangle vertices
            face_triangle: The face triangle vertices

        Returns:
            The face coordinates
        """
        A, B, C = spherical_triangle
        area_abc = self._constants['area_abc']
        volume_abc = self._constants['volumeABC']

        # Compute point P, where great circles through A&V and B&C intersect
        BxC = vec3.create()
        vec3.cross(BxC, B, C)
        volume_vbc = vec3.dot(V, BxC)
        P = vec3.create()
        vec3.scale(P, V, volume_abc)
        vec3.scaleAndAdd(P, P, A, -volume_vbc)
        D = vec3.length(P)
        ooD = 1 / D if D > 0 else 1
        vec3.scale(P, P, ooD)
        P = cast(Cartesian, (P[0], P[1], P[2]))

        # Obtain rho & alpha by ratio of areas
        area_abp = max(0, spherical_triangle_area(A, B, P))
        alpha = area_abp / area_abc
        rho = (D / volume_abc) * math.sqrt((1 + vec3.dot(A, P)) / (1 + vec3.dot(A, V)))

        # Construct barycentric triangle and map to face
        b = cast(Barycentric, (1 - rho, rho * (1 - alpha), rho * alpha))
        return barycentric_to_face(b, face_triangle)

    def inverse(self, face_point: Face, face_triangle: FaceTriangle, spherical_triangle: SphericalTriangle) -> Cartesian:
        """
        Inverse projection: converts face coordinates back to spherical coordinates

        Args:
            face_point: The face coordinates
            face_triangle: The face triangle vertices
            spherical_triangle: The spherical triangle vertices

        Returns:
            The spherical coordinates
        """
        A, B, C = spherical_triangle
        b = face_to_barycentric(face_point, face_triangle)

        threshold = 1 - 1e-14
        if b[0] > threshold:
            return A
        if b[1] > threshold:
            return B
        if b[2] > threshold:
            return C

        # Normalize odd (mirror-image) triangles to the canonical even orientation
        # by swapping B↔C and the matching weight b1↔b2, so alphaTransform is correct
        constants = self._constants
        area_abc = constants['area_abc']
        m = constants['alphaTransform']
        face_AdotB = vec3.dot(A, B)
        odd = abs(face_AdotB - constants['AdotB']) > abs(face_AdotB - constants['AdotC'])
        B_n = C if odd else B
        C_n = B if odd else C
        b2 = b[1] if odd else b[2]

        # Obtain rho & alpha
        rho = 1 - b[0]
        alpha = (b2 / rho) * area_abc

        # Inverse to obtain point P (see forward). weight_bc = alphaTransform * [cos, sin]
        cos_a = math.cos(alpha)
        sin_a = math.sin(alpha)
        weight_b = m[0] * cos_a + m[2] * sin_a + m[4]
        weight_c = m[1] * cos_a + m[3] * sin_a + m[5]
        P = vec3.create()
        vec3.scale(P, B_n, weight_b)
        vec3.scaleAndAdd(P, P, C_n, weight_c)
        vec3.normalize(P, P)
        P = cast(Cartesian, (P[0], P[1], P[2]))

        # Compute weights for A & P
        s = vec3.dot(A, P)
        t = 1 + rho * rho * (s - 1)
        weight_p = rho * math.sqrt((1 + t) / (1 + s))
        weight_a = t - s * weight_p

        out = vec3.create()
        vec3.scale(out, A, weight_a)
        vec3.scaleAndAdd(out, out, P, weight_p)
        return cast(Cartesian, (out[0], out[1], out[2]))