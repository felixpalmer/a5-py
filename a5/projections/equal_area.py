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
from typing import cast, Dict
from ..core.coordinate_systems import Cartesian, Face, Barycentric, FaceTriangle, SphericalTriangle
from ..core.coordinate_transforms import face_to_barycentric, barycentric_to_face
from ..geometry.spherical_polygon import spherical_triangle_area
from ..math import vec3, quat

class EqualAreaProjection:
    """
    Equal area projection using Slice & Dice algorithm

    Caches the shape-only invariants of the spherical triangle. A5 only ever
    projects the congruent face-triangles of a single dodecahedron, so these
    depend only on the triangle's shape, not its position — they are computed
    once from the canonical triangle passed to the constructor and reused for
    every projection. Deriving them eagerly from a fixed triangle (rather than
    lazily from whichever triangle is projected first) keeps results
    independent of call order: congruent triangles agree only to ~1 ulp, so a
    lazy cache would make outputs depend on process history.

    NOTE: `V` is a *signed* triple product, so this caching is only valid while
    every triangle shares the same winding (chirality). DodecahedronProjection
    guarantees this by ordering vertices consistently across normal and
    reflected faces. A · B and C · A are deliberately NOT cached — even/odd
    face triangles swap the roles of the B and C vertices, so those dot
    products differ by ~0.056 between triangles and must be computed per call.
    """

    def __init__(self, canonical_triangle: SphericalTriangle):
        self._constants = EqualAreaProjection.compute_constants(canonical_triangle)

    @staticmethod
    def compute_constants(spherical_triangle: SphericalTriangle) -> Dict[str, float]:
        A, B, C = spherical_triangle
        c1 = vec3.create()
        vec3.cross(c1, B, C)
        c12 = vec3.dot(B, C)
        return {
            'V': vec3.dot(A, c1),  # A · (B × C) — signed triple product
            'c12': c12,  # B · C
            's12': vec3.length(c1),  # |B × C|
            'kQ': 2 / math.acos(c12),
            'area_abc': spherical_triangle_area(A, B, C)
        }

    def forward(self, v: Cartesian, spherical_triangle: SphericalTriangle, face_triangle: FaceTriangle) -> Face:
        """
        Forward projection: converts a spherical point to face coordinates
        
        Args:
            v: The spherical point to project
            spherical_triangle: The spherical triangle vertices
            face_triangle: The face triangle vertices
            
        Returns:
            The face coordinates
        """
        A, B, C = spherical_triangle

        # When v is close to A, the quadruple product is unstable.
        # As we just need the intersection of two great circles we can use difference
        # between A and v, as it lies in the same plane of the great circle containing A & v
        Z = vec3.create()
        vec3.subtract(Z, v, A)
        vec3.normalize(Z, Z)
        Z = cast(Cartesian, (Z[0], Z[1], Z[2]))

        p = vec3.create()
        vec3.quadrupleProduct(p, A, Z, B, C)
        vec3.normalize(p, p)
        p = cast(Cartesian, (p[0], p[1], p[2]))

        h = vec3.vectorDifference(A, v) / vec3.vectorDifference(A, p)
        scaled_area = h / self._constants['area_abc']

        b = cast(Barycentric, (
            1 - h,
            scaled_area * spherical_triangle_area(A, p, C),
            scaled_area * spherical_triangle_area(A, B, p)
        ))

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
        
        # Cached shape constants (from the canonical triangle)
        constants = self._constants
        area_abc = constants['area_abc']
        c12 = constants['c12']
        s12 = constants['s12']
        kQ = constants['kQ']
        V = constants['V']

        # Point-dependent calculations
        h = 1 - b[0]
        R = b[2] / h
        alpha = R * area_abc
        S = math.sin(alpha)
        half_c = math.sin(alpha / 2)
        CC = 2 * half_c * half_c  # Half angle formula

        # Per-triangle: A·B and C·A swap between even/odd face triangles (see
        # class docstring), so they cannot come from the cached constants.
        c01 = vec3.dot(A, B)
        c20 = vec3.dot(C, A)

        f = S * V + CC * (c01 * c12 - c20)
        g = CC * s12 * (1 + c01)
        q = kQ * math.atan2(g, f)
        
        # Use gl-matrix style slerp for P = slerp(B, C, q)
        P = vec3.create()
        vec3.slerp(P, B, C, q)
        P = cast(Cartesian, (P[0], P[1], P[2]))
        
        # K = A - P  
        K = vec3.vectorDifference(A, P)
        t = self._safe_acos(h * K) / self._safe_acos(K)
        
        # Final slerp: out = slerp(A, P, t)
        out = [0.0, 0.0, 0.0]
        vec3.slerp(out, A, P, t)
        return cast(Cartesian, (out[0], out[1], out[2]))

    def _safe_acos(self, x: float) -> float:
        """
        Computes acos(1 - 2 * x * x) without loss of precision for small x
        
        Args:
            x: Input value
            
        Returns:
            acos(1 - x)
        """
        if x < 1e-3:
            return (2 * x + x * x * x / 3)
        else:
            return math.acos(1 - 2 * x * x) 