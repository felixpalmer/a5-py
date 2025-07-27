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
from typing import cast, Dict, Tuple
from ..core.coordinate_systems import Cartesian, Face, Barycentric, FaceTriangle, SphericalTriangle
from ..core.coordinate_transforms import face_to_barycentric, barycentric_to_face
from ..geometry.spherical_triangle import SphericalTriangleShape
from ..math import vec3, quat

class PolyhedralProjection:
    """
    Polyhedral Equal Area projection using Slice & Dice algorithm
    """
    
    def __init__(self):
        # Cache for triangle-dependent calculations in inverse projection
        self._inverse_triangle_cache: Dict[Tuple, Dict] = {}
        
        # Pre-allocated temporary vectors for performance
        self._temp_v = vec3.create()
        self._temp_a = vec3.create()
        self._temp_b = vec3.create()
        self._temp_c = vec3.create()
        self._temp_c1 = vec3.create()
        self._temp_p = vec3.create()
        self._temp_k = vec3.create()
        self._temp_out = vec3.create()
        self._temp_cross = vec3.create()
        self._temp_midpoint = vec3.create()
        self._temp_scaled_a = vec3.create()
        self._temp_scaled_b = vec3.create()
    
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
        # Use inputs directly
        A, B, C = spherical_triangle
        triangle_shape = SphericalTriangleShape(spherical_triangle)

        # When v is close to A, the quadruple product is unstable.
        # As we just need the intersection of two great circles we can use difference
        # between A and v, as it lies in the same plane of the great circle containing A & v
        vec3.copy(self._temp_v, v)
        vec3.copy(self._temp_a, A)
        vec3.subtract(self._temp_v, self._temp_v, self._temp_a)  # Z = v - A
        Z_norm = vec3.length(self._temp_v)
        
        # Handle case where v is exactly A (or very close)
        if Z_norm < 1e-14:
            # v is at vertex A, return the corresponding face coordinate
            # This should be the first vertex of the face triangle for barycentric coord [1,0,0]
            return face_triangle[0]
        
        vec3.normalize(self._temp_v, self._temp_v)
        Z = cast(Cartesian, (self._temp_v[0], self._temp_v[1], self._temp_v[2]))
        
        vec3.quadrupleProduct(self._temp_p, self._temp_a, self._temp_b, self._temp_c, self._temp_cross, 
                             self._temp_scaled_a, self._temp_scaled_b, A, Z, B, C)
        vec3.normalize(self._temp_p, self._temp_p)
        p = cast(Cartesian, (self._temp_p[0], self._temp_p[1], self._temp_p[2]))

        D_av = vec3.vectorDifference(self._temp_a, self._temp_b, self._temp_midpoint, self._temp_cross, self._temp_out, A, v)
        D_ap = vec3.vectorDifference(self._temp_a, self._temp_b, self._temp_midpoint, self._temp_cross, self._temp_out, A, p)
        h = D_av / D_ap
        area_abc = triangle_shape.get_area()
        scaled_area = h / area_abc
        
        b = cast(Barycentric, (
            1 - h,
            scaled_area * SphericalTriangleShape([A, p, C]).get_area(),
            scaled_area * SphericalTriangleShape([A, B, p]).get_area()
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
        
        # Get cached triangle-dependent constants
        constants = self._get_triangle_constants(spherical_triangle)
        area_abc = constants['area_abc']
        c1 = constants['c1']
        c01 = constants['c01']
        c12 = constants['c12']
        c20 = constants['c20']
        s12 = constants['s12']
        V = constants['V']
        
        # Point-dependent calculations
        h = 1 - b[0]
        R = b[2] / h
        alpha = R * area_abc
        S = math.sin(alpha)
        half_c = math.sin(alpha / 2)
        CC = 2 * half_c * half_c  # Half angle formula

        f = S * V + CC * (c01 * c12 - c20)
        g = CC * s12 * (1 + c01)
        q = (2 / math.acos(c12)) * math.atan2(g, f)
        
        # Copy vertices to temp vectors
        vec3.copy(self._temp_a, A)
        vec3.copy(self._temp_b, B)
        vec3.copy(self._temp_c, C)
        
        # Use gl-matrix style slerp for P = slerp(B, C, q)
        # Calculate gamma between B and C
        gamma = vec3.angle(self._temp_b, self._temp_c)
        if gamma < 1e-12:
            vec3.lerp(self._temp_p, self._temp_b, self._temp_c, q)
        else:
            weight_b = math.sin((1 - q) * gamma) / math.sin(gamma)
            weight_c = math.sin(q * gamma) / math.sin(gamma)
            vec3.scale(self._temp_b, self._temp_b, weight_b)
            vec3.scale(self._temp_c, self._temp_c, weight_c)
            vec3.add(self._temp_p, self._temp_b, self._temp_c)
        
        # K = A - P
        vec3.subtract(self._temp_k, self._temp_a, self._temp_p)
        k_mag = vec3.length(self._temp_k)
        
        t = self._safe_acos(h * k_mag) / self._safe_acos(k_mag)
        
        # Final slerp: out = slerp(A, P, t)
        gamma2 = vec3.angle(self._temp_a, self._temp_p)
        if gamma2 < 1e-12:
            vec3.lerp(self._temp_out, self._temp_a, self._temp_p, t)
        else:
            weight_a = math.sin((1 - t) * gamma2) / math.sin(gamma2)
            weight_p = math.sin(t * gamma2) / math.sin(gamma2)
            vec3.scale(self._temp_a, self._temp_a, weight_a)
            vec3.scale(self._temp_p, self._temp_p, weight_p)
            vec3.add(self._temp_out, self._temp_a, self._temp_p)
        
        return cast(Cartesian, (self._temp_out[0], self._temp_out[1], self._temp_out[2]))

    def _get_triangle_constants(self, spherical_triangle: SphericalTriangle):
        """
        Get cached triangle-dependent constants for inverse projection.
        These values only depend on the spherical triangle, not the input point.
        """
        # Create a cache key from the triangle vertices
        # Convert to tuples since lists aren't hashable
        A, B, C = spherical_triangle
        cache_key = (tuple(A), tuple(B), tuple(C))
        
        if cache_key not in self._inverse_triangle_cache:
            # Copy vertices to temporary vectors
            vec3.copy(self._temp_a, A)
            vec3.copy(self._temp_b, B)
            vec3.copy(self._temp_c, C)
            
            triangle_shape = SphericalTriangleShape(spherical_triangle)
            vec3.cross(self._temp_c1, self._temp_b, self._temp_c)
            
            constants = {
                'area_abc': triangle_shape.get_area(),
                'c1': (self._temp_c1[0], self._temp_c1[1], self._temp_c1[2]),  # Store as tuple
                'c01': vec3.dot(self._temp_a, self._temp_b),
                'c12': vec3.dot(self._temp_b, self._temp_c),
                'c20': vec3.dot(self._temp_c, self._temp_a),
                's12': vec3.length(self._temp_c1),
                'V': vec3.dot(self._temp_a, self._temp_c1)  # Triple product of A, B, C
            }
            self._inverse_triangle_cache[cache_key] = constants
        
        return self._inverse_triangle_cache[cache_key]

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