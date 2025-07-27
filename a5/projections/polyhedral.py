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
        triangle_shape = SphericalTriangleShape(spherical_triangle)

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
        Area_ABC = triangle_shape.get_area()
        scaled_area = h / Area_ABC
        
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
            triangle_shape = SphericalTriangleShape(spherical_triangle)
            c1 = vec3.create()
            vec3.cross(c1, B, C)
            
            constants = {
                'area_abc': triangle_shape.get_area(),
                'c1': (c1[0], c1[1], c1[2]),  # Store as tuple
                'c01': vec3.dot(A, B),
                'c12': vec3.dot(B, C),
                'c20': vec3.dot(C, A),
                's12': vec3.length(c1),
                'V': vec3.dot(A, c1)  # Triple product of A, B, C
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