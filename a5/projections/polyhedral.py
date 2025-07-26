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
import numpy as np
from typing import cast
from ..core.coordinate_systems import Cartesian, Face, Barycentric, FaceTriangle, SphericalTriangle
from ..core.coordinate_transforms import face_to_barycentric, barycentric_to_face
from ..geometry.spherical_triangle import SphericalTriangleShape
from ..utils.vector import vector_difference, quadruple_product, slerp, dot_product, cross_product, vector_magnitude

class PolyhedralProjection:
    """
    Polyhedral Equal Area projection using Slice & Dice algorithm
    """
    
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
        # Convert inputs to numpy arrays
        v = np.array(v, dtype=np.float64)
        face_triangle_arr = tuple(np.array(vertex, dtype=np.float64) for vertex in face_triangle)
        A, B, C = [np.array(vertex, dtype=np.float64) for vertex in spherical_triangle]
        triangle_shape = SphericalTriangleShape([A, B, C])

        # When v is close to A, the quadruple product is unstable.
        # As we just need the intersection of two great circles we can use difference
        # between A and v, as it lies in the same plane of the great circle containing A & v
        Z = v - A
        Z_norm = vector_magnitude(Z)
        
        # Handle case where v is exactly A (or very close)
        if Z_norm < 1e-14:
            # v is at vertex A, return the corresponding face coordinate
            # This should be the first vertex of the face triangle for barycentric coord [1,0,0]
            return face_triangle_arr[0]
        
        Z = Z / Z_norm
        Z = cast(Cartesian, Z)
        
        p = quadruple_product(A, Z, B, C)
        p_norm = vector_magnitude(p)
        p = (p[0] / p_norm, p[1] / p_norm, p[2] / p_norm)
        p = cast(Cartesian, p)

        h = vector_difference(A, v) / vector_difference(A, p)
        area_abc = triangle_shape.get_area()
        scaled_area = h / area_abc
        
        b = cast(Barycentric, (
            1 - h,
            scaled_area * SphericalTriangleShape([A, p, C]).get_area(),
            scaled_area * SphericalTriangleShape([A, B, p]).get_area()
        ))
        
        return barycentric_to_face(b, face_triangle_arr)

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
        # Convert inputs to numpy arrays
        face_point = np.array(face_point, dtype=np.float64)
        face_triangle_arr = tuple(np.array(vertex, dtype=np.float64) for vertex in face_triangle)
        A, B, C = [np.array(vertex, dtype=np.float64) for vertex in spherical_triangle]
        triangle_shape = SphericalTriangleShape([A, B, C])
        b = face_to_barycentric(face_point, face_triangle_arr)

        threshold = 1 - 1e-14
        if b[0] > threshold:
            return A
        if b[1] > threshold:
            return B
        if b[2] > threshold:
            return C
        
        c1 = cross_product(B, C)
        area_abc = triangle_shape.get_area()
        h = 1 - b[0]
        R = b[2] / h
        alpha = R * area_abc
        S = math.sin(alpha)
        half_c = math.sin(alpha / 2)
        CC = 2 * half_c * half_c  # Half angle formula

        c01 = dot_product(A, B)
        c12 = dot_product(B, C)
        c20 = dot_product(C, A)
        s12 = vector_magnitude(c1)

        V = dot_product(A, c1)  # Triple product of A, B, C. Constant??
        f = S * V + CC * (c01 * c12 - c20)
        g = CC * s12 * (1 + c01)
        q = (2 / math.acos(c12)) * math.atan2(g, f)
        P = slerp(B, C, q)
        K = vector_difference(A, P)
        t = self._safe_acos(h * K) / self._safe_acos(K)
        out = slerp(A, P, t)
        return cast(Cartesian, out)

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