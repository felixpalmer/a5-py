# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

import numpy as np
from typing import List, Tuple, Union
import math
from .coordinate_systems import Cartesian

# Type aliases for clarity
SphericalPolygon = List[Cartesian]

# Use Cartesian system for all calculations for greater accuracy
# Using [x, y, z] gives equal precision in all directions, unlike spherical coordinates
UP = np.array([0.0, 0.0, 1.0], dtype=np.float64)


class SphericalPolygonShape:
    def __init__(self, vertices: SphericalPolygon):
        self.vertices = vertices
        if not self._is_winding_correct():
            self.vertices.reverse()

    def get_boundary(self, n_segments: int = 1, closed_ring: bool = True) -> SphericalPolygon:
        """
        Returns a close boundary of the polygon, with n_segments points per edge
        
        Args:
            n_segments: Number of points per edge
            closed_ring: Whether to close the ring by adding the first point at the end
            
        Returns:
            List of Cartesian coordinates forming the boundary
        """
        points: SphericalPolygon = []
        N = len(self.vertices)
        
        for s in range(N * n_segments):
            t = s / n_segments
            points.append(self.slerp(t))
            
        if closed_ring:
            points.append(points[0])
        
        return points

    def slerp(self, t: float) -> Cartesian:
        """
        Interpolates along boundary of polygon. Pass t = 1.5 to get the midpoint between 2nd and 3rd vertices
        
        Args:
            t: Parameter along the boundary (can be fractional)
            
        Returns:
            Cartesian coordinate
        """
        N = len(self.vertices)
        f = t % 1.0
        i = int(math.floor(t % N))
        j = (i + 1) % N

        # Points A & B
        A = self.vertices[i]
        B = self.vertices[j]

        # Quaternion-based spherical linear interpolation
        q_oa = self._rotation_to(UP, A)
        q_ab = self._rotation_to(A, B)
        q_partial = self._slerp_quat(np.array([0.0, 0.0, 0.0, 1.0]), q_ab, f)
        q_combined = self._multiply_quat(q_partial, q_oa)

        # Transform unit vector [0, 0, 1] by the combined quaternion
        out = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        return self._transform_quat(out, q_combined)

    def get_transformed_vertices(self, t: float) -> Tuple[Cartesian, Cartesian, Cartesian]:
        """
        Returns the vertex given by index t, along with the vectors:
        - VA: Vector from vertex to point A
        - VB: Vector from vertex to point B
        
        Args:
            t: Parameter along the boundary
            
        Returns:
            Tuple of (V, VA, VB) where V is the vertex and VA, VB are vectors
        """
        N = len(self.vertices)
        i = int(math.floor(t % N))
        j = (i + 1) % N
        k = (i + N - 1) % N

        # Points A & B (vertex before and after)
        V = self.vertices[i].copy()
        VA = self.vertices[j].copy()
        VB = self.vertices[k].copy()
        
        VA = VA - V
        VB = VB - V
        
        return V, VA, VB

    def contains_point(self, point: Cartesian) -> float:
        """
        Determines if a point is inside the spherical polygon.
        Adaptation of algorithm from:
        'Locating a point on a spherical surface relative to a spherical polygon'
        Using only the condition of 'necessary strike'
        
        Args:
            point: Cartesian coordinate to test
            
        Returns:
            Positive value if inside, 0 if on edge, negative if outside
        """
        N = len(self.vertices)
        theta_delta_min = float('inf')

        for i in range(N):
            # Transform point and neighboring vertices into coordinate system centered on vertex
            V, VA, VB = self.get_transformed_vertices(i)
            VP = point - V

            # Normalize to obtain unit direction vectors
            VP = VP / np.linalg.norm(VP)
            VA = VA / np.linalg.norm(VA)
            VB = VB / np.linalg.norm(VB)

            # Cross products will point away from the center of the sphere when
            # point P is within arc formed by VA and VB
            cross_ap = np.cross(VA, VP)
            cross_pb = np.cross(VP, VB)

            # Dot product will be positive when point P is within arc formed by VA and VB
            # The magnitude of the dot product is the sine of the angle between the two vectors
            # which is the same as the angle for small angles.
            sin_ap = np.dot(V, cross_ap)
            sin_pb = np.dot(V, cross_pb)

            # By returning the minimum value we find the arc where the point is closest to being outside
            theta_delta_min = min(theta_delta_min, sin_ap, sin_pb)

        # If point is inside all arcs, will return a positive value
        # If point is on edge of arc, will return 0
        # If point is outside all arcs, will return -1, the further away from 0, the further away from the arc
        return theta_delta_min

    def get_triangle_area(self, v1: Cartesian, v2: Cartesian, v3: Cartesian) -> float:
        """
        Calculate the area of a spherical triangle given three vertices
        
        Args:
            v1: First vertex
            v2: Second vertex  
            v3: Third vertex
            
        Returns:
            Area of the spherical triangle in radians
        """
        # Calculate midpoints
        mid_a = (v2 + v3) / 2
        mid_b = (v3 + v1) / 2
        mid_c = (v1 + v2) / 2
        
        # Normalize midpoints
        mid_a = mid_a / np.linalg.norm(mid_a)
        mid_b = mid_b / np.linalg.norm(mid_b)
        mid_c = mid_c / np.linalg.norm(mid_c)
        
        # Calculate area using triple product
        S = np.dot(mid_a, np.cross(mid_b, mid_c))
        clamped = max(-1.0, min(1.0, S))
        
        # sin(x) = x for x < 1e-8
        if abs(clamped) < 1e-8:
            return 2 * clamped
        else:
            return math.asin(clamped) * 2

    def get_area(self) -> float:
        """
        Calculate the area of the spherical polygon by decomposing it into a fan of triangles
        
        Returns:
            The area of the spherical polygon in radians
        """
        if not hasattr(self, '_area') or self._area is None:
            self._area = self._get_area()
        return self._area

    def _get_area(self) -> float:
        """
        Internal method to calculate the area of the spherical polygon
        
        Returns:
            The area of the spherical polygon in radians
        """
        if len(self.vertices) < 3:
            return 0.0

        if len(self.vertices) == 3:
            return self.get_triangle_area(self.vertices[0], self.vertices[1], self.vertices[2])

        # Calculate center of polygon
        center = np.zeros(3)
        for vertex in self.vertices:
            center += vertex
        center = center / np.linalg.norm(center)

        # Sum fan of triangles around center
        area = 0.0
        for i in range(len(self.vertices)):
            v1 = self.vertices[i]
            v2 = self.vertices[(i + 1) % len(self.vertices)]
            tri_area = self.get_triangle_area(center, v1, v2)
            if not math.isnan(tri_area):
                area += tri_area

        return area

    def _is_winding_correct(self) -> bool:
        """Check if the polygon vertices are in the correct winding order"""
        V, VA, VB = self.get_transformed_vertices(0)
        cross = np.cross(VA, VB)
        return np.dot(V, cross) >= 0

    def _rotation_to(self, from_vec: Cartesian, to_vec: Cartesian) -> np.ndarray:
        """Create a quaternion that rotates from one vector to another"""
        # Normalize vectors
        from_vec = from_vec / np.linalg.norm(from_vec)
        to_vec = to_vec / np.linalg.norm(to_vec)
        
        # Calculate dot product
        dot = np.dot(from_vec, to_vec)
        
        # Handle nearly opposite vectors (dot < -0.999999)
        if dot < -0.999999:
            # Try cross product with x unit vector
            tmp_vec = np.cross(np.array([1.0, 0.0, 0.0]), from_vec)
            if np.linalg.norm(tmp_vec) < 0.000001:
                # If that fails, try y unit vector
                tmp_vec = np.cross(np.array([0.0, 1.0, 0.0]), from_vec)
            tmp_vec = tmp_vec / np.linalg.norm(tmp_vec)
            
            # Create quaternion for 180 degree rotation around tmp_vec
            angle = math.pi
            sin_half_angle = math.sin(angle / 2.0)
            cos_half_angle = math.cos(angle / 2.0)
            return np.array([
                tmp_vec[0] * sin_half_angle,
                tmp_vec[1] * sin_half_angle,
                tmp_vec[2] * sin_half_angle,
                cos_half_angle
            ])
        
        # Handle nearly parallel vectors (dot > 0.999999)
        elif dot > 0.999999:
            return np.array([0.0, 0.0, 0.0, 1.0])
        
        # General case
        else:
            cross_vec = np.cross(from_vec, to_vec)
            quat = np.array([cross_vec[0], cross_vec[1], cross_vec[2], 1.0 + dot])
            return quat / np.linalg.norm(quat)

    def _slerp_quat(self, q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
        """Spherical linear interpolation between two quaternions"""
        # Extract components
        ax, ay, az, aw = q1
        bx, by, bz, bw = q2.copy()
        
        # Calculate cosine
        cosom = ax * bx + ay * by + az * bz + aw * bw
        
        # Adjust signs if necessary
        if cosom < 0.0:
            cosom = -cosom
            bx = -bx
            by = -by
            bz = -bz
            bw = -bw
        
        # Calculate coefficients
        EPSILON = 1e-6
        if 1.0 - cosom > EPSILON:
            # Standard case (slerp)
            omega = math.acos(cosom)
            sinom = math.sin(omega)
            scale0 = math.sin((1.0 - t) * omega) / sinom
            scale1 = math.sin(t * omega) / sinom
        else:
            # "from" and "to" quaternions are very close
            # so we can do a linear interpolation
            scale0 = 1.0 - t
            scale1 = t
        
        # Calculate final values
        out = np.array([
            scale0 * ax + scale1 * bx,
            scale0 * ay + scale1 * by,
            scale0 * az + scale1 * bz,
            scale0 * aw + scale1 * bw
        ])
        
        return out

    def _multiply_quat(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Multiply two quaternions - quaternions stored as [x, y, z, w]"""
        # Extract components
        ax = q1[0]
        ay = q1[1]
        az = q1[2]
        aw = q1[3]
        bx = q2[0]
        by = q2[1]
        bz = q2[2]
        bw = q2[3]
        
        return np.array([
            ax * bw + aw * bx + ay * bz - az * by,
            ay * bw + aw * by + az * bx - ax * bz,
            az * bw + aw * bz + ax * by - ay * bx,
            aw * bw - ax * bx - ay * by - az * bz
        ])

    def _transform_quat(self, vec: Cartesian, quat: np.ndarray) -> Cartesian:
        """Transform a vector by a quaternion using optimized algorithm"""
        qx, qy, qz, qw = quat
        w2 = qw * 2
        x, y, z = vec
        
        # Cross product: qvec × a
        uvx = qy * z - qz * y
        uvy = qz * x - qx * z
        uvz = qx * y - qy * x
        
        # Cross product: qvec × uv, scaled by 2
        uuvx = (qy * uvz - qz * uvy) * 2
        uuvy = (qz * uvx - qx * uvz) * 2
        uuvz = (qx * uvy - qy * uvx) * 2
        
        # Final result: a + (uv * w2) + uuv
        return np.array([
            x + (uvx * w2) + uuvx,
            y + (uvy * w2) + uuvy,
            z + (uvz * w2) + uuvz
        ]) 