# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

from typing import List, Tuple, Union
import math
from ..core.coordinate_systems import Cartesian
from ..math import vec3, quat

# Type aliases for clarity
SphericalPolygon = List[Cartesian]

# Use Cartesian system for all calculations for greater accuracy
# Using [x, y, z] gives equal precision in all directions, unlike spherical coordinates
UP = (0.0, 0.0, 1.0)


class SphericalPolygonShape:
    def __init__(self, vertices: SphericalPolygon):
        # Store vertices as tuples for immutability and consistency
        self.vertices = tuple(tuple(v) if not isinstance(v, tuple) else v for v in vertices)
        # self._is_winding_correct()  # Debug check only, don't correct
        
        # Pre-allocated temporary vectors for performance
        self._temp_a = vec3.create()
        self._temp_b = vec3.create()
        self._temp_c = vec3.create()
        self._temp_cross = vec3.create()
        self._temp_out = vec3.create()
        self._temp_scaled_a = vec3.create()
        self._temp_scaled_b = vec3.create()

    def get_boundary(self, n_segments: int = 1, closed_ring: bool = True) -> SphericalPolygon:
        """
        Returns a closed boundary of the polygon, with n_segments points per edge
        
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
        
        # Spherical linear interpolation between A and B
        return quat.vectorSlerp(self._temp_out, self._temp_a, self._temp_b, self._temp_scaled_a, self._temp_scaled_b, A, B, f)

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
        V = self.vertices[i]
        temp_j = vec3.create()
        temp_k = vec3.create()
        temp_v = vec3.create()
        
        vec3.copy(temp_j, self.vertices[j])
        vec3.copy(temp_k, self.vertices[k])
        vec3.copy(temp_v, V)
        
        vec3.subtract(temp_j, temp_j, temp_v)  # VA
        vec3.subtract(temp_k, temp_k, temp_v)  # VB
        
        VA = (temp_j[0], temp_j[1], temp_j[2])
        VB = (temp_k[0], temp_k[1], temp_k[2])
        
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
            temp_point = vec3.create()
            temp_v_local = vec3.create()
            vec3.copy(temp_point, point)
            vec3.copy(temp_v_local, V)
            vec3.subtract(temp_point, temp_point, temp_v_local)
            VP = (temp_point[0], temp_point[1], temp_point[2])

            # Normalize to obtain unit direction vectors
            vec3.copy(self._temp_a, VP)
            norm_VP = vec3.length(self._temp_a)
            vec3.copy(self._temp_a, VA)
            norm_VA = vec3.length(self._temp_a)
            vec3.copy(self._temp_a, VB)
            norm_VB = vec3.length(self._temp_a)
            
            # Handle case where point is identical to vertex (zero-length vector)
            if norm_VP < 1e-14:
                return 1.0  # Point is exactly on the vertex, so it's inside
            if norm_VA < 1e-14 or norm_VB < 1e-14:
                continue  # Skip degenerate edge
                
            VP = (VP[0] / norm_VP, VP[1] / norm_VP, VP[2] / norm_VP)
            VA = (VA[0] / norm_VA, VA[1] / norm_VA, VA[2] / norm_VA)
            VB = (VB[0] / norm_VB, VB[1] / norm_VB, VB[2] / norm_VB)

            # Cross products will point away from the center of the sphere when
            # point P is within arc formed by VA and VB
            vec3.copy(self._temp_a, VA)
            vec3.copy(self._temp_b, VP)
            vec3.cross(self._temp_cross, self._temp_a, self._temp_b)
            cross_ap = (self._temp_cross[0], self._temp_cross[1], self._temp_cross[2])
            
            vec3.copy(self._temp_a, VP)
            vec3.copy(self._temp_b, VB)
            vec3.cross(self._temp_cross, self._temp_a, self._temp_b)
            cross_pb = (self._temp_cross[0], self._temp_cross[1], self._temp_cross[2])

            # Dot product will be positive when point P is within arc formed by VA and VB
            # The magnitude of the dot product is the sine of the angle between the two vectors
            # which is the same as the angle for small angles.
            vec3.copy(self._temp_a, V)
            vec3.copy(self._temp_b, cross_ap)
            sin_ap = vec3.dot(self._temp_a, self._temp_b)
            
            vec3.copy(self._temp_a, V)
            vec3.copy(self._temp_b, cross_pb)
            sin_pb = vec3.dot(self._temp_a, self._temp_b)

            # By returning the minimum value we find the arc where the point is closest to being outside
            theta_delta_min = min(theta_delta_min, sin_ap, sin_pb)

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
        # Calculate midpoints using gl-matrix style
        temp_a = vec3.create()
        temp_b = vec3.create() 
        temp_c = vec3.create()
        
        # mid_a = (v2 + v3) * 0.5
        vec3.add(temp_a, v2, v3)
        vec3.scale(temp_a, temp_a, 0.5)
        
        # mid_b = (v3 + v1) * 0.5
        vec3.add(temp_b, v3, v1)
        vec3.scale(temp_b, temp_b, 0.5)
        
        # mid_c = (v1 + v2) * 0.5
        vec3.add(temp_c, v1, v2)
        vec3.scale(temp_c, temp_c, 0.5)
        
        # Normalize midpoints
        vec3.normalize(temp_a, temp_a)
        vec3.normalize(temp_b, temp_b)
        vec3.normalize(temp_c, temp_c)
        
        mid_a = (temp_a[0], temp_a[1], temp_a[2])
        mid_b = (temp_b[0], temp_b[1], temp_b[2])
        mid_c = (temp_c[0], temp_c[1], temp_c[2])
        
        # Calculate area using triple product: mid_a · (mid_b × mid_c)
        vec3.copy(self._temp_a, mid_a)
        vec3.copy(self._temp_b, mid_b)
        vec3.copy(self._temp_c, mid_c)
        S = vec3.tripleProduct(self._temp_cross, self._temp_a, self._temp_b, self._temp_c)
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

        # Calculate center of polygon using gl-matrix style
        center = vec3.create()
        for vertex in self.vertices:
            vec3.add(center, center, vertex)
        
        # Normalize center
        vec3.normalize(center, center)
        center = (center[0], center[1], center[2])

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
        if len(self.vertices) < 3:
            return True
        area = self.get_area()
        return area > 0 