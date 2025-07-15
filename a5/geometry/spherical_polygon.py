# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

import numpy as np
from typing import List, Tuple, Union
import math
from ..core.coordinate_systems import Cartesian
from ..utils.vector import slerp, triple_product

# Type aliases for clarity
SphericalPolygon = List[Cartesian]

# Use Cartesian system for all calculations for greater accuracy
# Using [x, y, z] gives equal precision in all directions, unlike spherical coordinates
UP = np.array([0.0, 0.0, 1.0], dtype=np.float64)


class SphericalPolygonShape:
    def __init__(self, vertices: SphericalPolygon):
        # Convert all vertices to numpy arrays and ensure they're float64
        self.vertices = [np.array(v, dtype=np.float64) for v in vertices]
        # Make vertices immutable
        self.vertices = tuple(self.vertices)
        # self._is_winding_correct()  # Debug check only, don't correct

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
        
        return slerp(A, B, f)

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
        VA = self.vertices[j].copy() - V  # Directly subtract to match JS vec3.sub
        VB = self.vertices[k].copy() - V
        
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
        mid_a = (v2 + v3) * 0.5
        mid_b = (v3 + v1) * 0.5
        mid_c = (v1 + v2) * 0.5
        
        # Normalize midpoints
        mid_a = mid_a / np.linalg.norm(mid_a)
        mid_b = mid_b / np.linalg.norm(mid_b)
        mid_c = mid_c / np.linalg.norm(mid_c)
        
        # Calculate area using triple product
        S = triple_product(mid_a, mid_b, mid_c)
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
        center = np.zeros(3, dtype=np.float64)
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
        if len(self.vertices) < 3:
            return True
        area = self.get_area()
        return area > 0


class SphericalTriangleShape(SphericalPolygonShape):
    """
    A spherical triangle is a spherical polygon with exactly 3 vertices
    """
    
    def __init__(self, vertices: SphericalPolygon):
        if len(vertices) != 3:
            raise ValueError('SphericalTriangleShape requires exactly 3 vertices')
        super().__init__(vertices) 