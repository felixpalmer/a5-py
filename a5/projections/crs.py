"""
A5
SPDX-License-Identifier: Apache-2.0
Copyright (c) A5 contributors
"""

import math
from typing import List, cast
from ..core.coordinate_systems import Cartesian, Radians, Spherical
from ..core.coordinate_transforms import to_cartesian
from ..core.constants import distance_to_edge, distance_to_vertex
from ..core.origin import origins
from ..math import vec3


class CRS:
    """
    The Coordinate Reference System (CRS) of the dodecahedron is a set of 62 vertices:
    - 12 face centers
    - 20 vertices
    - 30 edge midpoints
    
    The vertices are used as a rigid frame of reference for the dodecahedron in the
    dodecahedron projection. By constructing them once, we can avoid recalculating
    and be sure of their correctness.
    """
    
    def __init__(self):
        self._vertices: List[Cartesian] = []
        self._invocations = 0
        
        self._add_face_centers()  # 12 centers
        self._add_vertices()      # 20 vertices
        self._add_midpoints()     # 30 midpoints
        
        if len(self._vertices) != 62:
            raise ValueError(f"Failed to construct CRS: vertices length is {len(self._vertices)}, not 62")
        
        # Make vertices read-only
        self._vertices = tuple(self._vertices)
    
    @property
    def vertices(self) -> List[Cartesian]:
        """Get the list of vertices (for testing access)."""
        return list(self._vertices)
    
    def get_vertex(self, point: Cartesian) -> Cartesian:
        """Find the CRS vertex that matches the given point."""
        self._invocations += 1
        if self._invocations == 10000:
            print('Too many CRS invocations, results should be cached')
        
        for vertex in self._vertices:
            # Calculate distance manually
            dx = point[0] - vertex[0]
            dy = point[1] - vertex[1] 
            dz = point[2] - vertex[2]
            distance = math.sqrt(dx * dx + dy * dy + dz * dz)
            if distance < 1e-5:
                return vertex
        
        raise ValueError("Failed to find vertex in CRS")
    
    def _add_face_centers(self) -> None:
        """Add face centers to vertices."""
        for origin in origins:
            cartesian_center = to_cartesian(origin.axis)
            self._add(cartesian_center)
    
    def _add_vertices(self) -> None:
        """Add dodecahedron vertices to the CRS."""
        phi_vertex = cast(Radians, math.atan(distance_to_vertex))
        
        for origin in origins:
            for i in range(5):
                theta_vertex = cast(Radians, (2 * i + 1) * math.pi / 5)
                spherical_vertex = cast(Spherical, (theta_vertex + origin.angle, phi_vertex))
                vertex = list(to_cartesian(spherical_vertex))
                vec3.transformQuat(vertex, vertex, origin.quat)
                self._add(vertex)
    
    def _add_midpoints(self) -> None:
        """Add edge midpoints to the CRS."""
        phi_midpoint = cast(Radians, math.atan(distance_to_edge))
        
        for origin in origins:
            for i in range(5):
                theta_midpoint = cast(Radians, (2 * i) * math.pi / 5)
                spherical_midpoint = cast(Spherical, (theta_midpoint + origin.angle, phi_midpoint))
                midpoint = list(to_cartesian(spherical_midpoint))
                vec3.transformQuat(midpoint, midpoint, origin.quat)
                self._add(midpoint)
    
    def _add(self, new_vertex: Cartesian) -> bool:
        """Add a new vertex if it doesn't already exist."""
        normalized = vec3.normalize(vec3.create(), new_vertex)
        
        # Check if vertex already exists
        for existing_vertex in self._vertices:
            distance = vec3.distance(normalized, existing_vertex)
            if distance < 1e-5:
                return False
        
        self._vertices.append(normalized)
        return True 