# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

import math
from typing import List, Tuple
from ..core.coordinate_systems import Face
from ..math import vec2

Pentagon = List[Face]

class PentagonShape:
    def __init__(self, vertices: Pentagon):
        self.vertices = list(vertices)  # Make a copy to avoid mutating original
        if not self._is_winding_correct():
            self.vertices.reverse()

    def get_area(self) -> float:
        """Calculate the signed area of the pentagon using the shoelace formula."""
        signed_area = 0.0
        n = len(self.vertices)
        for i in range(n):
            j = (i + 1) % n
            signed_area += (self.vertices[j][0] - self.vertices[i][0]) * (self.vertices[j][1] + self.vertices[i][1])
        return signed_area

    def _is_winding_correct(self) -> bool:
        """Check if the pentagon has counter-clockwise winding (positive area)."""
        return self.get_area() >= 0

    def get_vertices(self) -> Pentagon:
        """Get the vertices of the pentagon."""
        return self.vertices

    def scale(self, scale: float) -> "PentagonShape":
        """Scale the pentagon by the given factor."""
        for i, vertex in enumerate(self.vertices):
            self.vertices[i] = (vertex[0] * scale, vertex[1] * scale)
        return self

    def rotate180(self) -> "PentagonShape":
        """Rotate the pentagon 180 degrees (equivalent to negating x & y)."""
        for i, vertex in enumerate(self.vertices):
            self.vertices[i] = (-vertex[0], -vertex[1])
        return self

    def reflect_y(self) -> "PentagonShape":
        """
        Reflect the pentagon over the x-axis (equivalent to negating y)
        and reverse the winding order to maintain consistent orientation.
        """
        # First reflect all vertices
        for i, vertex in enumerate(self.vertices):
            self.vertices[i] = (vertex[0], -vertex[1])
        
        # Then reverse the winding order to maintain consistent orientation
        self.vertices.reverse()
        
        return self

    def translate(self, translation: Tuple[float, float]) -> "PentagonShape":
        """Translate the pentagon by the given vector."""
        for i, vertex in enumerate(self.vertices):
            self.vertices[i] = (vertex[0] + translation[0], vertex[1] + translation[1])
        return self

    def transform(self, transform: Tuple[Tuple[float, float], Tuple[float, float]]) -> "PentagonShape":
        """Apply a 2x2 transformation matrix to the pentagon."""
        for i, vertex in enumerate(self.vertices):
            # Manual matrix multiplication: transform @ vertex
            new_x = transform[0][0] * vertex[0] + transform[0][1] * vertex[1]
            new_y = transform[1][0] * vertex[0] + transform[1][1] * vertex[1]
            self.vertices[i] = (new_x, new_y)
        return self

    def transform2d(self, transform: Tuple[Tuple[float, float, float], Tuple[float, float, float]]) -> "PentagonShape":
        """Apply a 2x3 transformation matrix to the pentagon."""
        for i, vertex in enumerate(self.vertices):
            # Manual matrix multiplication for 2x3 matrix: transform[:, :2] @ vertex + transform[:, 2]
            new_x = transform[0][0] * vertex[0] + transform[0][1] * vertex[1] + transform[0][2]
            new_y = transform[1][0] * vertex[0] + transform[1][1] * vertex[1] + transform[1][2]
            self.vertices[i] = (new_x, new_y)
        return self

    def clone(self) -> "PentagonShape":
        """Create a deep copy of the pentagon."""
        return PentagonShape([vertex for vertex in self.vertices])

    def get_center(self) -> Face:
        """Get the center point of the pentagon."""
        n = len(self.vertices)
        sum_x = sum(v[0] for v in self.vertices) / n
        sum_y = sum(v[1] for v in self.vertices) / n
        return (sum_x, sum_y)

    def contains_point(self, point: Tuple[float, float]) -> float:
        """
        Test if a point is inside the pentagon by checking if it's on the correct side of all edges.
        Assumes consistent winding order (counter-clockwise).
        
        Args:
            point: The point to test
            
        Returns:
            1 if point is inside, otherwise a negative value proportional to the distance from the point to the edge
        """
        # TODO: later we can likely remove this, but for now it's useful for debugging
        if not self._is_winding_correct():
            raise ValueError("Pentagon is not counter-clockwise")

        n = len(self.vertices)
        d_max = 1
        for i in range(n):
            v1 = self.vertices[i]
            v2 = self.vertices[(i + 1) % n]
            
            # Calculate the cross product to determine which side of the line the point is on
            # (v1 - v2) × (point - v1)
            dx = v1[0] - v2[0]
            dy = v1[1] - v2[1]
            px = point[0] - v1[0]
            py = point[1] - v1[1]
            
            # Cross product: dx * py - dy * px
            # If positive, point is on the wrong side
            # If negative, point is on the correct side
            cross_product = dx * py - dy * px
            if cross_product < 0:
                # Only normalize by distance of point to edge as we can assume the edges of the
                # pentagon are all the same length
                p_length = math.sqrt(px * px + py * py)
                d_max = min(d_max, cross_product / p_length)
        
        return d_max

    def split_edges(self, segments: int) -> "PentagonShape":
        """
        Split each edge of the pentagon into the specified number of segments.
        
        Args:
            segments: Number of segments to split each edge into
            
        Returns:
            A new PentagonShape with more vertices, or the original PentagonShape if segments <= 1
        """
        if segments <= 1:
            return self

        new_vertices = []
        n = len(self.vertices)
        
        for i in range(n):
            v1 = self.vertices[i]
            v2 = self.vertices[(i + 1) % n]
            
            # Add the current vertex
            new_vertices.append(v1)
            
            # Add interpolated points along the edge (excluding the endpoints)
            for j in range(1, segments):
                t = j / segments
                interpolated = vec2.create()
                vec2.lerp(interpolated, v1, v2, t)
                new_vertices.append((interpolated[0], interpolated[1]))
        
        return PentagonShape(new_vertices)