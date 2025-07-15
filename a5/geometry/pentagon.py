# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

import numpy as np
from typing import List, Tuple, Optional, Literal, NamedTuple
from ..core.coordinate_systems import Radians, Face, Spherical
from ..core.hilbert import Orientation

OriginId = Literal[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

class Origin(NamedTuple):
    id: OriginId
    axis: Spherical
    quat: np.ndarray
    angle: Radians
    orientation: List[Orientation]
    first_quintant: int

Pentagon = List[Face]

class PentagonShape:
    def __init__(self, vertices: Pentagon):
        self.vertices = vertices
        self.id = {"i": 0, "j": 0, "k": 0, "resolution": 1, "segment": None, "origin": None}
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
        for vertex in self.vertices:
            vertex *= scale
        return self

    def rotate180(self) -> "PentagonShape":
        """Rotate the pentagon 180 degrees (equivalent to negating x & y)."""
        for vertex in self.vertices:
            vertex *= -1
        return self

    def reflect_y(self) -> "PentagonShape":
        """
        Reflect the pentagon over the x-axis (equivalent to negating y)
        and reverse the winding order to maintain consistent orientation.
        """
        # First reflect all vertices
        for vertex in self.vertices:
            vertex[1] = -vertex[1]
        
        # Then reverse the winding order to maintain consistent orientation
        self.vertices.reverse()
        
        return self

    def translate(self, translation: np.ndarray) -> "PentagonShape":
        """Translate the pentagon by the given vector."""
        for vertex in self.vertices:
            vertex += translation
        return self

    def transform(self, transform: np.ndarray) -> "PentagonShape":
        """Apply a 2x2 transformation matrix to the pentagon."""
        for i, vertex in enumerate(self.vertices):
            self.vertices[i] = np.dot(transform, vertex)
        return self

    def transform2d(self, transform: np.ndarray) -> "PentagonShape":
        """Apply a 2x3 transformation matrix to the pentagon."""
        for i, vertex in enumerate(self.vertices):
            self.vertices[i] = np.dot(transform[:, :2], vertex) + transform[:, 2]
        return self

    def clone(self) -> "PentagonShape":
        """Create a deep copy of the pentagon."""
        return PentagonShape([np.copy(v) for v in self.vertices])

    def get_center(self) -> Face:
        """Get the center point of the pentagon."""
        return np.sum(self.vertices, axis=0) / 5.0

    def contains_point(self, point: np.ndarray) -> float:
        """
        Test if a point is inside the pentagon by checking if it's on the correct side of all edges.
        Assumes consistent winding order (counter-clockwise).
        
        Args:
            point: The point to test
            
        Returns:
            -1 if point is inside, otherwise a value proportional to the distance from the point to the edge
        """
        # TODO: later we can likely remove this, but for now it's useful for debugging
        if not self._is_winding_correct():
            raise ValueError("Pentagon is not counter-clockwise")

        n = len(self.vertices)
        for i in range(n):
            v1 = self.vertices[i]
            v2 = self.vertices[(i + 1) % n]
            
            # Calculate the cross product to determine which side of the line the point is on
            # (v2 - v1) × (point - v1)
            dx = v2[0] - v1[0]
            dy = v2[1] - v1[1]
            px = point[0] - v1[0]
            py = point[1] - v1[1]
            
            # Cross product: dx * py - dy * px
            # If positive, point is on the wrong side
            # If negative, point is on the correct side
            cross_product = dx * py - dy * px
            if cross_product > 0:
                # Only normalize by distance of point to edge as we can assume the edges of the
                # pentagon are all the same length
                p_length = np.sqrt(px * px + py * py)
                return cross_product / p_length
        
        return -1

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
            new_vertices.append(np.copy(v1))
            
            # Add interpolated points along the edge (excluding the endpoints)
            for j in range(1, segments):
                t = j / segments
                interpolated = v1 + t * (v2 - v1)
                new_vertices.append(interpolated)
        
        return PentagonShape(new_vertices) 