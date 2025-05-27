# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

import numpy as np
from typing import List, Tuple, Optional, TypedDict,NamedTuple
from .triangle import Triangle
from .coordinate_systems import Radians,LonLat,Face, Spherical
from .hilbert import Orientation
from dataclasses import dataclass

vec2 = np.array
vec3 = np.array
mat2 = np.ndarray
mat2d = np.ndarray

class Origin(NamedTuple):
    id: int
    axis: Spherical
    quat: np.ndarray
    angle: Radians
    orientation: List[Orientation]
    first_quintant: int

Pentagon = List[Face]
Contour = List[LonLat]

class PentagonShape:
    def __init__(self, vertices: Pentagon):
        self.vertices = vertices
        self.id = {"i": 0, "j": 0, "k": 0, "resolution": 1, "segment": None, "origin": None}
        self.triangles: Optional[List[Triangle]] = None

    def get_vertices(self) -> Pentagon:
        return self.vertices

    def scale(self, scale: float) -> "PentagonShape":
        for vertex in self.vertices:
            np.multiply(vertex, scale, out=vertex)
        return self

    """
    Rotates the pentagon 180 degrees (equivalent to negating x & y)
    @returns The rotated pentagon
    """
    def rotate180(self) -> "PentagonShape":
        for vertex in self.vertices:
            np.negative(vertex, out=vertex)
        return self

    """
    Reflects the pentagon over the x-axis (equivalent to negating y)
    @returns The reflected pentagon
    """
    def reflectY(self) -> "PentagonShape":
        for vertex in self.vertices:
            vertex[1] = -vertex[1]
        return self

    def translate(self, translation: vec2) -> "PentagonShape":
        for vertex in self.vertices:
            np.add(vertex, translation, out=vertex)
        return self

    def transform(self, transform: np.ndarray) -> "PentagonShape":
        for i, vertex in enumerate(self.vertices):
            self.vertices[i] = np.dot(transform, vertex)
        return self

    def transform2d(self, transform: np.ndarray) -> "PentagonShape":
        for i, vertex in enumerate(self.vertices):
            self.vertices[i] = np.dot(transform[:, :2], vertex) + transform[:, 2]
        return self

    def clone(self) -> "PentagonShape":
        return PentagonShape([np.copy(v) for v in self.vertices])

    def get_center(self) -> Face:
        return np.sum(self.vertices, axis=0) / 5.0

    """
    Tests if a point is inside the pentagon by checking if it's in any of the three triangles
    that make up the pentagon. Assumes pentagon is convex.
    @param point The point to test
    @returns true if the point is inside the pentagon
    """
    def contains_point(self, point: vec2) -> bool:
        v0 = self.vertices[0]

        if self.triangles is None:
            self.triangles = []
            # Order triangles by size to increase chance of early return
            for i in [2, 1, 3]:
                v1 = self.vertices[i]
                v2 = self.vertices[i + 1]
                self.triangles.append(Triangle(v0, v1, v2))

        return any(triangle.contains_point(point) for triangle in self.triangles)

    """
    Normalizes longitude values in a contour to handle antimeridian crossing
    @param contour Array of [longitude, latitude] points
    @returns Normalized contour with consistent longitude values
    """
    @staticmethod
    def normalize_longitudes(contour: Contour) -> Contour:
        longitudes = [((lon + 180) % 360 + 360) % 360 - 180 for lon, _ in contour]
        # Calculate the average longitude
        center_lon = sum(longitudes) / len(longitudes)
        # Normalize center longitude to be in the range -180 to 180
        center_lon = ((center_lon + 180) % 360 + 360) % 360 - 180
    
        # Normalize each point relative to center
        normalized = []
        for lon, lat in contour:
            while lon - center_lon > 180:
                lon -= 360
            while lon - center_lon < -180:
                lon += 360
            normalized.append((lon, lat))
        return normalized


class A5Cell(TypedDict):
    origin: Origin
    segment: int
    S: int
    resolution: int


def triangle_area(v1: vec3, v2: vec3, v3: vec3) -> float:
    edge1 = v2 - v1
    edge2 = v3 - v1
    # Calculate cross product
    cross = np.cross(edge1, edge2)
    # Area is half the magnitude of the cross product
    return 0.5 * np.linalg.norm(cross)


def pentagon_area(pentagon: List[vec3]) -> float:
    area = 0.0
    v1 = pentagon[0]
    for i in range(1, 4):
        v2 = pentagon[i]
        v3 = pentagon[i + 1]
        area += abs(triangle_area(v1, v2, v3))
    return area