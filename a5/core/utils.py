# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

import math
from typing import List, Tuple, Optional, TypedDict, NamedTuple, Literal
from .coordinate_systems import Radians, LonLat, Face, Spherical
from .hilbert import Orientation
from dataclasses import dataclass

# Type aliases for vectors and matrices (now pure Python)
vec2 = Tuple[float, float]
vec3 = Tuple[float, float, float]
mat2 = Tuple[Tuple[float, float], Tuple[float, float]]
mat2d = Tuple[Tuple[float, float, float], Tuple[float, float, float]]

OriginId = Literal[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

class Origin(NamedTuple):
    id: OriginId
    axis: Spherical
    quat: Tuple[float, float, float, float]
    angle: Radians
    orientation: List[Orientation]
    first_quintant: int

Contour = List[LonLat]

class A5Cell(TypedDict):
    """
    A5 cell with its position information
    """
    origin: Origin  # Origin representing one of pentagon face of the dodecahedron
    segment: int    # Index (0-4) of triangular segment within pentagonal dodecahedron face  
    S: int         # Position along Hilbert curve within triangular segment
    resolution: int # Resolution of the cell

def triangle_area(v1: vec3, v2: vec3, v3: vec3) -> float:
    """Calculate the area of a triangle given three 3D vertices."""
    # Calculate edge vectors
    edge1 = (v2[0] - v1[0], v2[1] - v1[1], v2[2] - v1[2])
    edge2 = (v3[0] - v1[0], v3[1] - v1[1], v3[2] - v1[2])
    
    # Calculate cross product
    cross_x = edge1[1] * edge2[2] - edge1[2] * edge2[1]
    cross_y = edge1[2] * edge2[0] - edge1[0] * edge2[2]
    cross_z = edge1[0] * edge2[1] - edge1[1] * edge2[0]
    
    # Calculate magnitude of cross product
    magnitude = math.sqrt(cross_x * cross_x + cross_y * cross_y + cross_z * cross_z)
    
    # Area is half the magnitude of the cross product
    return 0.5 * magnitude

def pentagon_area(pentagon: List[vec3]) -> float:
    """Calculate the area of a pentagon by triangulation."""
    area = 0.0
    v1 = pentagon[0]
    for i in range(1, 4):
        v2 = pentagon[i]
        v3 = pentagon[i + 1]
        area += abs(triangle_area(v1, v2, v3))
    return area