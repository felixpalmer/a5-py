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

Contour = List[LonLat]




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