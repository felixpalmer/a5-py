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
    inverse_quat: Tuple[float, float, float, float]
    angle: Radians
    orientation: List[Orientation]
    first_quintant: int

class A5Cell(TypedDict):
    """
    A5 cell with its position information
    """
    origin: Origin  # Origin representing one of pentagon face of the dodecahedron
    segment: int    # Index (0-4) of triangular segment within pentagonal dodecahedron face  
    S: int         # Position along Hilbert curve within triangular segment
    resolution: int # Resolution of the cell