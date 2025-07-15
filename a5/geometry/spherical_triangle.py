# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

from typing import List
from ..core.coordinate_systems import Cartesian
from .spherical_polygon import SphericalPolygonShape

# Type alias for clarity
SphericalTriangle = List[Cartesian]


class SphericalTriangleShape(SphericalPolygonShape):
    """
    A spherical triangle is a spherical polygon with exactly 3 vertices
    """
    
    def __init__(self, vertices: SphericalTriangle):
        if len(vertices) != 3:
            raise ValueError('SphericalTriangleShape requires exactly 3 vertices')
        super().__init__(vertices) 