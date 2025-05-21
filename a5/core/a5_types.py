"""
A5
SPDX-License-Identifier: Apache-2.0
Copyright (c) A5 contributors
"""

import numpy as np
from typing import NewType, Tuple, TypeVar, Union, Any

# Base types
Degrees = NewType('Degrees', float)
Radians = NewType('Radians', float)

# Vector types
Vec2 = np.ndarray[Any, np.dtype[np.float64]]  # 2D vector
Vec3 = np.ndarray[Any, np.dtype[np.float64]]  # 3D vector

# 2D coordinate systems
"""
2D cartesian coordinate system with origin at the center of a dodecahedron face
"""
Face = NewType('Face', Vec2)

"""
2D polar coordinate system with origin at the center of a dodecahedron face
"""
Polar = NewType('Polar', Tuple[float, Radians])

"""
2D planar coordinate system defined by the eigenvectors of the lattice tiling
"""
IJ = NewType('IJ', Vec2)

"""
2D planar coordinate system formed by the transformation K -> I + J
"""
KJ = NewType('KJ', Vec2)

# 3D coordinate systems
"""
3D cartesian system centered on unit sphere/dodecahedron
"""
Cartesian = NewType('Cartesian', Vec3)

"""
3D spherical coordinate system centered on unit sphere/dodecahedron
"""
Spherical = NewType('Spherical', Tuple[Radians, Radians])

"""
Geographic longitude & latitude
"""
LonLat = NewType('LonLat', Tuple[Degrees, Degrees]) 