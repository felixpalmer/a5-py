"""
Quaternion helper functions

Portions of this code are derived from https://github.com/toji/gl-matrix

Copyright (c) 2015-2021, Brandon Jones, Colin MacKenzie IV.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import math
from typing import Tuple
from .coordinate_systems import Cartesian
from ..math import quat as quat_glm, vec3

def conjugate(q: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    """
    Calculate the conjugate of a quaternion.
    
    Args:
        q: The quaternion [x, y, z, w]
        
    Returns:
        The conjugate quaternion [-x, -y, -z, w]
    """
    out = quat_glm.create()
    quat_glm.conjugate(out, q)
    return (out[0], out[1], out[2], out[3])

def transform_quat(a: Cartesian, q: Tuple[float, float, float, float]) -> Cartesian:
    """
    Transform a vector by a quaternion.
    
    Args:
        a: The vector to transform [x, y, z]
        q: The quaternion [x, y, z, w]
        
    Returns:
        The transformed vector [x, y, z]
    """
    # Use gl-matrix style calculation
    qx, qy, qz, qw = q
    x, y, z = a

    # Calculate cross product q × a
    uvx = qy * z - qz * y
    uvy = qz * x - qx * z
    uvz = qx * y - qy * x

    # Calculate cross product q × (q × a)
    uuvx = qy * uvz - qz * uvy
    uuvy = qz * uvx - qx * uvz
    uuvz = qx * uvy - qy * uvx

    # Scale uv by 2 * w
    w2 = qw * 2
    uvx *= w2
    uvy *= w2
    uvz *= w2

    # Scale uuv by 2
    uuvx *= 2
    uuvy *= 2
    uuvz *= 2

    # Add all components
    return (
        x + uvx + uuvx,
        y + uvy + uuvy,
        z + uvz + uuvz
    )

def set_axis_angle(axis: Cartesian, rad: float) -> Tuple[float, float, float, float]:
    """
    Sets a quaternion from the given angle and rotation axis.
    
    Args:
        axis: The axis around which to rotate [x, y, z]
        rad: The angle in radians
        
    Returns:
        The quaternion [x, y, z, w]
    """
    out = quat_glm.create()
    quat_glm.fromAxisAngle(out, axis, rad)
    return (out[0], out[1], out[2], out[3])

def _cross_product(a: Cartesian, b: Cartesian) -> Cartesian:
    """Cross product of two 3D vectors."""
    temp_a = vec3.create()
    temp_b = vec3.create()
    temp_out = vec3.create()
    vec3.copy(temp_a, a)
    vec3.copy(temp_b, b)
    vec3.cross(temp_out, temp_a, temp_b)
    return (temp_out[0], temp_out[1], temp_out[2])

def _dot_product(a: Cartesian, b: Cartesian) -> float:
    """Dot product of two 3D vectors."""
    temp_a = vec3.create()
    temp_b = vec3.create()
    vec3.copy(temp_a, a)
    vec3.copy(temp_b, b)
    return vec3.dot(temp_a, temp_b)

def _norm(v: Cartesian) -> float:
    """Magnitude of a 3D vector."""
    temp_v = vec3.create()
    vec3.copy(temp_v, v)
    return vec3.length(temp_v)

def _normalize(v: Cartesian) -> Cartesian:
    """Normalize a 3D vector."""
    temp_v = vec3.create()
    vec3.copy(temp_v, v)
    vec3.normalize(temp_v, temp_v)
    return (temp_v[0], temp_v[1], temp_v[2])

def rotation_to(a: Cartesian, b: Cartesian) -> Tuple[float, float, float, float]:
    """
    Sets a quaternion to represent the shortest rotation from one vector to another.
    Both vectors are assumed to be unit length.
    
    Args:
        a: The initial vector [x, y, z]
        b: The destination vector [x, y, z]
        
    Returns:
        The quaternion [x, y, z, w]
    """
    out = quat_glm.create()
    quat_glm.rotationTo(out, a, b)
    return (out[0], out[1], out[2], out[3])