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

def conjugate(q: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    """
    Calculate the conjugate of a quaternion.
    
    Args:
        q: The quaternion [x, y, z, w]
        
    Returns:
        The conjugate quaternion [-x, -y, -z, w]
    """
    return (-q[0], -q[1], -q[2], q[3])

def transform_quat(a: Cartesian, q: Tuple[float, float, float, float]) -> Cartesian:
    """
    Transform a vector by a quaternion.
    
    Args:
        a: The vector to transform [x, y, z]
        q: The quaternion [x, y, z, w]
        
    Returns:
        The transformed vector [x, y, z]
    """
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
    rad = rad * 0.5
    s = math.sin(rad)
    return (
        s * axis[0],
        s * axis[1],
        s * axis[2],
        math.cos(rad)
    )

def _cross_product(a: Cartesian, b: Cartesian) -> Cartesian:
    """Cross product of two 3D vectors."""
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2], 
        a[0] * b[1] - a[1] * b[0]
    )

def _dot_product(a: Cartesian, b: Cartesian) -> float:
    """Dot product of two 3D vectors."""
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

def _norm(v: Cartesian) -> float:
    """Magnitude of a 3D vector."""
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])

def _normalize(v: Cartesian) -> Cartesian:
    """Normalize a 3D vector."""
    norm = _norm(v)
    return (v[0] / norm, v[1] / norm, v[2] / norm)

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
    dot = _dot_product(a, b)
    
    if dot < -0.999999:
        # Vectors are nearly opposite, use x-axis as reference
        tmpvec = _cross_product((1, 0, 0), a)
        if _norm(tmpvec) < 0.000001:
            # If x-axis is parallel to a, use y-axis
            tmpvec = _cross_product((0, 1, 0), a)
        tmpvec = _normalize(tmpvec)
        return set_axis_angle(tmpvec, math.pi)
    elif dot > 0.999999:
        # Vectors are nearly parallel, return identity quaternion
        return (0, 0, 0, 1)
    else:
        # Normal case
        tmpvec = _cross_product(a, b)
        out = (tmpvec[0], tmpvec[1], tmpvec[2], 1 + dot)
        # Normalize
        norm = math.sqrt(out[0] * out[0] + out[1] * out[1] + out[2] * out[2] + out[3] * out[3])
        return (out[0] / norm, out[1] / norm, out[2] / norm, out[3] / norm)