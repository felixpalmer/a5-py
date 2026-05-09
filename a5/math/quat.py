# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

"""
Quaternion operations following gl-matrix API patterns
https://glmatrix.net/docs/module-quat.html

Portions of this code are derived from https://github.com/toji/gl-matrix
"""

import math
from typing import List, Union
from . import vec3

# Type alias for quaternions - [x, y, z, w]
Quat = Union[List[float], tuple]

def create() -> List[float]:
    """
    Creates a new identity quat
    
    Returns:
        A new quaternion [x, y, z, w]
    """
    return [0.0, 0.0, 0.0, 1.0]

def length(a: Quat) -> float:
    """
    Calculates the length of a quat
    
    Args:
        a: quaternion to calculate length of
        
    Returns:
        length of a
    """
    x, y, z, w = a[0], a[1], a[2], a[3]
    return math.sqrt(x * x + y * y + z * z + w * w)

# Alias for length
len = length

def set_axis_angle(out: Quat, axis: List[float], rad: float) -> Quat:
    """
    Sets a quat from the given angle and rotation axis,
    then returns it.
    """
    rad = rad * 0.5
    s = math.sin(rad)
    out[0] = s * axis[0]
    out[1] = s * axis[1]
    out[2] = s * axis[2]
    out[3] = math.cos(rad)
    return out

def normalize(out: Quat, a: Quat) -> Quat:
    """Normalize a quat."""
    x, y, z, w = a[0], a[1], a[2], a[3]
    len_sq = x * x + y * y + z * z + w * w
    if len_sq > 0:
        inv = 1.0 / math.sqrt(len_sq)
        out[0] = x * inv
        out[1] = y * inv
        out[2] = z * inv
        out[3] = w * inv
    else:
        out[0] = 0.0
        out[1] = 0.0
        out[2] = 0.0
        out[3] = 0.0
    return out

# Module-level scratch vectors for rotation_to (mirrors gl-matrix closure state).
_rt_tmp = [0.0, 0.0, 0.0]
_rt_x_unit = [1.0, 0.0, 0.0]
_rt_y_unit = [0.0, 1.0, 0.0]

def rotation_to(out: Quat, a: List[float], b: List[float]) -> Quat:
    """
    Sets a quaternion to represent the shortest rotation from one
    vector to another.
    Both vectors are assumed to be unit length.
    """
    dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    if dot < -0.999999:
        # Cross with x-axis
        _rt_tmp[0] = _rt_x_unit[1] * a[2] - _rt_x_unit[2] * a[1]
        _rt_tmp[1] = _rt_x_unit[2] * a[0] - _rt_x_unit[0] * a[2]
        _rt_tmp[2] = _rt_x_unit[0] * a[1] - _rt_x_unit[1] * a[0]
        tmp_len = math.sqrt(_rt_tmp[0] ** 2 + _rt_tmp[1] ** 2 + _rt_tmp[2] ** 2)
        if tmp_len < 0.000001:
            _rt_tmp[0] = _rt_y_unit[1] * a[2] - _rt_y_unit[2] * a[1]
            _rt_tmp[1] = _rt_y_unit[2] * a[0] - _rt_y_unit[0] * a[2]
            _rt_tmp[2] = _rt_y_unit[0] * a[1] - _rt_y_unit[1] * a[0]
            tmp_len = math.sqrt(_rt_tmp[0] ** 2 + _rt_tmp[1] ** 2 + _rt_tmp[2] ** 2)
        if tmp_len > 0:
            _rt_tmp[0] /= tmp_len
            _rt_tmp[1] /= tmp_len
            _rt_tmp[2] /= tmp_len
        return set_axis_angle(out, _rt_tmp, math.pi)
    elif dot > 0.999999:
        out[0] = 0.0
        out[1] = 0.0
        out[2] = 0.0
        out[3] = 1.0
        return out
    else:
        out[0] = a[1] * b[2] - a[2] * b[1]
        out[1] = a[2] * b[0] - a[0] * b[2]
        out[2] = a[0] * b[1] - a[1] * b[0]
        out[3] = 1 + dot
        return normalize(out, out)

def conjugate(out: Quat, a: Quat) -> Quat:
    """
    Calculates the conjugate of a quat
    If the quaternion is normalized, this function is faster than quat.inverse and produces the same result.
    
    Args:
        out: the receiving quaternion
        a: quat to calculate conjugate of
        
    Returns:
        out
    """
    out[0] = -a[0]
    out[1] = -a[1]
    out[2] = -a[2]
    out[3] = a[3]
    return out

