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

