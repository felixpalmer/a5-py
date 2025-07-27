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

def identity(out: Quat) -> Quat:
    """
    Set a quat to the identity quaternion
    
    Args:
        out: the receiving quaternion
        
    Returns:
        out
    """
    out[0] = 0.0
    out[1] = 0.0
    out[2] = 0.0
    out[3] = 1.0
    return out

def copy(out: Quat, a: Quat) -> Quat:
    """
    Copy the values from one quat to another
    
    Args:
        out: the receiving quaternion
        a: the source quaternion
        
    Returns:
        out
    """
    out[0] = a[0]
    out[1] = a[1]
    out[2] = a[2]
    out[3] = a[3]
    return out

def set(out: Quat, x: float, y: float, z: float, w: float) -> Quat:
    """
    Set the components of a quat to the given values
    
    Args:
        out: the receiving quaternion
        x: X component
        y: Y component
        z: Z component
        w: W component
        
    Returns:
        out
    """
    out[0] = x
    out[1] = y
    out[2] = z
    out[3] = w
    return out

def add(out: Quat, a: Quat, b: Quat) -> Quat:
    """
    Adds two quat's
    
    Args:
        out: the receiving quaternion
        a: the first operand
        b: the second operand
        
    Returns:
        out
    """
    out[0] = a[0] + b[0]
    out[1] = a[1] + b[1]
    out[2] = a[2] + b[2]
    out[3] = a[3] + b[3]
    return out

def multiply(out: Quat, a: Quat, b: Quat) -> Quat:
    """
    Multiplies two quat's
    
    Args:
        out: the receiving quaternion
        a: the first operand
        b: the second operand
        
    Returns:
        out
    """
    ax, ay, az, aw = a[0], a[1], a[2], a[3]
    bx, by, bz, bw = b[0], b[1], b[2], b[3]
    
    out[0] = ax * bw + aw * bx + ay * bz - az * by
    out[1] = ay * bw + aw * by + az * bx - ax * bz
    out[2] = az * bw + aw * bz + ax * by - ay * bx
    out[3] = aw * bw - ax * bx - ay * by - az * bz
    return out

# Alias for multiply
mul = multiply

def scale(out: Quat, a: Quat, s: float) -> Quat:
    """
    Scales a quat by a scalar number
    
    Args:
        out: the receiving quaternion
        a: the quaternion to scale
        s: amount to scale the quaternion by
        
    Returns:
        out
    """
    out[0] = a[0] * s
    out[1] = a[1] * s
    out[2] = a[2] * s
    out[3] = a[3] * s
    return out

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

def squaredLength(a: Quat) -> float:
    """
    Calculates the squared length of a quat
    
    Args:
        a: quaternion to calculate squared length of
        
    Returns:
        squared length of a
    """
    x, y, z, w = a[0], a[1], a[2], a[3]
    return x * x + y * y + z * z + w * w

def normalize(out: Quat, a: Quat) -> Quat:
    """
    Normalize a quat
    
    Args:
        out: the receiving quaternion
        a: quaternion to normalize
        
    Returns:
        out
    """
    x, y, z, w = a[0], a[1], a[2], a[3]
    len_sq = x * x + y * y + z * z + w * w
    
    if len_sq > 0:
        inv_len = 1.0 / math.sqrt(len_sq)
        out[0] = x * inv_len
        out[1] = y * inv_len
        out[2] = z * inv_len
        out[3] = w * inv_len
    else:
        out[0] = 0
        out[1] = 0
        out[2] = 0
        out[3] = 1
    
    return out

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

def invert(out: Quat, a: Quat) -> Quat:
    """
    Calculates the inverse of a quat
    
    Args:
        out: the receiving quaternion
        a: quat to calculate inverse of
        
    Returns:
        out
    """
    a0, a1, a2, a3 = a[0], a[1], a[2], a[3]
    dot = a0 * a0 + a1 * a1 + a2 * a2 + a3 * a3
    
    if dot == 0:
        out[0] = 0
        out[1] = 0
        out[2] = 0
        out[3] = 1
        return out
    
    inv_dot = 1.0 / dot
    out[0] = -a0 * inv_dot
    out[1] = -a1 * inv_dot
    out[2] = -a2 * inv_dot
    out[3] = a3 * inv_dot
    return out

def dot(a: Quat, b: Quat) -> float:
    """
    Calculates the dot product of two quat's
    
    Args:
        a: the first operand
        b: the second operand
        
    Returns:
        dot product of a and b
    """
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]

def lerp(out: Quat, a: Quat, b: Quat, t: float) -> Quat:
    """
    Performs a linear interpolation between two quat's
    
    Args:
        out: the receiving quaternion
        a: the first operand
        b: the second operand
        t: interpolation amount, in the range [0-1], between the two inputs
        
    Returns:
        out
    """
    ax, ay, az, aw = a[0], a[1], a[2], a[3]
    out[0] = ax + t * (b[0] - ax)
    out[1] = ay + t * (b[1] - ay)
    out[2] = az + t * (b[2] - az)
    out[3] = aw + t * (b[3] - aw)
    return out

def slerp(out: Quat, a: Quat, b: Quat, t: float) -> Quat:
    """
    Performs a spherical linear interpolation between two quat
    
    Args:
        out: the receiving quaternion
        a: the first operand
        b: the second operand
        t: interpolation amount, in the range [0-1], between the two inputs
        
    Returns:
        out
    """
    # Calc cosine
    cosom = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
    
    # Adjust signs (if necessary)
    if cosom < 0.0:
        cosom = -cosom
        to_x = -b[0]
        to_y = -b[1]
        to_z = -b[2]
        to_w = -b[3]
    else:
        to_x = b[0]
        to_y = b[1]
        to_z = b[2]
        to_w = b[3]
    
    # Calculate coefficients
    if (1.0 - cosom) > 0.000001:
        # Standard case (slerp)
        omega = math.acos(cosom)
        sinom = math.sin(omega)
        scale0 = math.sin((1.0 - t) * omega) / sinom
        scale1 = math.sin(t * omega) / sinom
    else:
        # "from" and "to" quaternions are very close 
        # ... so we can do a linear interpolation
        scale0 = 1.0 - t
        scale1 = t
    
    # Calculate final values
    out[0] = scale0 * a[0] + scale1 * to_x
    out[1] = scale0 * a[1] + scale1 * to_y
    out[2] = scale0 * a[2] + scale1 * to_z
    out[3] = scale0 * a[3] + scale1 * to_w
    
    return out

def fromAxisAngle(out: Quat, axis: List[float], rad: float) -> Quat:
    """
    Sets a quat from the given angle and rotation axis,
    then returns it.
    
    Args:
        out: the receiving quaternion
        axis: the axis around which to rotate
        rad: the angle in radians
        
    Returns:
        out
    """
    rad = rad * 0.5
    s = math.sin(rad)
    out[0] = s * axis[0]
    out[1] = s * axis[1]
    out[2] = s * axis[2]
    out[3] = math.cos(rad)
    return out

def rotationTo(out: Quat, a: List[float], b: List[float]) -> Quat:
    """
    Sets a quaternion to represent the shortest rotation from one
    vector to another.
    
    Both vectors are assumed to be unit length.
    
    Args:
        out: the receiving quaternion
        a: the initial vector
        b: the destination vector
        
    Returns:
        out
    """
    # Use temp vectors for calculations
    temp_a = vec3.create()
    temp_b = vec3.create()
    temp_cross = vec3.create()
    
    vec3.copy(temp_a, a)
    vec3.copy(temp_b, b)
    
    dot_product = vec3.dot(temp_a, temp_b)
    
    if dot_product < -0.999999:
        # Vectors are nearly opposite
        # Find a perpendicular axis
        vec3.cross(temp_cross, [1, 0, 0], temp_a)
        if vec3.length(temp_cross) < 0.000001:
            # If a is parallel to x-axis, use y-axis
            vec3.cross(temp_cross, [0, 1, 0], temp_a)
        
        vec3.normalize(temp_cross, temp_cross)
        fromAxisAngle(out, temp_cross, math.pi)
        return out
    elif dot_product > 0.999999:
        # Vectors are nearly parallel
        identity(out)
        return out
    else:
        # Normal case
        vec3.cross(temp_cross, temp_a, temp_b)
        out[0] = temp_cross[0]
        out[1] = temp_cross[1]
        out[2] = temp_cross[2]
        out[3] = 1 + dot_product
        return normalize(out, out)

def exactEquals(a: Quat, b: Quat) -> bool:
    """
    Returns whether or not the quaternions have exactly the same elements in the same position (when compared with ===)
    
    Args:
        a: The first quaternion
        b: The second quaternion
        
    Returns:
        True if the quaternions are equal, False otherwise
    """
    return a[0] == b[0] and a[1] == b[1] and a[2] == b[2] and a[3] == b[3]

def equals(a: Quat, b: Quat, epsilon: float = 1e-6) -> bool:
    """
    Returns whether or not the quaternions have approximately the same elements
    
    Args:
        a: The first quaternion
        b: The second quaternion
        epsilon: The comparison epsilon
        
    Returns:
        True if the quaternions are equal, False otherwise
    """
    return (abs(a[0] - b[0]) <= epsilon and
            abs(a[1] - b[1]) <= epsilon and
            abs(a[2] - b[2]) <= epsilon and
            abs(a[3] - b[3]) <= epsilon)

def slerp(out: List[float], temp_a: List[float], temp_b: List[float], 
         temp_scaled_a: List[float], temp_scaled_b: List[float],
         A: "Cartesian", B: "Cartesian", t: float) -> "Cartesian":
    """
    Spherical linear interpolation between two 3D vectors
    
    Args:
        out, temp_a, temp_b, temp_scaled_a, temp_scaled_b: temporary vectors
        A: first vector
        B: second vector
        t: interpolation parameter (0 to 1)
        
    Returns:
        interpolated vector as tuple
    """
    import math
    from . import vec3
    
    vec3.copy(temp_a, A)
    vec3.copy(temp_b, B)
    
    # Calculate angle between vectors
    gamma = vec3.angle(temp_a, temp_b)
    
    if gamma < 1e-12:
        # Vectors are very close, use linear interpolation
        vec3.lerp(out, temp_a, temp_b, t)
    else:
        sin_gamma = math.sin(gamma)
        weight_a = math.sin((1 - t) * gamma) / sin_gamma
        weight_b = math.sin(t * gamma) / sin_gamma
        
        # Compute weighted sum
        vec3.scale(temp_scaled_a, temp_a, weight_a)
        vec3.scale(temp_scaled_b, temp_b, weight_b)
        vec3.add(out, temp_scaled_a, temp_scaled_b)
    
    return (out[0], out[1], out[2])