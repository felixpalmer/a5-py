"""
gl-matrix style vec3 operations for A5
Based on https://glmatrix.net/docs/module-vec3.html

All functions follow the gl-matrix convention of having an 'out' parameter
for the result, and return the 'out' parameter for chaining.
"""

import math
from typing import Tuple, Union, List

# Type alias for 3D vectors - can be list or tuple
Vec3 = Union[List[float], Tuple[float, float, float]]

def create() -> List[float]:
    """
    Creates a new vec3 initialized to [0, 0, 0]
    
    Returns:
        A new vec3
    """
    return [0.0, 0.0, 0.0]

def clone(a: Vec3) -> List[float]:
    """
    Creates a new vec3 initialized with values from an existing vector
    
    Args:
        a: vector to clone
        
    Returns:
        A new vec3
    """
    return [a[0], a[1], a[2]]

def copy(out: Vec3, a: Vec3) -> Vec3:
    """
    Copy the values from one vec3 to another
    
    Args:
        out: the receiving vector
        a: the source vector
        
    Returns:
        out
    """
    out[0] = a[0]
    out[1] = a[1]
    out[2] = a[2]
    return out

def set(out: Vec3, x: float, y: float, z: float) -> Vec3:
    """
    Set the components of a vec3 to the given values
    
    Args:
        out: the receiving vector
        x: X component
        y: Y component
        z: Z component
        
    Returns:
        out
    """
    out[0] = x
    out[1] = y
    out[2] = z
    return out

def add(out: Vec3, a: Vec3, b: Vec3) -> Vec3:
    """
    Adds two vec3's
    
    Args:
        out: the receiving vector
        a: the first operand
        b: the second operand
        
    Returns:
        out
    """
    out[0] = a[0] + b[0]
    out[1] = a[1] + b[1]
    out[2] = a[2] + b[2]
    return out

def subtract(out: Vec3, a: Vec3, b: Vec3) -> Vec3:
    """
    Subtracts vector b from vector a
    
    Args:
        out: the receiving vector
        a: the first operand
        b: the second operand
        
    Returns:
        out
    """
    out[0] = a[0] - b[0]
    out[1] = a[1] - b[1]
    out[2] = a[2] - b[2]
    return out

# Alias for subtract
sub = subtract

def multiply(out: Vec3, a: Vec3, b: Vec3) -> Vec3:
    """
    Multiplies two vec3's (component-wise)
    
    Args:
        out: the receiving vector
        a: the first operand
        b: the second operand
        
    Returns:
        out
    """
    out[0] = a[0] * b[0]
    out[1] = a[1] * b[1]
    out[2] = a[2] * b[2]
    return out

# Alias for multiply
mul = multiply

def scale(out: Vec3, a: Vec3, s: float) -> Vec3:
    """
    Scales a vec3 by a scalar number
    
    Args:
        out: the receiving vector
        a: the vector to scale
        s: amount to scale the vector by
        
    Returns:
        out
    """
    out[0] = a[0] * s
    out[1] = a[1] * s
    out[2] = a[2] * s
    return out

def dot(a: Vec3, b: Vec3) -> float:
    """
    Calculates the dot product of two vec3's
    
    Args:
        a: the first operand
        b: the second operand
        
    Returns:
        dot product of a and b
    """
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

def cross(out: Vec3, a: Vec3, b: Vec3) -> Vec3:
    """
    Computes the cross product of two vec3's
    
    Args:
        out: the receiving vector
        a: the first operand
        b: the second operand
        
    Returns:
        out
    """
    ax, ay, az = a[0], a[1], a[2]
    bx, by, bz = b[0], b[1], b[2]
    
    out[0] = ay * bz - az * by
    out[1] = az * bx - ax * bz
    out[2] = ax * by - ay * bx
    return out

def length(a: Vec3) -> float:
    """
    Calculates the length of a vec3
    
    Args:
        a: vector to calculate length of
        
    Returns:
        length of a
    """
    x, y, z = a[0], a[1], a[2]
    return math.sqrt(x * x + y * y + z * z)

# Alias for length
len = length

def squaredLength(a: Vec3) -> float:
    """
    Calculates the squared length of a vec3
    
    Args:
        a: vector to calculate squared length of
        
    Returns:
        squared length of a
    """
    x, y, z = a[0], a[1], a[2]
    return x * x + y * y + z * z

def normalize(out: Vec3, a: Vec3) -> Vec3:
    """
    Normalize a vec3
    
    Args:
        out: the receiving vector
        a: vector to normalize
        
    Returns:
        out
    """
    x, y, z = a[0], a[1], a[2]
    len_sq = x * x + y * y + z * z
    
    if len_sq > 0:
        inv_len = 1.0 / math.sqrt(len_sq)
        out[0] = x * inv_len
        out[1] = y * inv_len
        out[2] = z * inv_len
    else:
        out[0] = 0
        out[1] = 0
        out[2] = 0
    
    return out

def distance(a: Vec3, b: Vec3) -> float:
    """
    Calculates the euclidean distance between two vec3's
    
    Args:
        a: the first operand
        b: the second operand
        
    Returns:
        distance between a and b
    """
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    dz = b[2] - a[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)

def squaredDistance(a: Vec3, b: Vec3) -> float:
    """
    Calculates the squared euclidean distance between two vec3's
    
    Args:
        a: the first operand
        b: the second operand
        
    Returns:
        squared distance between a and b
    """
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    dz = b[2] - a[2]
    return dx * dx + dy * dy + dz * dz

def negate(out: Vec3, a: Vec3) -> Vec3:
    """
    Negates the components of a vec3
    
    Args:
        out: the receiving vector
        a: vector to negate
        
    Returns:
        out
    """
    out[0] = -a[0]
    out[1] = -a[1]
    out[2] = -a[2]
    return out

def inverse(out: Vec3, a: Vec3) -> Vec3:
    """
    Returns the inverse of the components of a vec3
    
    Args:
        out: the receiving vector
        a: vector to invert
        
    Returns:
        out
    """
    out[0] = 1.0 / a[0] if a[0] != 0 else 0
    out[1] = 1.0 / a[1] if a[1] != 0 else 0
    out[2] = 1.0 / a[2] if a[2] != 0 else 0
    return out

def lerp(out: Vec3, a: Vec3, b: Vec3, t: float) -> Vec3:
    """
    Performs a linear interpolation between two vec3's
    
    Args:
        out: the receiving vector
        a: the first operand
        b: the second operand
        t: interpolation amount, in the range [0-1], between the two inputs
        
    Returns:
        out
    """
    ax, ay, az = a[0], a[1], a[2]
    out[0] = ax + t * (b[0] - ax)
    out[1] = ay + t * (b[1] - ay)
    out[2] = az + t * (b[2] - az)
    return out

def angle(a: Vec3, b: Vec3) -> float:
    """
    Get the angle between two 3D vectors
    
    Args:
        a: The first operand
        b: The second operand
        
    Returns:
        The angle in radians
    """
    # Normalize both vectors
    temp_a = normalize(create(), a)
    temp_b = normalize(create(), b)
    
    cos_angle = dot(temp_a, temp_b)
    
    # Clamp to avoid numerical errors
    cos_angle = max(-1.0, min(1.0, cos_angle))
    
    return math.acos(cos_angle)

def exactEquals(a: Vec3, b: Vec3) -> bool:
    """
    Returns whether or not the vectors have exactly the same elements in the same position (when compared with ===)
    
    Args:
        a: The first vector
        b: The second vector
        
    Returns:
        True if the vectors are equal, False otherwise
    """
    return a[0] == b[0] and a[1] == b[1] and a[2] == b[2]

def equals(a: Vec3, b: Vec3, epsilon: float = 1e-6) -> bool:
    """
    Returns whether or not the vectors have approximately the same elements in the same position
    
    Args:
        a: The first vector
        b: The second vector
        epsilon: The comparison epsilon
        
    Returns:
        True if the vectors are equal, False otherwise
    """
    return (abs(a[0] - b[0]) <= epsilon and
            abs(a[1] - b[1]) <= epsilon and
            abs(a[2] - b[2]) <= epsilon)

def transformQuat(out: Vec3, a: Vec3, q: List[float]) -> Vec3:
    """
    Transforms the vec3 with a quat
    
    Args:
        out: the receiving vector
        a: the vector to transform
        q: quaternion to transform with [x, y, z, w]
        
    Returns:
        out
    """
    # Get quaternion components
    qx, qy, qz, qw = q[0], q[1], q[2], q[3]
    x, y, z = a[0], a[1], a[2]

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
    out[0] = x + uvx + uuvx
    out[1] = y + uvy + uuvy
    out[2] = z + uvz + uuvz
    return out

def zero(out: Vec3) -> Vec3:
    """
    Set a vec3 to the zero vector
    
    Args:
        out: the receiving vector
        
    Returns:
        out
    """
    out[0] = 0.0
    out[1] = 0.0
    out[2] = 0.0
    return out