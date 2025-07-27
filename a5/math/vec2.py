"""
gl-matrix style vec2 operations for A5
Based on https://glmatrix.net/docs/module-vec2.html

All functions follow the gl-matrix convention of having an 'out' parameter
for the result, and return the 'out' parameter for chaining.
"""

import math
from typing import Tuple, Union, List

# Type alias for 2D vectors - can be list or tuple
Vec2 = Union[List[float], Tuple[float, float]]

def create() -> List[float]:
    """
    Creates a new vec2 initialized to [0, 0]
    
    Returns:
        A new vec2
    """
    return [0.0, 0.0]

def clone(a: Vec2) -> List[float]:
    """
    Creates a new vec2 initialized with values from an existing vector
    
    Args:
        a: vector to clone
        
    Returns:
        A new vec2
    """
    return [a[0], a[1]]

def copy(out: Vec2, a: Vec2) -> Vec2:
    """
    Copy the values from one vec2 to another
    
    Args:
        out: the receiving vector
        a: the source vector
        
    Returns:
        out
    """
    out[0] = a[0]
    out[1] = a[1]
    return out

def set(out: Vec2, x: float, y: float) -> Vec2:
    """
    Set the components of a vec2 to the given values
    
    Args:
        out: the receiving vector
        x: X component
        y: Y component
        
    Returns:
        out
    """
    out[0] = x
    out[1] = y
    return out

def add(out: Vec2, a: Vec2, b: Vec2) -> Vec2:
    """
    Adds two vec2's
    
    Args:
        out: the receiving vector
        a: the first operand
        b: the second operand
        
    Returns:
        out
    """
    out[0] = a[0] + b[0]
    out[1] = a[1] + b[1]
    return out

def subtract(out: Vec2, a: Vec2, b: Vec2) -> Vec2:
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
    return out

# Alias for subtract
sub = subtract

def multiply(out: Vec2, a: Vec2, b: Vec2) -> Vec2:
    """
    Multiplies two vec2's (component-wise)
    
    Args:
        out: the receiving vector
        a: the first operand
        b: the second operand
        
    Returns:
        out
    """
    out[0] = a[0] * b[0]
    out[1] = a[1] * b[1]
    return out

# Alias for multiply
mul = multiply

def scale(out: Vec2, a: Vec2, s: float) -> Vec2:
    """
    Scales a vec2 by a scalar number
    
    Args:
        out: the receiving vector
        a: the vector to scale
        s: amount to scale the vector by
        
    Returns:
        out
    """
    out[0] = a[0] * s
    out[1] = a[1] * s
    return out

def dot(a: Vec2, b: Vec2) -> float:
    """
    Calculates the dot product of two vec2's
    
    Args:
        a: the first operand
        b: the second operand
        
    Returns:
        dot product of a and b
    """
    return a[0] * b[0] + a[1] * b[1]

def cross(a: Vec2, b: Vec2) -> float:
    """
    Computes the cross product of two vec2's
    Note that the cross product must by definition produce a 3D vector
    but for 2D vectors we return the Z component
    
    Args:
        a: the first operand
        b: the second operand
        
    Returns:
        Z component of cross product
    """
    return a[0] * b[1] - a[1] * b[0]

def length(a: Vec2) -> float:
    """
    Calculates the length of a vec2
    
    Args:
        a: vector to calculate length of
        
    Returns:
        length of a
    """
    x, y = a[0], a[1]
    return math.sqrt(x * x + y * y)

# Alias for length
len = length

def squaredLength(a: Vec2) -> float:
    """
    Calculates the squared length of a vec2
    
    Args:
        a: vector to calculate squared length of
        
    Returns:
        squared length of a
    """
    x, y = a[0], a[1]
    return x * x + y * y

def normalize(out: Vec2, a: Vec2) -> Vec2:
    """
    Normalize a vec2
    
    Args:
        out: the receiving vector
        a: vector to normalize
        
    Returns:
        out
    """
    x, y = a[0], a[1]
    len_sq = x * x + y * y
    
    if len_sq > 0:
        inv_len = 1.0 / math.sqrt(len_sq)
        out[0] = x * inv_len
        out[1] = y * inv_len
    else:
        out[0] = 0
        out[1] = 0
    
    return out

def distance(a: Vec2, b: Vec2) -> float:
    """
    Calculates the euclidean distance between two vec2's
    
    Args:
        a: the first operand
        b: the second operand
        
    Returns:
        distance between a and b
    """
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    return math.sqrt(dx * dx + dy * dy)

def squaredDistance(a: Vec2, b: Vec2) -> float:
    """
    Calculates the squared euclidean distance between two vec2's
    
    Args:
        a: the first operand
        b: the second operand
        
    Returns:
        squared distance between a and b
    """
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    return dx * dx + dy * dy

def negate(out: Vec2, a: Vec2) -> Vec2:
    """
    Negates the components of a vec2
    
    Args:
        out: the receiving vector
        a: vector to negate
        
    Returns:
        out
    """
    out[0] = -a[0]
    out[1] = -a[1]
    return out

def inverse(out: Vec2, a: Vec2) -> Vec2:
    """
    Returns the inverse of the components of a vec2
    
    Args:
        out: the receiving vector
        a: vector to invert
        
    Returns:
        out
    """
    out[0] = 1.0 / a[0] if a[0] != 0 else 0
    out[1] = 1.0 / a[1] if a[1] != 0 else 0
    return out

def lerp(out: Vec2, a: Vec2, b: Vec2, t: float) -> Vec2:
    """
    Performs a linear interpolation between two vec2's
    
    Args:
        out: the receiving vector
        a: the first operand
        b: the second operand
        t: interpolation amount, in the range [0-1], between the two inputs
        
    Returns:
        out
    """
    ax, ay = a[0], a[1]
    out[0] = ax + t * (b[0] - ax)
    out[1] = ay + t * (b[1] - ay)
    return out

def angle(a: Vec2, b: Vec2) -> float:
    """
    Get the angle between two 2D vectors
    
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

def rotate(out: Vec2, a: Vec2, origin: Vec2, rad: float) -> Vec2:
    """
    Rotate a 2D vector around an origin point
    
    Args:
        out: The receiving vec2
        a: The vec2 point to rotate
        origin: The origin of rotation
        rad: The angle of rotation in radians
        
    Returns:
        out
    """
    # Translate point to origin
    p0 = a[0] - origin[0]
    p1 = a[1] - origin[1]
    
    sin_c = math.sin(rad)
    cos_c = math.cos(rad)
    
    # Perform rotation and translate back
    out[0] = p0 * cos_c - p1 * sin_c + origin[0]
    out[1] = p0 * sin_c + p1 * cos_c + origin[1]
    
    return out

def exactEquals(a: Vec2, b: Vec2) -> bool:
    """
    Returns whether or not the vectors have exactly the same elements in the same position (when compared with ===)
    
    Args:
        a: The first vector
        b: The second vector
        
    Returns:
        True if the vectors are equal, False otherwise
    """
    return a[0] == b[0] and a[1] == b[1]

def equals(a: Vec2, b: Vec2, epsilon: float = 1e-6) -> bool:
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
            abs(a[1] - b[1]) <= epsilon)

def transformMat2(out: Vec2, a: Vec2, m: List[float]) -> Vec2:
    """
    Transforms the vec2 with a mat2
    
    Args:
        out: the receiving vector
        a: the vector to transform
        m: matrix to transform with (4 elements in column-major order)
        
    Returns:
        out
    """
    x, y = a[0], a[1]
    out[0] = m[0] * x + m[2] * y
    out[1] = m[1] * x + m[3] * y
    return out

def zero(out: Vec2) -> Vec2:
    """
    Set a vec2 to the zero vector
    
    Args:
        out: the receiving vector
        
    Returns:
        out
    """
    out[0] = 0.0
    out[1] = 0.0
    return out