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