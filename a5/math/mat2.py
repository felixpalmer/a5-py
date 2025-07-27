# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

"""
2x2 matrix operations following gl-matrix API patterns
https://glmatrix.net/docs/module-mat2.html
"""

from typing import List

# Type alias for 2x2 matrix (4 elements in column-major order)
Mat2 = List[float]

def create() -> Mat2:
    """Creates a new identity mat2"""
    return [1.0, 0.0, 0.0, 1.0]

def copy(out: Mat2, a: Mat2) -> Mat2:
    """Copy the values from one mat2 to another"""
    out[0] = a[0]
    out[1] = a[1] 
    out[2] = a[2]
    out[3] = a[3]
    return out

def identity(out: Mat2) -> Mat2:
    """Set a mat2 to the identity matrix"""
    out[0] = 1.0
    out[1] = 0.0
    out[2] = 0.0
    out[3] = 1.0
    return out

def transpose(out: Mat2, a: Mat2) -> Mat2:
    """Transpose the values of a mat2"""
    # If we are transposing ourselves we can skip a few steps
    if out is a:
        a01 = a[1]
        out[1] = a[2]
        out[2] = a01
    else:
        out[0] = a[0]
        out[1] = a[2]
        out[2] = a[1]
        out[3] = a[3]
    return out

def invert(out: Mat2, a: Mat2) -> Mat2 | None:
    """Inverts a mat2"""
    a0 = a[0]
    a1 = a[1] 
    a2 = a[2]
    a3 = a[3]
    
    det = a0 * a3 - a2 * a1
    
    if det == 0.0:
        return None
        
    det = 1.0 / det
    
    out[0] = a3 * det
    out[1] = -a1 * det
    out[2] = -a2 * det
    out[3] = a0 * det
    
    return out

def determinant(a: Mat2) -> float:
    """Calculates the determinant of a mat2"""
    return a[0] * a[3] - a[2] * a[1]

def multiply(out: Mat2, a: Mat2, b: Mat2) -> Mat2:
    """Multiplies two mat2s"""
    a0 = a[0]
    a1 = a[1]
    a2 = a[2] 
    a3 = a[3]
    b0 = b[0]
    b1 = b[1]
    b2 = b[2]
    b3 = b[3]
    
    out[0] = a0 * b0 + a2 * b1
    out[1] = a1 * b0 + a3 * b1
    out[2] = a0 * b2 + a2 * b3
    out[3] = a1 * b2 + a3 * b3
    
    return out

def rotate(out: Mat2, a: Mat2, rad: float) -> Mat2:
    """Rotates a mat2 by the given angle"""
    import math
    
    a0 = a[0]
    a1 = a[1]
    a2 = a[2]
    a3 = a[3]
    
    s = math.sin(rad)
    c = math.cos(rad)
    
    out[0] = a0 * c + a2 * s
    out[1] = a1 * c + a3 * s
    out[2] = a0 * -s + a2 * c
    out[3] = a1 * -s + a3 * c
    
    return out

def scale(out: Mat2, a: Mat2, v: List[float]) -> Mat2:
    """Scales the mat2 by the dimensions in the given vec2"""
    a0 = a[0]
    a1 = a[1]
    a2 = a[2]
    a3 = a[3]
    
    v0 = v[0]
    v1 = v[1]
    
    out[0] = a0 * v0
    out[1] = a1 * v0
    out[2] = a2 * v1
    out[3] = a3 * v1
    
    return out

def from_rotation(out: Mat2, rad: float) -> Mat2:
    """Creates a matrix from a given angle"""
    import math
    
    s = math.sin(rad)
    c = math.cos(rad)
    
    out[0] = c
    out[1] = s
    out[2] = -s
    out[3] = c
    
    return out

def from_scaling(out: Mat2, v: List[float]) -> Mat2:
    """Creates a matrix from a vector scaling"""
    out[0] = v[0]
    out[1] = 0.0
    out[2] = 0.0
    out[3] = v[1]
    
    return out

def from_values(m00: float, m01: float, m10: float, m11: float) -> Mat2:
    """Create a new mat2 with the given values"""
    return [m00, m01, m10, m11]