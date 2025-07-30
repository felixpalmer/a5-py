"""
gl-matrix style vec3 operations for A5
Based on https://glmatrix.net/docs/module-vec3.html

All functions follow the gl-matrix convention of having an 'out' parameter
for the result, and return the 'out' parameter for chaining.
"""

import math
from typing import Tuple, Union, List, cast

# Type alias for 3D vectors - can be list or tuple
Vec3 = Union[List[float], Tuple[float, float, float]]

# Pre-allocated temporary vectors for performance (like TypeScript gl-matrix)
midpointAB = [0.0, 0.0, 0.0]
crossCD = [0.0, 0.0, 0.0]
scaledA = [0.0, 0.0, 0.0]
scaledB = [0.0, 0.0, 0.0]

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

# Alias for TypeScript compatibility
sub = subtract

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

def tripleProduct(a: Vec3, b: Vec3, c: Vec3) -> float:
    """
    Computes the triple product of three vectors: a · (b × c)
    
    Args:
        a: first vector
        b: second vector  
        c: third vector
        
    Returns:
        scalar result a · (b × c)
    """
    # Compute cross product b × c using global temp vector
    cross(crossCD, b, c)
    # Return dot product a · (b × c)
    return dot(a, crossCD)

def vectorDifference(A: "Cartesian", B: "Cartesian") -> float:
    """
    Returns a difference measure between two vectors, a - b
    D = sqrt(1 - dot(a,b)) / sqrt(2)
    D = 1: a and b are perpendicular
    D = 0: a and b are the same
    D = NaN: a and b are opposite (shouldn't happen in IVEA as we're using normalized vectors in the same hemisphere)
    
    D is a measure of the angle between the two vectors. sqrt(2) can be ignored when comparing ratios.
    
    Args:
        A: first vector
        B: second vector
        
    Returns:
        difference measure between A and B
    """
    # Original implementation is unstable for small angles as dot(A, B) approaches 1
    # dot(A, B) = cos(x) as A and B are normalized
    # Using double angle formula for cos(2x) = 1 - 2sin(x)^2, can rewrite as:
    # 1 - cos(x) = 2 * sin(x/2)^2)
    #            = 2 * sin(x/2)^2
    # ⇒ sqrt(1 - cos(x)) = sqrt(2) * sin(x/2) 
    # Angle x/2 can be obtained as the angle between A and the normalized midpoint of A and B
    # ⇒ sin(x/2) = |cross(A, midpointAB)|
    lerp(midpointAB, A, B, 0.5)
    normalize(midpointAB, midpointAB)
    cross(midpointAB, A, midpointAB)
    D = length(midpointAB)

    # Math.sin(x) = x for x < 1e-8
    if D < 1e-8:
        # When A and B are close or equal sin(x/2) ≈ x/2, just take the half-distance between A and B
        subtract(crossCD, A, B)
        half_distance = 0.5 * length(crossCD)
        return half_distance
    return D

def quadrupleProduct(out: Vec3, A: "Cartesian", B: "Cartesian", C: "Cartesian", D: "Cartesian") -> Vec3:
    """
    Computes the quadruple product of four vectors
    
    Args:
        out: output vector
        A, B, C, D: input vectors
        
    Returns:
        out
    """
    cross(crossCD, C, D)
    triple_product_acd = dot(A, crossCD)
    triple_product_bcd = dot(B, crossCD)
    scale(scaledA, A, triple_product_bcd)
    scale(scaledB, B, triple_product_acd)
    return subtract(out, scaledB, scaledA)

def slerp(out: Vec3, A: "Cartesian", B: "Cartesian", t: float) -> "Cartesian":
    """
    Spherical linear interpolation between two vectors
    
    Args:
        out: The target vector to write the result to
        A: The first vector
        B: The second vector
        t: The interpolation parameter (0 to 1)
        
    Returns:
        The interpolated vector (same as out)
    """
    gamma = angle(A, B)
    if gamma < 1e-12:
        return lerp(out, A, B, t)
    
    weight_a = math.sin((1 - t) * gamma) / math.sin(gamma)
    weight_b = math.sin(t * gamma) / math.sin(gamma)
    scale(scaledA, A, weight_a)
    scale(scaledB, B, weight_b)
    add(out, scaledA, scaledB)
    return cast("Cartesian", (out[0], out[1], out[2]))