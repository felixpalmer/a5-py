# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

import math
from typing import Optional
from ..core.coordinate_systems import Cartesian
from ..math import vec3, vector_utils

# Pre-allocated temporary vectors for performance
_temp_a = vec3.create()
_temp_b = vec3.create()
_temp_c = vec3.create()
_temp_midpoint = vec3.create()
_temp_cross = vec3.create()
_temp_scaled_a = vec3.create()
_temp_scaled_b = vec3.create()
_temp_out = vec3.create()

def cross_product(A: Cartesian, B: Cartesian) -> Cartesian:
    """
    Computes the cross product of two vectors using gl-matrix style
    
    Args:
        A: The first vector
        B: The second vector
    Returns:
        The cross product A × B as a vector
    """
    vec3.copy(_temp_a, A)
    vec3.copy(_temp_b, B)
    vec3.cross(_temp_out, _temp_a, _temp_b)
    return (_temp_out[0], _temp_out[1], _temp_out[2])

def dot_product(A: Cartesian, B: Cartesian) -> float:
    """
    Computes the dot product of two vectors using gl-matrix style
    
    Args:
        A: The first vector
        B: The second vector
    Returns:
        The scalar result A · B
    """
    vec3.copy(_temp_a, A)
    vec3.copy(_temp_b, B)
    return vec3.dot(_temp_a, _temp_b)

def vector_magnitude(A: Cartesian) -> float:
    """
    Computes the magnitude (length) of a vector using gl-matrix style
    
    Args:
        A: The vector
    Returns:
        The magnitude ||A||
    """
    vec3.copy(_temp_a, A)
    return vec3.length(_temp_a)

def triple_product(A: Cartesian, B: Cartesian, C: Cartesian) -> float:
    """
    Computes the triple product of three vectors using gl-matrix style
    
    Args:
        A: The first vector
        B: The second vector
        C: The third vector
    Returns:
        The scalar result A · (B × C)
    """
    vec3.copy(_temp_a, A)
    vec3.copy(_temp_b, B)
    vec3.copy(_temp_c, C)
    
    # First compute cross product: B × C
    vec3.cross(_temp_cross, _temp_b, _temp_c)
    
    # Then compute dot product: A · (B × C)
    return vec3.dot(_temp_a, _temp_cross)

def vector_difference(A: Cartesian, B: Cartesian) -> float:
    """
    Returns a difference measure between two vectors using gl-matrix style
    D = sqrt(1 - dot(a,b)) / sqrt(2)
    
    Args:
        A: The first vector
        B: The second vector
    Returns:
        The difference between the two vectors
    """
    vec3.copy(_temp_a, A)
    vec3.copy(_temp_b, B)
    
    # Compute midpoint of A and B
    vec3.lerp(_temp_midpoint, _temp_a, _temp_b, 0.5)
    
    # Normalize the midpoint
    vec3.normalize(_temp_midpoint, _temp_midpoint)
    
    # Compute cross product: A × normalized_midpoint
    vec3.cross(_temp_cross, _temp_a, _temp_midpoint)
    
    # Compute magnitude of cross product
    D = vec3.length(_temp_cross)
    
    # Handle case when A and B are very close
    if D < 1e-8:
        # When A and B are close or equal sin(x/2) ≈ x/2, just take the half-distance between A and B
        vec3.subtract(_temp_out, _temp_a, _temp_b)
        half_distance = 0.5 * vec3.length(_temp_out)
        return half_distance
    
    return D

def quadruple_product(A: Cartesian, B: Cartesian, C: Cartesian, D: Cartesian) -> Cartesian:
    """
    Computes the quadruple product of four vectors using gl-matrix style
    
    Args:
        A: The first vector
        B: The second vector
        C: The third vector
        D: The fourth vector
    Returns:
        The result vector
    """
    vec3.copy(_temp_a, A)
    vec3.copy(_temp_b, B)
    vec3.copy(_temp_c, C)
    
    # Compute cross product C × D
    vec3.cross(_temp_cross, _temp_c, D)
    
    # Compute triple products
    triple_product_acd = vec3.dot(_temp_a, _temp_cross)
    triple_product_bcd = vec3.dot(_temp_b, _temp_cross)
    
    # Scale vectors
    vec3.scale(_temp_scaled_a, _temp_a, triple_product_bcd)
    vec3.scale(_temp_scaled_b, _temp_b, triple_product_acd)
    
    # Compute scaled_b - scaled_a
    vec3.subtract(_temp_out, _temp_scaled_b, _temp_scaled_a)
    
    return (_temp_out[0], _temp_out[1], _temp_out[2])

def slerp(A: Cartesian, B: Cartesian, t: float) -> Cartesian:
    """
    Spherical linear interpolation between two vectors using gl-matrix style
    
    Args:
        A: The first vector
        B: The second vector
        t: The interpolation parameter (0 to 1)
    Returns:
        The interpolated vector
    """
    vec3.copy(_temp_a, A)
    vec3.copy(_temp_b, B)
    
    # Calculate angle between vectors
    gamma = vec3.angle(_temp_a, _temp_b)
    
    if gamma < 1e-12:
        # Vectors are very close, use linear interpolation
        vec3.lerp(_temp_out, _temp_a, _temp_b, t)
        return (_temp_out[0], _temp_out[1], _temp_out[2])
    
    sin_gamma = math.sin(gamma)
    weight_a = math.sin((1 - t) * gamma) / sin_gamma
    weight_b = math.sin(t * gamma) / sin_gamma
    
    # Compute weighted sum
    vec3.scale(_temp_scaled_a, _temp_a, weight_a)
    vec3.scale(_temp_scaled_b, _temp_b, weight_b)
    vec3.add(_temp_out, _temp_scaled_a, _temp_scaled_b)
    
    return (_temp_out[0], _temp_out[1], _temp_out[2]) 