# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

import math
from typing import Optional
from ..core.coordinate_systems import Cartesian

def cross_product(A: Cartesian, B: Cartesian) -> Cartesian:
    """
    Computes the cross product of two vectors without using numpy
    
    Args:
        A: The first vector
        B: The second vector
    Returns:
        The cross product A × B as a vector
    """
    return (
        A[1] * B[2] - A[2] * B[1],  # x component
        A[2] * B[0] - A[0] * B[2],  # y component  
        A[0] * B[1] - A[1] * B[0]   # z component
    )

def dot_product(A: Cartesian, B: Cartesian) -> float:
    """
    Computes the dot product of two vectors without using numpy
    
    Args:
        A: The first vector
        B: The second vector
    Returns:
        The scalar result A · B
    """
    return A[0] * B[0] + A[1] * B[1] + A[2] * B[2]

def vector_magnitude(A: Cartesian) -> float:
    """
    Computes the magnitude (length) of a vector without using numpy
    
    Args:
        A: The vector
    Returns:
        The magnitude ||A||
    """
    return math.sqrt(A[0] * A[0] + A[1] * A[1] + A[2] * A[2])

def triple_product(A: Cartesian, B: Cartesian, C: Cartesian) -> float:
    """
    Computes the triple product of three vectors without using numpy operations
    
    Args:
        A: The first vector
        B: The second vector
        C: The third vector
    Returns:
        The scalar result A · (B × C)
    """
    # First compute cross product: B × C
    cross_bc = cross_product(B, C)
    
    # Then compute dot product: A · (B × C)
    return dot_product(A, cross_bc)

def vector_difference(A: Cartesian, B: Cartesian) -> float:
    """
    Returns a difference measure between two vectors without using numpy operations
    D = sqrt(1 - dot(a,b)) / sqrt(2)
    
    Args:
        A: The first vector
        B: The second vector
    Returns:
        The difference between the two vectors
    """
    # Compute midpoint of A and B
    midpoint = (
        (A[0] + B[0]) * 0.5,
        (A[1] + B[1]) * 0.5,
        (A[2] + B[2]) * 0.5
    )
    
    # Normalize the midpoint
    midpoint_norm = vector_magnitude(midpoint)
    normalized_midpoint = (
        midpoint[0] / midpoint_norm,
        midpoint[1] / midpoint_norm,
        midpoint[2] / midpoint_norm
    )
    
    # Compute cross product: A × normalized_midpoint
    cross_result = cross_product(A, normalized_midpoint)
    
    # Compute magnitude of cross product
    D = vector_magnitude(cross_result)
    
    # Handle case when A and B are very close
    if D < 1e-8:
        # When A and B are close or equal sin(x/2) ≈ x/2, just take the half-distance between A and B
        diff = (A[0] - B[0], A[1] - B[1], A[2] - B[2])
        half_distance = 0.5 * vector_magnitude(diff)
        return half_distance
    
    return D

def quadruple_product(A: Cartesian, B: Cartesian, C: Cartesian, D: Cartesian) -> Cartesian:
    """
    Computes the quadruple product of four vectors without using numpy operations
    
    Args:
        A: The first vector
        B: The second vector
        C: The third vector
        D: The fourth vector
    Returns:
        The result vector
    """
    cross_cd = cross_product(C, D)
    triple_product_acd = dot_product(A, cross_cd)
    triple_product_bcd = dot_product(B, cross_cd)
    
    # scaled_a = A * triple_product_bcd
    scaled_a = (
        A[0] * triple_product_bcd,
        A[1] * triple_product_bcd,
        A[2] * triple_product_bcd
    )
    
    # scaled_b = B * triple_product_acd  
    scaled_b = (
        B[0] * triple_product_acd,
        B[1] * triple_product_acd,
        B[2] * triple_product_acd
    )
    
    # return scaled_b - scaled_a
    return (
        scaled_b[0] - scaled_a[0],
        scaled_b[1] - scaled_a[1],
        scaled_b[2] - scaled_a[2]
    )

def slerp(A: Cartesian, B: Cartesian, t: float) -> Cartesian:
    """
    Spherical linear interpolation between two vectors without using numpy
    
    Args:
        A: The first vector
        B: The second vector
        t: The interpolation parameter (0 to 1)
    Returns:
        The interpolated vector
    """
    # Calculate angle between vectors
    cos_gamma = dot_product(A, B)
    
    # Clamp cos_gamma to valid range [-1.0, 1.0] to avoid domain errors
    cos_gamma = max(-1.0, min(1.0, cos_gamma))
    
    gamma = math.acos(cos_gamma)
    
    if gamma < 1e-12:
        # Vectors are very close, use linear interpolation
        # return A * (1 - t) + B * t
        return (
            A[0] * (1 - t) + B[0] * t,
            A[1] * (1 - t) + B[1] * t,
            A[2] * (1 - t) + B[2] * t
        )
    
    sin_gamma = math.sin(gamma)
    weight_a = math.sin((1 - t) * gamma) / sin_gamma
    weight_b = math.sin(t * gamma) / sin_gamma
    
    # return weight_a * A + weight_b * B
    return (
        weight_a * A[0] + weight_b * B[0],
        weight_a * A[1] + weight_b * B[1],
        weight_a * A[2] + weight_b * B[2]
    ) 