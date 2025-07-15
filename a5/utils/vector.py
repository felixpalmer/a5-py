# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

import numpy as np
from typing import Optional
from ..core.coordinate_systems import Cartesian

def vector_difference(A: Cartesian, B: Cartesian) -> float:
    """
    Returns a difference measure between two vectors, a - b
    D = sqrt(1 - dot(a,b)) / sqrt(2)
    D = 1: a and b are perpendicular
    D = 0: a and b are the same
    D = NaN: a and b are opposite (shouldn't happen in IVEA as we're using normalized vectors in the same hemisphere)
    
    D is a measure of the angle between the two vectors. sqrt(2) can be ignored when comparing ratios.
    
    Args:
        A: The first vector
        B: The second vector
    Returns:
        The difference between the two vectors
    """
    # Original implementation is unstable for small angles as dot(A, B) approaches 1
    # return np.sqrt(1 - np.dot(A, B))

    # dot(A, B) = cos(x) as A and B are normalized
    # Using double angle formula for cos(2x) = 1 - 2sin(x)^2, can rewrite as:
    # 1 - cos(x) = 2 * sin(x/2)^2)
    #            = 2 * sin(x/2)^2
    # ⇒ sqrt(1 - cos(x)) = sqrt(2) * sin(x/2) 
    # Angle x/2 can be obtained as the angle between A and the normalized midpoint of A and B
    # ⇒ sin(x/2) = |cross(A, midpointAB)|
    midpoint_ab = (A + B) * 0.5
    midpoint_ab = midpoint_ab / np.linalg.norm(midpoint_ab)
    D = np.linalg.norm(np.cross(A, midpoint_ab))

    # Math.sin(x) = x for x < 1e-8
    if D < 1e-8:
        # When A and B are close or equal sin(x/2) ≈ x/2, just take the half-distance between A and B
        AB = A - B
        half_distance = 0.5 * np.linalg.norm(AB)
        return half_distance
    return D

def triple_product(A: Cartesian, B: Cartesian, C: Cartesian) -> float:
    """
    Computes the triple product of three vectors
    
    Args:
        A: The first vector
        B: The second vector
        C: The third vector
    Returns:
        The scalar result
    """
    return np.dot(A, np.cross(B, C))

def quadruple_product(A: Cartesian, B: Cartesian, C: Cartesian, D: Cartesian) -> Cartesian:
    """
    Computes the quadruple product of four vectors
    
    Args:
        A: The first vector
        B: The second vector
        C: The third vector
        D: The fourth vector
    Returns:
        The result vector
    """
    cross_cd = np.cross(C, D)
    triple_product_acd = np.dot(A, cross_cd)
    triple_product_bcd = np.dot(B, cross_cd)
    scaled_a = A * triple_product_bcd
    scaled_b = B * triple_product_acd
    return scaled_b - scaled_a

def slerp(A: Cartesian, B: Cartesian, t: float) -> Cartesian:
    """
    Spherical linear interpolation between two vectors
    
    Args:
        A: The first vector
        B: The second vector
        t: The interpolation parameter (0 to 1)
    Returns:
        The interpolated vector
    """
    # Calculate angle between vectors
    cos_gamma = np.clip(np.dot(A, B), -1.0, 1.0)  # Ensure value is in valid range
    gamma = np.arccos(cos_gamma)
    
    if gamma < 1e-12:
        # Vectors are very close, use linear interpolation
        return A * (1 - t) + B * t
    
    weight_a = np.sin((1 - t) * gamma) / np.sin(gamma)
    weight_b = np.sin(t * gamma) / np.sin(gamma)
    return weight_a * A + weight_b * B 