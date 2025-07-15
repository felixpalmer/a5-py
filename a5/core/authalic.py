"""
A5
SPDX-License-Identifier: Apache-2.0
Copyright (c) A5 contributors
"""

import warnings
from ..projections.authalic import AuthalicProjection
from .coordinate_systems import Radians

# Create a singleton instance for the old functions to use
_authalic = AuthalicProjection()

def geodetic_to_authalic(phi: Radians) -> Radians:
    """
    Convert geodetic latitude to authalic latitude
    
    Args:
        phi: Geodetic latitude in radians
        
    Returns:
        Authalic latitude in radians
        
    Deprecated:
        Use AuthalicProjection class from projections.authalic instead
    """
    warnings.warn(
        "geodetic_to_authalic() is deprecated. Use AuthalicProjection class from projections.authalic instead",
        DeprecationWarning,
        stacklevel=2
    )
    return _authalic.forward(phi)

def authalic_to_geodetic(phi: Radians) -> Radians:
    """
    Convert authalic latitude to geodetic latitude
    
    Args:
        phi: Authalic latitude in radians
        
    Returns:
        Geodetic latitude in radians
        
    Deprecated:
        Use AuthalicProjection class from projections.authalic instead
    """
    warnings.warn(
        "authalic_to_geodetic() is deprecated. Use AuthalicProjection class from projections.authalic instead",
        DeprecationWarning,
        stacklevel=2
    )
    return _authalic.inverse(phi) 