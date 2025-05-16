"""
A5
SPDX-License-Identifier: Apache-2.0
Copyright (c) A5 contributors
"""

import math
from typing import cast
from .types import Polar, Spherical

def project_gnomonic(polar: Polar) -> Spherical:
    """Project polar coordinates to spherical coordinates using gnomonic projection.
    
    Args:
        polar: Tuple of (rho, gamma) where:
            rho: Radial distance from face center
            gamma: Azimuthal angle in radians
    
    Returns:
        Tuple of (theta, phi) in radians
    """
    rho, gamma = polar
    return cast(Spherical, (gamma, math.atan(rho)))

def unproject_gnomonic(spherical: Spherical) -> Polar:
    """Unproject spherical coordinates to polar coordinates using gnomonic projection.
    
    Args:
        spherical: Tuple of (theta, phi) in radians
    
    Returns:
        Tuple of (rho, gamma) where:
            rho: Radial distance from face center
            gamma: Azimuthal angle in radians
    """
    theta, phi = spherical
    return cast(Polar, (math.tan(phi), theta))

__all__ = ['project_gnomonic', 'unproject_gnomonic'] 