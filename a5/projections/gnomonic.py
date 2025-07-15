"""
A5
SPDX-License-Identifier: Apache-2.0
Copyright (c) A5 contributors
"""

import math
from typing import cast
from ..core.coordinate_systems import Polar, Spherical

class GnomonicProjection:
    """
    Gnomonic projection implementation that converts between spherical and polar coordinates.
    """
    
    def forward(self, spherical: Spherical) -> Polar:
        """
        Projects spherical coordinates to polar coordinates using gnomonic projection
        
        Args:
            spherical: Spherical coordinates [theta, phi]
            
        Returns:
            Polar coordinates [rho, gamma]
        """
        theta, phi = spherical
        return cast(Polar, (math.tan(phi), theta))

    def inverse(self, polar: Polar) -> Spherical:
        """
        Unprojects polar coordinates to spherical coordinates using gnomonic projection
        
        Args:
            polar: Polar coordinates [rho, gamma]
            
        Returns:
            Spherical coordinates [theta, phi]
        """
        rho, gamma = polar
        return cast(Spherical, (gamma, math.atan(rho)))

__all__ = ['GnomonicProjection'] 