"""
A5
SPDX-License-Identifier: Apache-2.0
Copyright (c) A5 contributors
"""

import numpy as np
from typing import cast
from ..core.coordinate_systems import Radians

class AuthalicProjection:
    """
    Authalic projection implementation that converts between geodetic and authalic latitudes.
    """
    
    # Authalic conversion coefficients obtained from: https://arxiv.org/pdf/2212.05818
    # See: authalic_constants.py for the derivation of the coefficients
    GEODETIC_TO_AUTHALIC = np.array([
        -2.2392098386786394e-03,
        2.1308606513250217e-06,
        -2.5592576864212742e-09,
        3.3701965267802837e-12,
        -4.6675453126112487e-15,
        6.6749287038481596e-18
    ], dtype=np.float64)

    AUTHALIC_TO_GEODETIC = np.array([
        2.2392089963541657e-03,
        2.8831978048607556e-06,
        5.0862207399726603e-09,
        1.0201812377816100e-11,
        2.1912872306767718e-14,
        4.9284235482523806e-17
    ], dtype=np.float64)

    def _apply_coefficients(self, phi: Radians, C: np.ndarray) -> Radians:
        """
        Applies coefficients using Clenshaw summation algorithm (order 6)
        
        Args:
            phi: Angle in radians
            C: Array of coefficients
            
        Returns:
            Transformed angle in radians
        """
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)
        X = 2 * (cos_phi - sin_phi) * (cos_phi + sin_phi)
        
        u0 = X * C[5] + C[4]
        u1 = X * u0 + C[3]
        u0 = X * u1 - u0 + C[2]
        u1 = X * u0 - u1 + C[1]
        u0 = X * u1 - u0 + C[0]
        
        return phi + 2 * sin_phi * cos_phi * u0

    def forward(self, phi: Radians) -> Radians:
        """
        Convert geodetic latitude to authalic latitude
        
        Args:
            phi: Geodetic latitude in radians
            
        Returns:
            Authalic latitude in radians
        """
        return cast(Radians, self._apply_coefficients(phi, self.GEODETIC_TO_AUTHALIC))

    def inverse(self, phi: Radians) -> Radians:
        """
        Convert authalic latitude to geodetic latitude
        
        Args:
            phi: Authalic latitude in radians
            
        Returns:
            Geodetic latitude in radians
        """
        return cast(Radians, self._apply_coefficients(phi, self.AUTHALIC_TO_GEODETIC))

__all__ = ['AuthalicProjection'] 