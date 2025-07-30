"""
A5
SPDX-License-Identifier: Apache-2.0
Copyright (c) A5 contributors
"""

import math
from typing import cast, Tuple
from ..core.coordinate_systems import Radians

# Authalic conversion coefficients obtained from: https://arxiv.org/pdf/2212.05818
# See: authalic_constants.py for the derivation of the coefficients
GEODETIC_TO_AUTHALIC = (
    -2.2392098386786394e-03,
    2.1308606513250217e-06,
    -2.5592576864212742e-09,
    3.3701965267802837e-12,
    -4.6675453126112487e-15,
    6.6749287038481596e-18
)

AUTHALIC_TO_GEODETIC = (
    2.2392089963541657e-03,
    2.8831978048607556e-06,
    5.0862207399726603e-09,
    1.0201812377816100e-11,
    2.1912872306767718e-14,
    4.9284235482523806e-17
)

# Adaptation of applyCoefficients from DGGAL project: authalic.ec
#
# BSD 3-Clause License
# 
# Copyright (c) 2014-2025, Ecere Corporation
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

class AuthalicProjection:
    """
    Authalic projection implementation that converts between geodetic and authalic latitudes.
    """
    
    def _apply_coefficients(self, phi: Radians, C: Tuple[float, ...]) -> Radians:
        """
        Applies coefficients using Clenshaw summation algorithm (order 6)
        
        Args:
            phi: Angle in radians
            C: Tuple of coefficients
            
        Returns:
            Transformed angle in radians
        """
        sin_phi = math.sin(phi)
        cos_phi = math.cos(phi)
        X = 2 * (cos_phi - sin_phi) * (cos_phi + sin_phi)
        
        u0 = X * C[5] + C[4]
        u1 = X * u0 + C[3]
        u0 = X * u1 - u0 + C[2]
        u1 = X * u0 - u1 + C[1]
        u0 = X * u1 - u0 + C[0]
        
        return cast(Radians, phi + 2 * sin_phi * cos_phi * u0)

    def forward(self, phi: Radians) -> Radians:
        """
        Convert geodetic latitude to authalic latitude
        
        Args:
            phi: Geodetic latitude in radians
            
        Returns:
            Authalic latitude in radians
        """
        return self._apply_coefficients(phi, GEODETIC_TO_AUTHALIC)

    def inverse(self, phi: Radians) -> Radians:
        """
        Convert authalic latitude to geodetic latitude
        
        Args:
            phi: Authalic latitude in radians
            
        Returns:
            Geodetic latitude in radians
        """
        return self._apply_coefficients(phi, AUTHALIC_TO_GEODETIC)

__all__ = ['AuthalicProjection']