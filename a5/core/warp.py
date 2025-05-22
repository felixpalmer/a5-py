"""
A5
SPDX-License-Identifier: Apache-2.0
Copyright (c) A5 contributors
"""

import math
from typing import cast
from .coordinate_systems import Radians, Polar
from .constants import distance_to_edge, PI_OVER_5, TWO_PI_OVER_5, WARP_FACTOR

def normalize_gamma(gamma: Radians) -> Radians:
    segment = gamma / TWO_PI_OVER_5
    s_center = round(segment)
    s_offset = segment - s_center

    # Azimuthal angle from triangle bisector
    beta = s_offset * TWO_PI_OVER_5
    return cast(Radians, beta)

def _warp_beta(beta: float) -> float:
    shifted_beta = beta * WARP_FACTOR
    return math.tan(shifted_beta)

def _unwarp_beta(beta: float) -> float:
    shifted_beta = math.atan(beta)
    return shifted_beta / WARP_FACTOR

beta_max = PI_OVER_5
WARP_SCALER = _warp_beta(beta_max) / beta_max

def warp_beta(beta: float) -> float:
    return _warp_beta(beta) / WARP_SCALER

def unwarp_beta(beta: float) -> float:
    return _unwarp_beta(beta * WARP_SCALER)

def warp_rho(rho: float, beta: float) -> float:
    beta_ratio = abs(beta) / beta_max
    shifted_rho = rho * (0.95 - 0.05 * beta_ratio)
    return math.tan(shifted_rho)

def unwarp_rho(rho: float, beta: float) -> float:
    beta_ratio = abs(beta) / beta_max
    shifted_rho = math.atan(rho)
    return shifted_rho / (0.95 - 0.05 * beta_ratio)

def warp_polar(polar: Polar) -> Polar:
    rho, gamma = polar
    beta = normalize_gamma(gamma)
    
    beta2 = warp_beta(normalize_gamma(gamma))
    delta_beta = beta2 - beta

    # Distance to edge will change, so shift rho to match
    scale = math.cos(beta) / math.cos(beta2)
    rho_out = scale * rho

    rho_max = distance_to_edge / math.cos(beta2)
    scaler2 = warp_rho(rho_max, beta2) / rho_max
    rho_warped = warp_rho(rho_out, beta2) / scaler2

    return cast(Polar, (rho_warped, gamma + delta_beta))

def unwarp_polar(polar: Polar) -> Polar:
    rho, gamma = polar
    beta2 = normalize_gamma(gamma)
    beta = unwarp_beta(beta2)
    delta_beta = beta2 - beta

    # Reverse the rho warping
    rho_max = distance_to_edge / math.cos(beta2)
    scaler2 = warp_rho(rho_max, beta2) / rho_max
    rho_unwarped = unwarp_rho(rho * scaler2, beta2)
    
    # Reverse the scale adjustment
    scale = math.cos(beta) / math.cos(beta2)
    return cast(Polar, (rho_unwarped / scale, gamma - delta_beta))