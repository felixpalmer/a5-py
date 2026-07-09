# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

# Public A5 space-filling curve: point -> s, using the L-system curve
# (a5/lattice/lsystem/). The s <-> cell mappings live in lsystem/__init__.py
# (s_to_cell / s_to_triple) and triple.py (triple_to_s).

from ..core.coordinate_systems import IJ
from .lsystem import sum_point_to_s
from .types import Orientation


def ij_to_s(ij: IJ, resolution: int, orientation: Orientation = 'uv') -> int:
    """
    Fractional IJ point -> curve position `s` of the containing cell, by direct
    L-system descent. The IJ plane maps onto the L-system's corner-sum frame by
    the exact affine map target = (12*(i+j), -12*j).
    """
    return sum_point_to_s(12.0 * (ij[0] + ij[1]), -12.0 * ij[1], resolution, orientation)
