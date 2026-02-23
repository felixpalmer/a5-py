# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

from typing import Tuple
from ..core.coordinate_systems import IJ, KJ


def ij_to_kj(ij: IJ) -> KJ:
    """Convert from IJ coordinates to KJ coordinates."""
    i, j = ij
    return (i + j, j)


def kj_to_ij(kj: KJ) -> IJ:
    """Convert from KJ coordinates to IJ coordinates."""
    k, j = kj
    return (k - j, j)
