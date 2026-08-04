# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

# Public A5 space-filling curve: point -> s, using the L-system curve
# (a5/lattice/lsystem/). The s <-> cell mappings live in lsystem/__init__.py
# (s_to_cell / s_to_triple) and triple.py (triple_to_s).

import math

from ..core.coordinate_systems import IJ
from .types import Triple


def round_to_triple(ij: IJ, resolution: int) -> Triple:
    """
    Locate the lattice triangle containing a fractional IJ point, as a triple.

    The triples tile the IJ plane as triangles: the unit square (m, n) =
    (floor(i), floor(j)) splits along the diagonal u+v = 1 into a lower triangle
    (the parity-0 cell (-n, m+n, -m), centroid (m+1/3, n+1/3)) and an upper
    triangle (the parity-1 cell (-n, m+n+1, -m), centroid (m+2/3, n+2/3)) -- the
    centroid correspondences follow from the exact IJ <-> corner-sum affine map
    were derived from the exact IJ <-> corner-sum affine map
    target = (12*(i+j), -12*j) and validated against the old-engine
    discretization. Point location is two floors + one diagonal comparison. Points exactly on a triangle edge have no unique cell; the >=
    tie-break below is the fixed convention.

    The result is clamped into quintant bounds (m >= 0, n >= 0, m+n+parity <=
    max_row, equivalent to triple_in_bounds): a point slightly outside the
    quintant (as the estimate path can produce near quintant edges) must still
    map to a valid cell for the exact encode.
    """
    max_row = (1 << resolution) - 1
    floor_i = math.floor(ij[0])
    floor_j = math.floor(ij[1])
    m = floor_i
    n = floor_j
    parity = 1 if (ij[0] - floor_i) + (ij[1] - floor_j) >= 1.0 else 0
    if m < 0:
        m = 0
    if n < 0:
        n = 0
    if m + n + parity > max_row:
        parity = 0
        if m + n > max_row:
            over = m + n - max_row
            dm = min(m, over)
            m -= dm
            n -= over - dm
    return Triple(-n, m + n + parity, -m)
