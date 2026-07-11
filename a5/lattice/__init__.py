# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

# The canonical A5 curve is currently the ORIGINAL construction (compat.py):
# the two-motif quaternary L-system with the shift_digits recode on top, so
# cell IDs remain bit-identical to previous releases. The non-self-intersecting
# L-system curve (lsystem/ + curve.py) powers the machinery underneath and is
# fully implemented and pinned by fixtures (tests/lattice/test_lsystem.py);
# making it canonical is a planned follow-up -- a breaking change of all cell
# IDs that swaps the exports below to the lsystem versions and regenerates the
# fixtures.

from .types import Orientation, Triple

from .lsystem import Cell

# The engine uses the compat (original) curve, exported under the plain names.
from .compat import (
    compat_s_to_cell as s_to_cell,
    compat_s_to_triple as s_to_triple,
    compat_triple_to_s as triple_to_s,
    compat_ij_to_s as ij_to_s,
)

# Also exported under their own names, so the old-curve behavior stays pinned
# explicitly (tests/lattice/test_compat.py) across the future canonical swap.
from .compat import compat_s_to_cell, compat_s_to_triple, compat_triple_to_s, compat_ij_to_s

from .triple import triple_parity, triple_in_bounds
