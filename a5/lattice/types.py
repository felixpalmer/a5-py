# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

from typing import Literal, NamedTuple

# Orientation of the space-filling curve. The curve fills a space defined by the triangle with
# vertices u, v & w. The orientation describes which corner the curve starts and ends at, e.g. wv
# is a curve that starts at w and ends at v.
Orientation = Literal['uv', 'vu', 'uw', 'wu', 'vw', 'wv']


class Triple(NamedTuple):
    """
    Triple coordinates for the triangular grid underlying the pentagonal A5 grid.

    Neighbors differ by +/-1 in exactly one coordinate while the other two stay constant.
    Triple coordinates are orientation-independent -- the same geometric cell always has
    the same triple coords regardless of curve orientation.
    """
    x: int
    y: int
    z: int
