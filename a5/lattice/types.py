# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

from typing import Tuple, List, Literal

Orientation = Literal['uv', 'vu', 'uw', 'wu', 'vw', 'wv']

Quaternary = Literal[0, 1, 2, 3]
YES: Literal[-1] = -1
NO: Literal[1] = 1
Flip = Literal[-1, 1]


class Anchor:
    def __init__(self, q: Quaternary, offset: Tuple[float, float], flips: Tuple[Flip, Flip]):
        self.q = q
        self.offset = offset
        self.flips = flips
