# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

import math
from typing import List, cast

from ..core.coordinate_systems import Cartesian
from ..core.constants import AUTHALIC_RADIUS_EARTH
from ..math import vec3
from ..math.vec3 import precompute_slerp, slerp, angle


def great_circle_distance(a: Cartesian, b: Cartesian) -> float:
    """
    Great-circle distance in meters between two unit vectors on the authalic sphere.
    Uses 2*atan2(|a-b|, |a+b|) rather than acos(a.b): the latter returns 0 for
    any points closer than ~1e-8 rad (~6 cm) and loses half its digits near the
    antipode, while this form is accurate over the whole range.
    """
    return angle(a, b) * AUTHALIC_RADIUS_EARTH


def sample_great_circle_arc(a: Cartesian, b: Cartesian, sample_interval: float) -> List[Cartesian]:
    """
    Sample interior points along the great-circle arc from `a` to `b` at roughly
    `sample_interval` meters spacing. Endpoints are NOT included -- the caller
    already has them. Returned vectors live on the authalic unit sphere.
    """
    dist = great_circle_distance(a, b)
    num_segments = max(1, math.ceil(dist / sample_interval))
    samples: List[Cartesian] = []
    if num_segments <= 1:
        return samples
    slerp_ctx = precompute_slerp(a, b)
    for j in range(1, num_segments):
        v = vec3.create()
        slerp(v, a, b, j / num_segments, slerp_ctx)
        samples.append(cast(Cartesian, (v[0], v[1], v[2])))
    return samples
