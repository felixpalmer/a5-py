# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

import math
from typing import List, cast

from ..core.coordinate_systems import Cartesian, Spherical
from ..core.coordinate_transforms import to_cartesian
from ..math import quat, vec3

# Number of perturbed sample points the spiral can produce. Tuned on a
# corpus of ~3500 spherical points x 8 resolutions, such that the spiral
# hits a strictly-containing cell within these many iterations for all
# but a handful of points right at the polar singularity at very high
# resolutions.
SPIRAL_SAMPLE_COUNT = 24

# Azimuthal step between consecutive samples in the rotated tangent plane.
# 1.4 rad (~80 deg) sits on a flat plateau of the parameter sweep.
_ANGLE_STEP_RAD = 1.4

# Precomputed unit-direction spiral at the canonical pole's tangent plane
# (z=0). Each entry is the tangent direction of one sample. The pattern
# is independent of resolution; per spiral the directions are rotated to
# the input point's tangent plane via a single quaternion.
_POLE: List[float] = [0.0, 0.0, 1.0]
_SPIRAL_DIRECTIONS: List[List[float]] = [
    [math.cos((i + 1) * _ANGLE_STEP_RAD), math.sin((i + 1) * _ANGLE_STEP_RAD), 0.0]
    for i in range(SPIRAL_SAMPLE_COUNT)
]


class Spiral:
    """
    Lazy spiral sampler around a center point on the unit sphere -- used by
    `spherical_to_cell` to discover nearby cells when the projection-based
    estimate lands in the wrong one.

    Construction precomputes the pole->center quaternion. `sample(out, i)`
    rotates the i-th cached direction into the tangent plane at `center`,
    scales by the appropriate radius, and returns a Cartesian point near
    the unit sphere -- the consumer of the spiral (the dodecahedron
    projection) wants Cartesian anyway, so we skip the spherical
    round-trip entirely. The point is slightly off the unit sphere by
    O(R^2); downstream callers either tolerate this or normalise.

    All state is per-instance (no module-level mutable scratch), so each
    thread/task can hold its own without locking.
    """

    def __init__(self, center: Spherical, scale_rad: float):
        """
        Initialise a spiral around `center` on the unit sphere. The
        tangent-plane radius of the outermost sample is `scale_rad`;
        intermediate samples scale linearly between 0 and that.
        `quat.rotation_to` handles the antipode case internally.
        """
        c0 = to_cartesian(center)
        self.c0: List[float] = [c0[0], c0[1], c0[2]]
        self.q = quat.create()
        quat.rotation_to(self.q, _POLE, self.c0)
        self.scale_rad = scale_rad
        self.scratch: List[float] = vec3.create()

    def sample(self, out: List[float], i: int) -> Cartesian:
        """
        Write the i-th spiral sample (0 <= i < SPIRAL_SAMPLE_COUNT) into
        `out` and return it. Sample i sits at tangent-plane offset of
        magnitude `(i+1)/(SPIRAL_SAMPLE_COUNT+1) * scale_rad` from `center`,
        rotated by azimuth `(i+1) * 1.4 rad` in `center`'s tangent frame.

        `out` is supplied by the caller so the same buffer can be reused
        across all samples in a search, avoiding per-iteration allocation.
        """
        vec3.transformQuat(self.scratch, _SPIRAL_DIRECTIONS[i], self.q)
        R = ((i + 1) / (SPIRAL_SAMPLE_COUNT + 1)) * self.scale_rad
        out[0] = self.c0[0] + self.scratch[0] * R
        out[1] = self.c0[1] + self.scratch[1] * R
        out[2] = self.c0[2] + self.scratch[2] * R
        return cast(Cartesian, (out[0], out[1], out[2]))
