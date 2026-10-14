# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

import math

from a5.core.cell import cell_to_spherical
from a5.core.serialization import deserialize
from a5.projections.authalic import AuthalicProjection
from a5.projections.dodecahedron import DodecahedronProjection
from a5.projections.gnomonic import GnomonicProjection

from .utils import BATCH, create_random, sample_cells

# Spherical points paired with the origin of the face they fall on
cells = sample_cells(10, BATCH)
sphericals = [cell_to_spherical(c) for c in cells]
origin_ids = [deserialize(c)['origin'].id for c in cells]

dodecahedron = DodecahedronProjection()
faces = [dodecahedron.forward(sphericals[i], origin_ids[i]) for i in range(BATCH)]

authalic = AuthalicProjection()
gnomonic = GnomonicProjection()
_random = create_random(7)
phis = [math.pi * (_random() - 0.5) for _ in range(BATCH)]
polars = [gnomonic.forward(sphericals[i]) for i in range(BATCH)]


def bench_dodecahedron_forward_x100(benchmark):
    def run():
        for i in range(BATCH):
            dodecahedron.forward(sphericals[i], origin_ids[i])

    benchmark(run)


def bench_dodecahedron_inverse_x100(benchmark):
    def run():
        for i in range(BATCH):
            dodecahedron.inverse(faces[i], origin_ids[i])

    benchmark(run)


def bench_authalic_forward_x100(benchmark):
    def run():
        for i in range(BATCH):
            authalic.forward(phis[i])

    benchmark(run)


def bench_authalic_inverse_x100(benchmark):
    def run():
        for i in range(BATCH):
            authalic.inverse(phis[i])

    benchmark(run)


def bench_gnomonic_forward_x100(benchmark):
    def run():
        for i in range(BATCH):
            gnomonic.forward(sphericals[i])

    benchmark(run)


def bench_gnomonic_inverse_x100(benchmark):
    def run():
        for i in range(BATCH):
            gnomonic.inverse(polars[i])

    benchmark(run)
