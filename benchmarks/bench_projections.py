# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

import itertools
import math

from a5.core.cell import cell_to_spherical
from a5.core.serialization import deserialize
from a5.projections.authalic import AuthalicProjection
from a5.projections.dodecahedron import DodecahedronProjection
from a5.projections.gnomonic import GnomonicProjection

from .utils import create_random, sample_cells

N = 256

# Spherical points paired with the origin of the face they fall on
cells = sample_cells(10, N)
sphericals = [cell_to_spherical(c) for c in cells]
origin_ids = [deserialize(c)['origin'].id for c in cells]

dodecahedron = DodecahedronProjection()
faces = [dodecahedron.forward(sphericals[i], origin_ids[i]) for i in range(N)]

authalic = AuthalicProjection()
gnomonic = GnomonicProjection()
_random = create_random(7)
phis = [math.pi * (_random() - 0.5) for _ in range(N)]
polars = [gnomonic.forward(sphericals[i]) for i in range(N)]


def bench_dodecahedron_forward(benchmark):
    counter = itertools.count()

    def run():
        n = next(counter) & (N - 1)
        return dodecahedron.forward(sphericals[n], origin_ids[n])

    benchmark(run)


def bench_dodecahedron_inverse(benchmark):
    counter = itertools.count()

    def run():
        n = next(counter) & (N - 1)
        return dodecahedron.inverse(faces[n], origin_ids[n])

    benchmark(run)


def bench_authalic_forward(benchmark):
    counter = itertools.count()
    benchmark(lambda: authalic.forward(phis[next(counter) & (N - 1)]))


def bench_authalic_inverse(benchmark):
    counter = itertools.count()
    benchmark(lambda: authalic.inverse(phis[next(counter) & (N - 1)]))


def bench_gnomonic_forward(benchmark):
    counter = itertools.count()
    benchmark(lambda: gnomonic.forward(sphericals[next(counter) & (N - 1)]))


def bench_gnomonic_inverse(benchmark):
    counter = itertools.count()
    benchmark(lambda: gnomonic.inverse(polars[next(counter) & (N - 1)]))
