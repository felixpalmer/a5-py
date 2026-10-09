# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

import itertools

import pytest

import a5

from .utils import sample_cells

N = 64
cells8 = sample_cells(8, N)
cells15 = sample_cells(15, N)

# Absent from baselines that predate them
cell_to_subcell = getattr(a5, 'cell_to_subcell', None)
cell_to_supercell = getattr(a5, 'cell_to_supercell', None)
pytestmark = pytest.mark.skipif(cell_to_subcell is None or cell_to_supercell is None, reason='not in this version of a5')


def bench_cell_to_supercell_res_15_to_8(benchmark):
    counter = itertools.count()
    benchmark(lambda: cell_to_supercell(cells15[next(counter) & (N - 1)], 8))


def bench_cell_to_subcell_res_8_to_11(benchmark):
    counter = itertools.count()
    benchmark(lambda: cell_to_subcell(cells8[next(counter) & (N - 1)], 11))


def bench_cell_to_subcell_res_8_to_16(benchmark):
    counter = itertools.count()
    benchmark(lambda: cell_to_subcell(cells8[next(counter) & (N - 1)], 16))
