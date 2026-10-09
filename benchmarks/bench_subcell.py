# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

import itertools

from a5 import cell_to_subcell, cell_to_supercell

from .utils import sample_cells

N = 64
cells8 = sample_cells(8, N)
cells15 = sample_cells(15, N)


def bench_cell_to_supercell_res_15_to_8(benchmark):
    counter = itertools.count()
    benchmark(lambda: cell_to_supercell(cells15[next(counter) & (N - 1)], 8))


def bench_cell_to_subcell_res_8_to_11(benchmark):
    counter = itertools.count()
    benchmark(lambda: cell_to_subcell(cells8[next(counter) & (N - 1)], 11))


def bench_cell_to_subcell_res_8_to_16(benchmark):
    counter = itertools.count()
    benchmark(lambda: cell_to_subcell(cells8[next(counter) & (N - 1)], 16))
