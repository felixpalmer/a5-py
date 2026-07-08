# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

import itertools

from a5 import (
    cell_area,
    cell_to_children,
    cell_to_parent,
    get_num_cells,
    get_num_children,
    get_res0_cells,
    get_resolution,
)

from .utils import sample_cells

N = 256
cells15 = sample_cells(15, N)
cells10 = sample_cells(10, N)


def bench_get_resolution_res_15(benchmark):
    counter = itertools.count()
    benchmark(lambda: get_resolution(cells15[next(counter) & (N - 1)]))


def bench_cell_to_parent_res_15_to_14(benchmark):
    counter = itertools.count()
    benchmark(lambda: cell_to_parent(cells15[next(counter) & (N - 1)]))


def bench_cell_to_parent_res_15_to_5(benchmark):
    counter = itertools.count()
    benchmark(lambda: cell_to_parent(cells15[next(counter) & (N - 1)], 5))


def bench_cell_to_children_res_15_to_16(benchmark):
    counter = itertools.count()
    benchmark(lambda: cell_to_children(cells15[next(counter) & (N - 1)]))


def bench_cell_to_children_res_10_to_13(benchmark):
    counter = itertools.count()
    benchmark(lambda: cell_to_children(cells10[next(counter) & (N - 1)], 13))


def bench_get_res0_cells(benchmark):
    benchmark(get_res0_cells)


def bench_get_num_cells_res_15(benchmark):
    benchmark(lambda: get_num_cells(15))


def bench_get_num_children_res_0_to_15(benchmark):
    benchmark(lambda: get_num_children(0, 15))


def bench_cell_area_res_15(benchmark):
    benchmark(lambda: cell_area(15))
