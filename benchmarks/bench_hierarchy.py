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

from .utils import BATCH, sample_cells

N = 256
cells15 = sample_cells(15, BATCH)
cells10 = sample_cells(10, N)


def bench_get_resolution_res_15_x100(benchmark):
    def run():
        for i in range(BATCH):
            get_resolution(cells15[i])

    benchmark(run)


def bench_cell_to_parent_res_15_to_14_x100(benchmark):
    def run():
        for i in range(BATCH):
            cell_to_parent(cells15[i])

    benchmark(run)


def bench_cell_to_parent_res_15_to_5_x100(benchmark):
    def run():
        for i in range(BATCH):
            cell_to_parent(cells15[i], 5)

    benchmark(run)


def bench_cell_to_children_res_15_to_16_x100(benchmark):
    def run():
        for i in range(BATCH):
            cell_to_children(cells15[i])

    benchmark(run)


def bench_cell_to_children_res_10_to_13(benchmark):
    counter = itertools.count()
    benchmark(lambda: cell_to_children(cells10[next(counter) & (N - 1)], 13))


def bench_get_res0_cells_x100(benchmark):
    def run():
        for _ in range(BATCH):
            get_res0_cells()

    benchmark(run)


def bench_get_num_cells_res_15_x100(benchmark):
    def run():
        for _ in range(BATCH):
            get_num_cells(15)

    benchmark(run)


def bench_get_num_children_res_0_to_15_x100(benchmark):
    def run():
        for _ in range(BATCH):
            get_num_children(0, 15)

    benchmark(run)


def bench_cell_area_res_15_x100(benchmark):
    def run():
        for _ in range(BATCH):
            cell_area(15)

    benchmark(run)
