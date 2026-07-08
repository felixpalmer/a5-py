# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

from a5 import grid_disk, grid_disk_vertex, lonlat_to_cell

london = lonlat_to_cell((-0.1276, 51.5072), 9)


def bench_grid_disk_k_1(benchmark):
    benchmark(lambda: grid_disk(london, 1))


def bench_grid_disk_k_5(benchmark):
    benchmark(lambda: grid_disk(london, 5))


def bench_grid_disk_k_20(benchmark):
    benchmark(lambda: grid_disk(london, 20))


def bench_grid_disk_vertex_k_5(benchmark):
    benchmark(lambda: grid_disk_vertex(london, 5))
