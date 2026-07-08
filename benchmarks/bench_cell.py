# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

import itertools

from a5 import cell_to_boundary, cell_to_lonlat, lonlat_to_cell

from .utils import sample_cells, sample_points

N = 256
points = sample_points(N)


def _make_lonlat_to_cell(resolution):
    counter = itertools.count()

    def run():
        return lonlat_to_cell(points[next(counter) & (N - 1)], resolution)

    return run


def bench_lonlat_to_cell_res_5(benchmark):
    benchmark(_make_lonlat_to_cell(5))


def bench_lonlat_to_cell_res_15(benchmark):
    benchmark(_make_lonlat_to_cell(15))


def bench_lonlat_to_cell_res_30(benchmark):
    benchmark(_make_lonlat_to_cell(30))


def _make_cell_to_lonlat(resolution):
    cells = sample_cells(resolution, N)
    counter = itertools.count()

    def run():
        return cell_to_lonlat(cells[next(counter) & (N - 1)])

    return run


def bench_cell_to_lonlat_res_5(benchmark):
    benchmark(_make_cell_to_lonlat(5))


def bench_cell_to_lonlat_res_15(benchmark):
    benchmark(_make_cell_to_lonlat(15))


def bench_cell_to_lonlat_res_30(benchmark):
    benchmark(_make_cell_to_lonlat(30))


def _make_cell_to_boundary(resolution, options=None):
    cells = sample_cells(resolution, N)
    counter = itertools.count()

    def run():
        return cell_to_boundary(cells[next(counter) & (N - 1)], options)

    return run


def bench_cell_to_boundary_res_5(benchmark):
    benchmark(_make_cell_to_boundary(5))


def bench_cell_to_boundary_res_15(benchmark):
    benchmark(_make_cell_to_boundary(15))


def bench_cell_to_boundary_res_30(benchmark):
    benchmark(_make_cell_to_boundary(30))


def bench_cell_to_boundary_res_15_segments_10(benchmark):
    benchmark(_make_cell_to_boundary(15, {'segments': 10}))
