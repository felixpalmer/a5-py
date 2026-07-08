# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

from a5 import lonlat_to_cell, spherical_cap

london9 = lonlat_to_cell((-0.1276, 51.5072), 9)
london12 = lonlat_to_cell((-0.1276, 51.5072), 12)


def bench_spherical_cap_res_9_radius_10km(benchmark):
    benchmark(lambda: spherical_cap(london9, 10_000))


def bench_spherical_cap_res_9_radius_100km(benchmark):
    benchmark(lambda: spherical_cap(london9, 100_000))


def bench_spherical_cap_res_12_radius_5km(benchmark):
    benchmark(lambda: spherical_cap(london12, 5_000))
