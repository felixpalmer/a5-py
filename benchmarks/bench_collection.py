# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

from a5 import contains, count, difference, intersect, lonlat_to_cell, polygon_to_cells, spherical_cap, union

from .utils import country_polygon, create_random

# Compacted collections at resolution 12: two neighboring countries and a cap overlapping both
france = polygon_to_cells(country_polygon('France'), 12)
uk = polygon_to_cells(country_polygon('United Kingdom'), 12)
cap_paris = spherical_cap(lonlat_to_cell((2.3522, 48.8566), 12), 400_000)

# Point-in-polygon probes: cells, at the coverage's resolution, of random points in France's bounding box
_random = create_random(7)
probes = [lonlat_to_cell((-5 + 13 * _random(), 42 + 9 * _random()), 12) for _ in range(1000)]


def bench_union_france_uk_res_12(benchmark):
    benchmark(lambda: union(france, uk))


def bench_intersect_france_paris_cap_res_12(benchmark):
    benchmark(lambda: intersect(france, cap_paris))


def bench_difference_france_paris_cap_res_12(benchmark):
    benchmark(lambda: difference(france, cap_paris))


def bench_count_france_res_12(benchmark):
    benchmark(lambda: count(france))


def bench_contains_france_res_12_1000_points(benchmark):
    def run():
        for probe in probes:
            contains(france, probe)

    benchmark(run)
