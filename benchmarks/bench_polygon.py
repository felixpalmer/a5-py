# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

from a5 import polygon_to_cells

from .utils import country_polygon

# Country outlines cover the interesting cases: many vertices (line tracing),
# large interiors (flood fill), multi-ring coastlines and high latitudes.
# All names verified present in tests/regions/fixtures/polygon.json.
CASES = [
    ('United Kingdom', 7),
    ('France', 7),
    ('Brazil', 6),
    ('United States of America', 5),
    ('Fiji', 8),  # antimeridian
]

_polygons = {name: country_polygon(name) for name, _ in CASES}


def bench_polygon_united_kingdom_res_7(benchmark):
    poly = _polygons['United Kingdom']
    benchmark(lambda: polygon_to_cells(poly, 7))


def bench_polygon_france_res_7(benchmark):
    poly = _polygons['France']
    benchmark(lambda: polygon_to_cells(poly, 7))


def bench_polygon_brazil_res_6(benchmark):
    poly = _polygons['Brazil']
    benchmark(lambda: polygon_to_cells(poly, 6))


def bench_polygon_united_states_res_5(benchmark):
    poly = _polygons['United States of America']
    benchmark(lambda: polygon_to_cells(poly, 5))


def bench_polygon_fiji_res_8(benchmark):
    poly = _polygons['Fiji']
    benchmark(lambda: polygon_to_cells(poly, 8))
