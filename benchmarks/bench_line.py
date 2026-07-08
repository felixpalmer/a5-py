# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

from a5 import line_string_to_cells

london_paris = [
    (-0.1276, 51.5072),
    (2.3522, 48.8566),
]

round_the_world = [
    (-122.4194, 37.7749),  # San Francisco
    (-74.006, 40.7128),    # New York
    (-0.1276, 51.5072),    # London
    (139.6917, 35.6895),   # Tokyo
    (151.2093, -33.8688),  # Sydney
]


def bench_line_string_london_paris_res_9(benchmark):
    benchmark(lambda: line_string_to_cells(london_paris, 9))


def bench_line_string_round_the_world_res_6(benchmark):
    benchmark(lambda: line_string_to_cells(round_the_world, 6))
