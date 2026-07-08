# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

import itertools

from a5 import hex_to_u64, u64_to_hex

from .utils import sample_cells

N = 256
cells = sample_cells(20, N)
hexes = [u64_to_hex(c) for c in cells]


def bench_u64_to_hex(benchmark):
    counter = itertools.count()
    benchmark(lambda: u64_to_hex(cells[next(counter) & (N - 1)]))


def bench_hex_to_u64(benchmark):
    counter = itertools.count()
    benchmark(lambda: hex_to_u64(hexes[next(counter) & (N - 1)]))
