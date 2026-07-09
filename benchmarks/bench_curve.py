# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

# Benchmarks for the space-filling curve: s -> cell decode, cell -> s encode,
# and fractional-point location (ij_to_s).

import itertools
from typing import List

from a5.lattice import ij_to_s, s_to_cell, triple_to_s
from a5.lattice.types import Triple

from .utils import create_random

N = 256


def sample_s(resolution: int, n: int, seed: int = 42) -> List[int]:
    """Deterministic S values in [0, 4**resolution)."""
    random = create_random(seed)
    max_s = 1 << (2 * resolution)
    values: List[int] = []
    for _ in range(n):
        hi = int(random() * 0x100000000)
        lo = int(random() * 0x100000000)
        values.append(((hi << 32) | lo) % max_s)
    return values


def _centroid_ij(t: Triple):
    """The cell's centroid in IJ coordinates
    (parity 0: (x+y+1/3, -x+1/3), parity 1: (x+y-1/3, -x+2/3))."""
    parity = t.x + t.y + t.z
    if parity == 0:
        return (t.x + t.y + 1 / 3, -t.x + 1 / 3)
    return (t.x + t.y - 1 / 3, -t.x + 2 / 3)


def _make_s_to_cell(resolution, orientation):
    values = sample_s(resolution, N)
    counter = itertools.count()

    def run():
        return s_to_cell(values[next(counter) & (N - 1)], resolution, orientation)

    return run


def bench_s_to_cell_res_5_uv(benchmark):
    benchmark(_make_s_to_cell(5, 'uv'))


def bench_s_to_cell_res_15_uv(benchmark):
    benchmark(_make_s_to_cell(15, 'uv'))


def bench_s_to_cell_res_28_uv(benchmark):
    benchmark(_make_s_to_cell(28, 'uv'))


def bench_s_to_cell_res_15_wu(benchmark):
    benchmark(_make_s_to_cell(15, 'wu'))


def _make_triple_to_s(resolution):
    values = sample_s(resolution, N)
    triples = [s_to_cell(s, resolution, 'uv').triple for s in values]
    counter = itertools.count()

    def run():
        return triple_to_s(triples[next(counter) & (N - 1)], resolution, 'uv')

    return run


def bench_triple_to_s_res_5(benchmark):
    benchmark(_make_triple_to_s(5))


def bench_triple_to_s_res_15(benchmark):
    benchmark(_make_triple_to_s(15))


def bench_triple_to_s_res_28(benchmark):
    benchmark(_make_triple_to_s(28))


def bench_ij_to_s_res_15(benchmark):
    values = sample_s(15, N)
    ijs = [_centroid_ij(s_to_cell(s, 15, 'uv').triple) for s in values]
    counter = itertools.count()
    benchmark(lambda: ij_to_s(ijs[next(counter) & (N - 1)], 15, 'uv'))
