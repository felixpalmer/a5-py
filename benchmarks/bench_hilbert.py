# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

# Benchmarks for the space-filling curve (pre-L-system engine):
# s -> anchor decode, anchor -> s encode, and fractional-point location (ij_to_s).

import itertools
from typing import List

from a5.lattice import anchor_to_s, ij_to_s, s_to_anchor

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


def _make_s_to_anchor(resolution, orientation):
    values = sample_s(resolution, N)
    counter = itertools.count()

    def run():
        return s_to_anchor(values[next(counter) & (N - 1)], resolution, orientation)

    return run


def bench_s_to_anchor_res_5_uv(benchmark):
    benchmark(_make_s_to_anchor(5, 'uv'))


def bench_s_to_anchor_res_15_uv(benchmark):
    benchmark(_make_s_to_anchor(15, 'uv'))


def bench_s_to_anchor_res_28_uv(benchmark):
    benchmark(_make_s_to_anchor(28, 'uv'))


def bench_s_to_anchor_res_15_wu(benchmark):
    benchmark(_make_s_to_anchor(15, 'wu'))


def _make_anchor_to_s(resolution):
    values = sample_s(resolution, N)
    anchors = [s_to_anchor(s, resolution, 'uv') for s in values]
    counter = itertools.count()

    def run():
        return anchor_to_s(anchors[next(counter) & (N - 1)], resolution, 'uv')

    return run


def bench_anchor_to_s_res_5(benchmark):
    benchmark(_make_anchor_to_s(5))


def bench_anchor_to_s_res_15(benchmark):
    benchmark(_make_anchor_to_s(15))


def bench_anchor_to_s_res_28(benchmark):
    benchmark(_make_anchor_to_s(28))


def bench_ij_to_s_res_15(benchmark):
    values = sample_s(15, N)
    ijs = [s_to_anchor(s, 15, 'uv').offset for s in values]
    counter = itertools.count()
    benchmark(lambda: ij_to_s(ijs[next(counter) & (N - 1)], 15, 'uv'))
