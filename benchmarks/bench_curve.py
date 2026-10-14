# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

# Benchmarks for the space-filling curve: s -> cell decode and cell -> s encode.

from typing import List

from a5.lattice import Triple, s_to_cell, triple_to_s

from .utils import BATCH, create_random


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


def triples_of(values: List[int], resolution: int, orientation: str) -> List[Triple]:
    """The triples of the cells at `values`."""
    return [s_to_cell(values[i], resolution, orientation).triple for i in range(len(values))]


def _make_s_to_cell(resolution, orientation):
    values = sample_s(resolution, BATCH)

    def run():
        for i in range(BATCH):
            s_to_cell(values[i], resolution, orientation)

    return run


def _make_triple_to_s(resolution, orientation):
    triples = triples_of(sample_s(resolution, BATCH), resolution, orientation)

    def run():
        for i in range(BATCH):
            triple_to_s(triples[i], resolution, orientation)

    return run


def bench_s_to_cell_res_5_x100(benchmark):
    benchmark(_make_s_to_cell(5, 'uv'))


def bench_s_to_cell_res_15_x100(benchmark):
    benchmark(_make_s_to_cell(15, 'uv'))


def bench_s_to_cell_res_28_x100(benchmark):
    benchmark(_make_s_to_cell(28, 'uv'))


# Orientation with both flip and reversal transforms
def bench_s_to_cell_res_15_wu_x100(benchmark):
    benchmark(_make_s_to_cell(15, 'wu'))


def bench_triple_to_s_res_5_x100(benchmark):
    benchmark(_make_triple_to_s(5, 'uv'))


def bench_triple_to_s_res_15_x100(benchmark):
    benchmark(_make_triple_to_s(15, 'uv'))


def bench_triple_to_s_res_28_x100(benchmark):
    benchmark(_make_triple_to_s(28, 'uv'))


def bench_triple_to_s_res_15_wu_x100(benchmark):
    benchmark(_make_triple_to_s(15, 'wu'))
