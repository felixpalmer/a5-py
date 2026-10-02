# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

# Benchmarks for the space-filling curve: cell -> s encode (triple_to_s) and
# fractional-point location (ij_to_s).
#
# CI runs this same file against both the PR and its merge-base with main, so
# it must import and run on either side of the L-system migration. It uses ONLY
# the API common to both engines -- `triple_to_s` and `ij_to_s` -- whose
# signatures and (bit-identical) behavior are unchanged across the swap, so both
# runs measure the equivalent operation on identical inputs. The decode
# primitive changed name across the migration (s_to_anchor -> s_to_cell) with no
# common symbol, so it is not benchmarked here.

import itertools
from typing import List

from a5.lattice import ij_to_s, triple_in_bounds, triple_to_s, Triple

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


def sample_triples(resolution: int, n: int, seed: int = 42) -> List[Triple]:
    """
    Deterministic valid triples in the quintant, derived from the shared PRNG
    sample. Construction guarantees parity in {0,1} and in-bounds coordinates.
    """
    raw = sample_s(resolution, n, seed)
    max_row = (1 << resolution) - 1
    out: List[Triple] = []
    for r in raw:
        y = r % (max_row + 1)
        p = (r >> 20) & 1
        if y - p < 0:
            p = 0
        span = y - p
        x = -((r >> 8) % (span + 1))
        z = p - x - y
        t = Triple(x, y, z)
        out.append(t if triple_in_bounds(t, max_row) else Triple(0, 0, 0))
    return out


def _centroid_ij(t: Triple):
    """The cell's centroid in IJ coordinates
    (parity 0: (x+y+1/3, -x+1/3), parity 1: (x+y-1/3, -x+2/3))."""
    parity = t.x + t.y + t.z
    if parity == 0:
        return (t.x + t.y + 1 / 3, -t.x + 1 / 3)
    return (t.x + t.y - 1 / 3, -t.x + 2 / 3)


def _make_triple_to_s(resolution, orientation):
    triples = sample_triples(resolution, N)
    counter = itertools.count()

    def run():
        return triple_to_s(triples[next(counter) & (N - 1)], resolution, orientation)

    return run


def bench_triple_to_s_res_5(benchmark):
    benchmark(_make_triple_to_s(5, 'uv'))


def bench_triple_to_s_res_15(benchmark):
    benchmark(_make_triple_to_s(15, 'uv'))


def bench_triple_to_s_res_28(benchmark):
    benchmark(_make_triple_to_s(28, 'uv'))


def bench_triple_to_s_res_15_wu(benchmark):
    benchmark(_make_triple_to_s(15, 'wu'))


def bench_ij_to_s_res_15(benchmark):
    ijs = [_centroid_ij(t) for t in sample_triples(15, N)]
    counter = itertools.count()
    benchmark(lambda: ij_to_s(ijs[next(counter) & (N - 1)], 15, 'uv'))
