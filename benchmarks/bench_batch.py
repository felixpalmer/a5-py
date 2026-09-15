# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

"""Batch hierarchy/cell-info operations.

These exist because a Python/Rust boundary crossing costs more than the shift
and mask each of these operations performs, so the scalar bindings gain little.
The benchmarks below deliberately mirror bench_hierarchy.py one-for-one, so the
two files can be read side by side: divide a scalar timing by the batch size to
see what the crossing was costing.
"""

import itertools

from a5 import batch

from .utils import sample_cells

# Batch size. Large enough that the per-call boundary crossing is amortised into
# noise, small enough to stay in cache.
BATCH = 1024

cells15 = sample_cells(15, BATCH)
cells10 = sample_cells(10, BATCH)

# Rotated over so no single run benchmarks one fixed slice of the corpus.
CHUNKS = 8
CHUNK = BATCH // CHUNKS
chunks15 = [cells15[i * CHUNK:(i + 1) * CHUNK] for i in range(CHUNKS)]
resolutions = [(i % 31) - 1 for i in range(CHUNK)]


def bench_batch_get_resolution_res_15(benchmark):
    benchmark(lambda: batch.get_resolution(cells15))


def bench_batch_cell_to_parent_res_15_to_14(benchmark):
    benchmark(lambda: batch.cell_to_parent(cells15))


def bench_batch_cell_to_parent_res_15_to_5(benchmark):
    benchmark(lambda: batch.cell_to_parent(cells15, 5))


def bench_batch_cell_to_children_res_15_to_16(benchmark):
    counter = itertools.count()
    benchmark(lambda: batch.cell_to_children(chunks15[next(counter) % CHUNKS]))


def bench_batch_cell_to_children_res_10_to_13(benchmark):
    benchmark(lambda: batch.cell_to_children(cells10[:CHUNK], 13))


def bench_batch_cell_area(benchmark):
    benchmark(lambda: batch.cell_area(resolutions))
