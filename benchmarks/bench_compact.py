# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

from a5 import compact, polygon_to_cells, uncompact

from .utils import country_polygon

uk = country_polygon('United Kingdom')

# A realistic mixed-resolution cell set: country fill expanded to a flat list
flat = uncompact(polygon_to_cells(uk, 10))
compacted_12 = polygon_to_cells(uk, 12)


def bench_compact_uk_res_10(benchmark):
    # flat cell count captured at collection time
    benchmark(lambda: compact(flat))


def bench_uncompact_uk_res_12(benchmark):
    benchmark(lambda: uncompact(compacted_12))
