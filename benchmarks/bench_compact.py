# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

import inspect

from a5 import compact, polygon_to_cells, uncompact

from .utils import country_polygon

uk = country_polygon('United Kingdom')


def _uncompact_at(cells, resolution):
    # The resolution argument is for the pre-collection uncompact(cells, resolution),
    # which the baseline run may use; uncompact now reads it from the compaction marker.
    if len(inspect.signature(uncompact).parameters) == 2:
        return uncompact(cells, resolution)
    return uncompact(cells)


# A realistic mixed-resolution cell set: country fill expanded to a flat list
flat = _uncompact_at(polygon_to_cells(uk, 10), 10)
compacted_12 = polygon_to_cells(uk, 12)


def bench_compact_uk_res_10(benchmark):
    # flat cell count captured at collection time
    benchmark(lambda: compact(flat))


def bench_uncompact_uk_res_12(benchmark):
    benchmark(lambda: _uncompact_at(compacted_12, 12))
