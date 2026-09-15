# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

"""Batch variants of the cheap hierarchy and cell-info operations.

``cell_to_parent``, ``cell_to_children``, ``get_resolution`` and ``cell_area``
are each a handful of shifts and masks. Called one at a time through the
compiled backend, the Python/Rust boundary crossing costs more than the work
itself, so the scalar bindings show little gain over pure Python. These
functions take a sequence and return a list, so the crossing is paid once per
batch instead of once per element, and the compiled backend runs the loop with
the GIL released.

Signatures:

* ``cell_to_parent(cells, parent_resolution=None) -> List[int]``
* ``cell_to_children(cells, child_resolution=None) -> List[List[int]]``
* ``get_resolution(cells) -> List[int]``
* ``cell_area(resolutions) -> List[float]`` -- resolutions, not cell IDs,
  mirroring the scalar ``a5.cell_area(resolution)``

Element semantics are exactly those of the scalar functions, including the
exceptions they raise. The names deliberately mirror the scalar API rather than
being suffixed, so ``from a5 import batch`` then
``batch.cell_to_parent(cells, 5)`` reads as the plural of
``a5.cell_to_parent(cell, 5)``.

They live in their own module because they are specific to the Python port --
the TypeScript and Rust implementations have no boundary to amortise. Both
backends implement them; on pure Python they are ordinary comprehensions, so
switching backends changes speed and nothing else.
"""

from typing import List, Optional, Sequence

from a5._backend import BACKEND, native_module

__all__ = ['cell_to_parent', 'cell_to_children', 'get_resolution', 'cell_area']

if BACKEND == 'rust':
    _native = native_module()

    # Bound directly rather than wrapped: a forwarding def would reintroduce
    # per-call Python overhead on the very path these exist to remove.
    cell_to_parent = _native.cell_to_parent_batch
    cell_to_children = _native.cell_to_children_batch
    get_resolution = _native.get_resolution_batch
    cell_area = _native.cell_area_batch
else:
    from a5.core.cell_info import cell_area as _cell_area
    from a5.core.serialization import (
        cell_to_children as _cell_to_children,
        cell_to_parent as _cell_to_parent,
        get_resolution as _get_resolution,
    )

    def cell_to_parent(cells: Sequence[int], parent_resolution: Optional[int] = None) -> List[int]:
        """Parent of each cell, or one level up when parent_resolution is None."""
        return [_cell_to_parent(cell, parent_resolution) for cell in cells]

    def cell_to_children(cells: Sequence[int], child_resolution: Optional[int] = None) -> List[List[int]]:
        """Children of each cell, or one level down when child_resolution is None."""
        return [_cell_to_children(cell, child_resolution) for cell in cells]

    def get_resolution(cells: Sequence[int]) -> List[int]:
        """Resolution of each cell."""
        return [_get_resolution(cell) for cell in cells]

    def cell_area(resolutions: Sequence[int]) -> List[float]:
        """Cell area in square metres for each resolution level."""
        return [_cell_area(resolution) for resolution in resolutions]
