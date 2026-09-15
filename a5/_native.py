# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

"""Compiled backend: the public A5 API, backed by the a5-rs crate.

Importing this module requires the extension module ``a5._a5``; use
:func:`a5.get_backend` rather than importing it directly.

Most bindings in ``a5._a5`` already match the pure-Python signatures exactly --
same argument order, same argument names (so keyword calls work), same defaults
-- and are re-exported here unwrapped, because a Python-level forwarding
function would cost more than the call it forwards for the cheap operations.

Three groups need more than a re-export, each marked below:

* ``cell_to_boundary`` and ``polygon_to_cells`` take an options mapping, which
  the Rust API models as a typed struct.
* ``get_num_cells``, ``get_num_children``, ``hex_to_u64`` and ``u64_to_hex``
  are served from the pure-Python implementation, because a5-rs behaves
  differently there and the public API must not change with the backend.
* ``get_res0_cells`` caches, as the pure-Python implementation does.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from . import _a5

# These four are re-exported unchanged from the pure-Python modules; see the
# "Pure-Python for correctness" section below for why.
from .core.cell_info import get_num_cells, get_num_children
from .core.hex import hex_to_u64, u64_to_hex

__all__ = [
    'cell_to_boundary', 'cell_to_lonlat', 'lonlat_to_cell',
    'hex_to_u64', 'u64_to_hex',
    'cell_to_parent', 'cell_to_children', 'get_resolution', 'get_res0_cells',
    'get_num_cells', 'get_num_children', 'cell_area', 'cell_edge_length_avg',
    'compact', 'uncompact',
    'grid_disk', 'grid_disk_vertex', 'spherical_cap', 'line_string_to_cells',
    'polygon_to_cells',
]

# -- Direct re-exports: signature-compatible as-is -------------------------

cell_to_lonlat = _a5.cell_to_lonlat
lonlat_to_cell = _a5.lonlat_to_cell

cell_to_parent = _a5.cell_to_parent
cell_to_children = _a5.cell_to_children
get_resolution = _a5.get_resolution
cell_area = _a5.cell_area
cell_edge_length_avg = _a5.cell_edge_length_avg

compact = _a5.compact
uncompact = _a5.uncompact

grid_disk = _a5.grid_disk
grid_disk_vertex = _a5.grid_disk_vertex
spherical_cap = _a5.spherical_cap
line_string_to_cells = _a5.line_string_to_cells


# -- Pure-Python for correctness -------------------------------------------
#
# `get_num_cells` / `get_num_children` (imported above):
#   a5-rs hard-codes the JavaScript double-rounded values for
#   get_num_cells(28..30) to match the TypeScript `number` overload, although it
#   returns u64. Python implements the exact `bigint` overload, which is what the
#   shared fixture records as `countBigInt`. Binding the Rust version would make
#   `a5.get_num_cells(28)` depend on which backend is active.
#
# `hex_to_u64` / `u64_to_hex` (imported above):
#   a5-rs parses hex with `u64::from_str_radix`, which rejects the `0x` prefix,
#   digit separators, surrounding whitespace and a leading `-`, all of which
#   `int(s, 16)` accepts. Switching backend must not change which inputs a
#   caller can pass.
#
# None of the four has anything to win from crossing into Rust: they are a few
# integer operations, and `int(s, 16)` / `hex(v)` are already C. The a5-rs
# bindings stay available as `_a5.get_num_cells`, `_a5.get_num_children`,
# `_a5.hex_to_u64` and `_a5.u64_to_hex`, where tests/test_differential.py tracks
# the upstream behaviour. When a5-rs is fixed and the pin bumped, that test
# fails and the cell-count import can go.


# -- Adapted entry points --------------------------------------------------

_res0_cells = None


def get_res0_cells() -> List[int]:
    """Return the 12 resolution-0 cells (dodecahedron faces).

    Cached like the pure-Python implementation, which is otherwise twice as fast
    here: these 12 cells are a constant, and recomputing them through a boundary
    crossing on every call is pure loss. A fresh list is returned each time so a
    caller mutating the result cannot corrupt the cache.
    """
    global _res0_cells
    if _res0_cells is None:
        _res0_cells = _a5.get_res0_cells()
    return list(_res0_cells)


def cell_to_boundary(
    cell_id: int,
    options: Optional[Dict[str, Any]] = None,
) -> List[Tuple[float, float]]:
    """Get the boundary coordinates of a cell.

    See :func:`a5.core.cell.cell_to_boundary` for the full contract. The
    ``'auto'`` sentinel for ``segments`` is passed to Rust as ``None``, which
    means the same thing there.
    """
    closed_ring = True
    segments = None
    if options is not None:
        closed_ring = options.get('closed_ring', True)
        segments = options.get('segments', 'auto')
        if segments == 'auto':
            segments = None
    return _a5.cell_to_boundary(cell_id, closed_ring, segments)


def polygon_to_cells(
    polygon: Union[Sequence[Tuple[float, float]], Sequence[Sequence[Tuple[float, float]]]],
    resolution: int,
    options: Optional[Dict[str, Any]] = None,
) -> List[int]:
    """Find all cells within a polygon.

    See :func:`a5.regions.polygon.polygon_to_cells` for the full contract. The
    Rust API only accepts GeoJSON-style ``[outer, *holes]``, so the bare-ring
    shorthand is expanded here, using the same test the pure-Python
    implementation uses.
    """
    if len(polygon) == 0:
        return []
    is_nested = not isinstance(polygon[0][0], (int, float))
    rings = polygon if is_nested else [polygon]
    containment = (options or {}).get('containment', 'center')
    return _a5.polygon_to_cells(rings, resolution, containment == 'overlapping')
