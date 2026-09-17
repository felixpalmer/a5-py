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

Only ``cell_to_boundary`` and ``polygon_to_cells`` need more than a re-export:
they take an options mapping, which the Rust API models as a typed struct.

This layer does not paper over a5-rs defects. Where the two backends disagree,
the divergence is recorded in RUST_BUGS.md and pinned by a test, so it gets
fixed upstream rather than hidden here.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from . import _a5

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
hex_to_u64 = _a5.hex_to_u64
u64_to_hex = _a5.u64_to_hex

cell_to_parent = _a5.cell_to_parent
cell_to_children = _a5.cell_to_children
get_resolution = _a5.get_resolution
get_res0_cells = _a5.get_res0_cells
get_num_cells = _a5.get_num_cells
get_num_children = _a5.get_num_children
cell_area = _a5.cell_area
cell_edge_length_avg = _a5.cell_edge_length_avg

compact = _a5.compact
uncompact = _a5.uncompact

grid_disk = _a5.grid_disk
grid_disk_vertex = _a5.grid_disk_vertex
spherical_cap = _a5.spherical_cap
line_string_to_cells = _a5.line_string_to_cells


# -- Adapted entry points --------------------------------------------------

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
