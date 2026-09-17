"""
A5 Python package.

The public API below is served by one of two interchangeable backends -- the
pure-Python reference implementation, or PyO3 bindings to the a5-rs crate.
Signatures, return types and exceptions are identical either way; see
:mod:`a5._backend` for how one is chosen and :func:`get_backend` for which one
is active.
"""

from a5._backend import BACKEND, get_backend

if BACKEND == 'rust':
    from a5._native import (
        # Indexing
        cell_to_boundary, cell_to_lonlat, lonlat_to_cell,
        hex_to_u64, u64_to_hex,
        # Hierarchy
        cell_to_parent, cell_to_children, get_resolution, get_res0_cells,
        get_num_cells, get_num_children, cell_area, cell_edge_length_avg,
        # Compaction
        compact, uncompact,
        # Traversal
        grid_disk, grid_disk_vertex, spherical_cap, line_string_to_cells,
        # Regions
        polygon_to_cells,
    )
else:
    # Indexing
    from a5.core.cell import cell_to_boundary, cell_to_lonlat, lonlat_to_cell
    from a5.core.hex import hex_to_u64, u64_to_hex

    # Hierarchy
    from a5.core.serialization import cell_to_parent, cell_to_children, get_resolution, get_res0_cells
    from a5.core.cell_info import get_num_cells, get_num_children, cell_area, cell_edge_length_avg

    # Compaction
    from a5.core.compact import compact, uncompact

    # Traversal
    from a5.traversal import grid_disk, grid_disk_vertex, spherical_cap, line_string_to_cells

    # Regions
    from a5.regions import polygon_to_cells

# Constants and types are backend-independent -- they carry no computation.
from a5.core.serialization import MAX_RESOLUTION, WORLD_CELL
from a5.core.coordinate_systems import Degrees, Radians
from a5.core.utils import A5Cell

__all__ = [
    # Indexing
    'cell_to_boundary', 'cell_to_lonlat', 'lonlat_to_cell',
    'hex_to_u64', 'u64_to_hex',
    # Hierarchy
    'cell_to_parent', 'cell_to_children', 'get_resolution', 'get_res0_cells', 'MAX_RESOLUTION', 'WORLD_CELL',
    'get_num_cells', 'get_num_children', 'cell_area', 'cell_edge_length_avg',
    # Compaction
    'compact', 'uncompact',
    # Traversal
    'grid_disk', 'grid_disk_vertex', 'spherical_cap', 'line_string_to_cells',
    # Regions
    'polygon_to_cells',
    # Types
    'Degrees', 'Radians', 'A5Cell',
    # Backend introspection (Python port only)
    'get_backend',
]
