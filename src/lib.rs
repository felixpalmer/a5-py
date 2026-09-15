// A5
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) A5 contributors

//! PyO3 bindings for the `a5` crate (a5-rs), exposed to Python as `a5._a5`.
//!
//! This module is private. The supported Python API is the `a5` package, which
//! selects between these bindings and the pure-Python implementation at import
//! time (see `a5/_backend.py`).
//!
//! Design rules for this layer:
//!
//! * **No API divergence.** Every function mirrors the pure-Python signature
//!   exactly -- same order, same parameter names (so keyword calls work), same
//!   defaults, same exception type (`ValueError`, since that is what the
//!   pure-Python implementation raises). The two places where the Python API is
//!   richer than the Rust one -- the `options` dicts of `cell_to_boundary` and
//!   `polygon_to_cells`, and `polygon_to_cells` accepting a bare ring -- are
//!   normalised by the thin shim in `a5/_native.py` before reaching here.
//! * **Release the GIL** for anything that is not O(1): the batch entry points
//!   and the traversal/region functions all run without holding it.
//! * **No panics.** `panic = "abort"` is set in `[profile.release]`, so a panic
//!   would take down the interpreter instead of raising. Inputs are validated
//!   here to the same bounds the pure-Python implementation enforces.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use a5::{LonLat, MAX_RESOLUTION};

/// a5-rs reports failures as `String`; the pure-Python implementation raises
/// `ValueError` with the same message text, so map straight onto it.
#[inline]
fn to_py(err: String) -> PyErr {
    PyValueError::new_err(err)
}

/// Points arrive as `[f64; 2]` rather than `(f64, f64)` so that both `(lon, lat)`
/// tuples and `[lon, lat]` lists are accepted -- GeoJSON-shaped input is lists,
/// and the pure-Python implementation simply indexes, so it takes either.
#[inline]
fn lonlat(p: [f64; 2]) -> LonLat {
    LonLat::new(p[0], p[1])
}

/// Clamp a hop count to the non-negative range.
///
/// `grid_disk` takes `usize` in Rust but `int` in Python, and the pure-Python
/// BFS treats any `k <= 0` as "just the centre cell" (the ring loop body never
/// runs). Clamping reproduces that instead of raising `OverflowError` on the
/// `usize` conversion.
#[inline]
fn hops(k: i64) -> usize {
    if k < 0 {
        0
    } else {
        k as usize
    }
}

// ---------------------------------------------------------------------------
// Indexing
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (lon_lat, resolution))]
fn lonlat_to_cell(lon_lat: [f64; 2], resolution: i32) -> PyResult<u64> {
    a5::lonlat_to_cell(lonlat(lon_lat), resolution).map_err(to_py)
}

#[pyfunction]
#[pyo3(signature = (cell_id))]
fn cell_to_lonlat(cell_id: u64) -> PyResult<(f64, f64)> {
    a5::cell_to_lonlat(cell_id)
        .map(|ll| (ll.longitude(), ll.latitude()))
        .map_err(to_py)
}

/// `options` handling lives in the Python shim; this takes the resolved values.
/// `segments = None` means "derive from the cell's resolution".
#[pyfunction]
#[pyo3(signature = (cell_id, closed_ring, segments))]
fn cell_to_boundary(
    py: Python<'_>,
    cell_id: u64,
    closed_ring: bool,
    segments: Option<i32>,
) -> PyResult<Vec<(f64, f64)>> {
    let options = a5::core::cell::CellToBoundaryOptions {
        closed_ring,
        segments,
    };
    py.allow_threads(|| a5::cell_to_boundary(cell_id, Some(options)))
        .map(|ring| ring.into_iter().map(|ll| (ll.longitude(), ll.latitude())).collect())
        .map_err(to_py)
}

#[pyfunction]
#[pyo3(signature = (hex_str))]
fn hex_to_u64(hex_str: &str) -> PyResult<u64> {
    a5::hex_to_u64(hex_str).map_err(to_py)
}

#[pyfunction]
#[pyo3(signature = (value))]
fn u64_to_hex(value: u64) -> String {
    a5::u64_to_hex(value)
}

// ---------------------------------------------------------------------------
// Hierarchy
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (index))]
fn get_resolution(index: u64) -> i32 {
    a5::get_resolution(index)
}

#[pyfunction]
#[pyo3(signature = (index, parent_resolution=None))]
fn cell_to_parent(index: u64, parent_resolution: Option<i32>) -> PyResult<u64> {
    a5::cell_to_parent(index, parent_resolution).map_err(to_py)
}

#[pyfunction]
#[pyo3(signature = (index, child_resolution=None))]
fn cell_to_children(
    py: Python<'_>,
    index: u64,
    child_resolution: Option<i32>,
) -> PyResult<Vec<u64>> {
    py.allow_threads(|| a5::cell_to_children(index, child_resolution))
        .map_err(to_py)
}

#[pyfunction]
fn get_res0_cells() -> PyResult<Vec<u64>> {
    a5::get_res0_cells().map_err(to_py)
}

#[pyfunction]
#[pyo3(signature = (resolution))]
fn get_num_cells(resolution: i32) -> u64 {
    a5::get_num_cells(resolution)
}

#[pyfunction]
#[pyo3(signature = (parent_resolution, child_resolution))]
fn get_num_children(parent_resolution: i32, child_resolution: i32) -> usize {
    a5::get_num_children(parent_resolution, child_resolution)
}

#[pyfunction]
#[pyo3(signature = (resolution))]
fn cell_area(resolution: i32) -> f64 {
    a5::cell_area(resolution)
}

#[pyfunction]
#[pyo3(signature = (resolution))]
fn cell_edge_length_avg(resolution: i32) -> f64 {
    a5::cell_edge_length_avg(resolution)
}

// ---------------------------------------------------------------------------
// Compaction
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (cells))]
fn compact(py: Python<'_>, cells: Vec<u64>) -> PyResult<Vec<u64>> {
    py.allow_threads(|| a5::compact(&cells)).map_err(to_py)
}

#[pyfunction]
#[pyo3(signature = (cells, target_resolution))]
fn uncompact(py: Python<'_>, cells: Vec<u64>, target_resolution: i32) -> PyResult<Vec<u64>> {
    py.allow_threads(|| a5::uncompact(&cells, target_resolution))
        .map_err(to_py)
}

// ---------------------------------------------------------------------------
// Traversal
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (cell_id, k))]
fn grid_disk(py: Python<'_>, cell_id: u64, k: i64) -> PyResult<Vec<u64>> {
    py.allow_threads(|| a5::grid_disk(cell_id, hops(k)))
        .map_err(to_py)
}

#[pyfunction]
#[pyo3(signature = (cell_id, k))]
fn grid_disk_vertex(py: Python<'_>, cell_id: u64, k: i64) -> PyResult<Vec<u64>> {
    py.allow_threads(|| a5::grid_disk_vertex(cell_id, hops(k)))
        .map_err(to_py)
}

#[pyfunction]
#[pyo3(signature = (cell_id, radius))]
fn spherical_cap(py: Python<'_>, cell_id: u64, radius: f64) -> PyResult<Vec<u64>> {
    py.allow_threads(|| a5::spherical_cap(cell_id, radius))
        .map_err(to_py)
}

#[pyfunction]
#[pyo3(signature = (waypoints, resolution))]
fn line_string_to_cells(
    py: Python<'_>,
    waypoints: Vec<[f64; 2]>,
    resolution: i32,
) -> PyResult<Vec<u64>> {
    let waypoints: Vec<LonLat> = waypoints.into_iter().map(lonlat).collect();
    py.allow_threads(|| a5::line_string_to_cells(&waypoints, resolution))
        .map_err(to_py)
}

// ---------------------------------------------------------------------------
// Regions
// ---------------------------------------------------------------------------

/// `polygon` always arrives as `[outer, *holes]` -- the Python shim expands the
/// bare-ring shorthand. `overlapping` is the resolved `containment` option.
#[pyfunction]
#[pyo3(signature = (polygon, resolution, overlapping))]
fn polygon_to_cells(
    py: Python<'_>,
    polygon: Vec<Vec<[f64; 2]>>,
    resolution: i32,
    overlapping: bool,
) -> PyResult<Vec<u64>> {
    let rings: Vec<Vec<LonLat>> = polygon
        .into_iter()
        .map(|ring| ring.into_iter().map(lonlat).collect())
        .collect();
    let options = a5::regions::polygon::PolygonToCellsOptions {
        containment: if overlapping {
            a5::regions::polygon::Containment::Overlapping
        } else {
            a5::regions::polygon::Containment::Center
        },
    };
    py.allow_threads(|| a5::polygon_to_cells(&rings, resolution, Some(options)))
        .map_err(to_py)
}

// ---------------------------------------------------------------------------
// Batch variants
//
// The hierarchy and cell-info operations are individually far cheaper than a
// Python<->Rust call, so per-call FFI overhead dominates and the scalar
// bindings show little or no gain over pure Python. These take a sequence and
// return a list, amortising the crossing over the whole batch, and run with the
// GIL released.
// ---------------------------------------------------------------------------

/// Parent of each cell, or one level up when parent_resolution is None.
#[pyfunction]
#[pyo3(signature = (cells, parent_resolution=None))]
fn cell_to_parent_batch(
    py: Python<'_>,
    cells: Vec<u64>,
    parent_resolution: Option<i32>,
) -> PyResult<Vec<u64>> {
    py.allow_threads(|| {
        cells
            .iter()
            .map(|&cell| a5::cell_to_parent(cell, parent_resolution))
            .collect::<Result<Vec<u64>, String>>()
    })
    .map_err(to_py)
}

/// Children of each cell, or one level down when child_resolution is None.
#[pyfunction]
#[pyo3(signature = (cells, child_resolution=None))]
fn cell_to_children_batch(
    py: Python<'_>,
    cells: Vec<u64>,
    child_resolution: Option<i32>,
) -> PyResult<Vec<Vec<u64>>> {
    py.allow_threads(|| {
        cells
            .iter()
            .map(|&cell| a5::cell_to_children(cell, child_resolution))
            .collect::<Result<Vec<Vec<u64>>, String>>()
    })
    .map_err(to_py)
}

/// Resolution of each cell.
#[pyfunction]
#[pyo3(signature = (cells))]
fn get_resolution_batch(py: Python<'_>, cells: Vec<u64>) -> Vec<i32> {
    py.allow_threads(|| cells.iter().map(|&cell| a5::get_resolution(cell)).collect())
}

/// Cell area in square metres for each resolution level.
#[pyfunction]
#[pyo3(signature = (resolutions))]
fn cell_area_batch(py: Python<'_>, resolutions: Vec<i32>) -> Vec<f64> {
    py.allow_threads(|| resolutions.iter().map(|&res| a5::cell_area(res)).collect())
}

// ---------------------------------------------------------------------------

#[pymodule]
#[pyo3(name = "_a5")]
fn a5_ext(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Indexing
    m.add_function(wrap_pyfunction!(lonlat_to_cell, m)?)?;
    m.add_function(wrap_pyfunction!(cell_to_lonlat, m)?)?;
    m.add_function(wrap_pyfunction!(cell_to_boundary, m)?)?;
    m.add_function(wrap_pyfunction!(hex_to_u64, m)?)?;
    m.add_function(wrap_pyfunction!(u64_to_hex, m)?)?;

    // Hierarchy
    m.add_function(wrap_pyfunction!(get_resolution, m)?)?;
    m.add_function(wrap_pyfunction!(cell_to_parent, m)?)?;
    m.add_function(wrap_pyfunction!(cell_to_children, m)?)?;
    m.add_function(wrap_pyfunction!(get_res0_cells, m)?)?;
    m.add_function(wrap_pyfunction!(get_num_cells, m)?)?;
    m.add_function(wrap_pyfunction!(get_num_children, m)?)?;
    m.add_function(wrap_pyfunction!(cell_area, m)?)?;
    m.add_function(wrap_pyfunction!(cell_edge_length_avg, m)?)?;

    // Compaction
    m.add_function(wrap_pyfunction!(compact, m)?)?;
    m.add_function(wrap_pyfunction!(uncompact, m)?)?;

    // Traversal
    m.add_function(wrap_pyfunction!(grid_disk, m)?)?;
    m.add_function(wrap_pyfunction!(grid_disk_vertex, m)?)?;
    m.add_function(wrap_pyfunction!(spherical_cap, m)?)?;
    m.add_function(wrap_pyfunction!(line_string_to_cells, m)?)?;

    // Regions
    m.add_function(wrap_pyfunction!(polygon_to_cells, m)?)?;

    // Batch
    m.add_function(wrap_pyfunction!(cell_to_parent_batch, m)?)?;
    m.add_function(wrap_pyfunction!(cell_to_children_batch, m)?)?;
    m.add_function(wrap_pyfunction!(get_resolution_batch, m)?)?;
    m.add_function(wrap_pyfunction!(cell_area_batch, m)?)?;

    // Constants. `a5/__init__.py` serves these from pure Python on both
    // backends, since they carry no computation; exposing them here lets
    // tests/test_differential.py::test_constants_agree check that stays safe.
    m.add("MAX_RESOLUTION", MAX_RESOLUTION)?;
    m.add("WORLD_CELL", a5::WORLD_CELL)?;

    Ok(())
}
