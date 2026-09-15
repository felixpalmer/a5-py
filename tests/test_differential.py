# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

"""Differential tests: compiled backend against the pure-Python reference.

The rest of the suite runs once per backend and checks each against the shared
cross-port fixtures. That catches anything a fixture covers. This module closes
the remaining gap by running *both* implementations in the same process over a
much larger generated corpus and comparing them to each other, so a divergence
in a case no fixture happens to exercise still shows up.

The pure-Python implementation is the reference: it is what the fixtures were
written against, so where the two disagree, Python is presumed right until
proven otherwise.

Skipped entirely when the extension module is not built.
"""

import json
import math
import random
from pathlib import Path

import pytest

# Skips the whole module when the extension is not built. Must come before the
# `a5._native` import below, which requires it.
pytest.importorskip(
    'a5._a5',
    reason='compiled extension a5._a5 is not built; run `maturin develop --release`',
)

from a5._backend import native_module  # noqa: E402

from a5._native import cell_to_boundary as rs_cell_to_boundary  # noqa: E402
from a5._native import polygon_to_cells as rs_polygon_to_cells  # noqa: E402

import a5.core.cell as py_cell  # noqa: E402
import a5.core.cell_info as py_cell_info
import a5.core.compact as py_compact
import a5.core.hex as py_hex
import a5.core.serialization as py_serialization
import a5.regions.polygon as py_polygon
import a5.traversal.cap as py_cap
import a5.traversal.line as py_line

# `a5.traversal.__init__` rebinds the name `grid_disk` to the function, shadowing
# the submodule, so these are imported by name rather than through the package.
from a5.traversal.grid_disk import grid_disk as py_grid_disk
from a5.traversal.grid_disk import grid_disk_vertex as py_grid_disk_vertex

native = native_module()

# Degrees. The two ports run the same algorithm in different languages, so they
# agree far more tightly than this in practice (~2e-13 observed), but per the
# porting policy cross-language float results are compared at 1e-10 rather than
# required to be bit-identical. 1e-10 degrees is ~0.01 mm at the equator.
TOLERANCE = 1e-10

FIXTURES = Path(__file__).parent


def _sample_points(n, seed=20260914):
    """Points distributed uniformly over the sphere (area-uniform in latitude)."""
    rng = random.Random(seed)
    return [
        (rng.uniform(-180.0, 180.0), math.degrees(math.asin(rng.uniform(-1.0, 1.0))))
        for _ in range(n)
    ]


POINTS = _sample_points(400)
RESOLUTIONS = (0, 1, 2, 3, 5, 8, 12, 16, 20, 25, 29, 30)


def _cells(resolution, count=60):
    return [py_cell.lonlat_to_cell(p, resolution) for p in POINTS[:count]]


def _assert_lonlat_close(actual, expected, context):
    assert len(actual) == len(expected), context
    for i, (a, b) in enumerate(zip(actual, expected)):
        assert abs(a[0] - b[0]) < TOLERANCE, (context, i, 'lon', a, b)
        assert abs(a[1] - b[1]) < TOLERANCE, (context, i, 'lat', a, b)


class TestIndexing:
    def test_lonlat_to_cell(self):
        for resolution in RESOLUTIONS:
            for point in POINTS:
                assert py_cell.lonlat_to_cell(point, resolution) == native.lonlat_to_cell(
                    point, resolution
                ), (point, resolution)

    def test_lonlat_to_cell_accepts_lists_and_tuples(self):
        # GeoJSON-shaped input is lists; the pure-Python implementation just
        # indexes, so the bindings must not insist on tuples.
        for point in POINTS[:20]:
            assert native.lonlat_to_cell(list(point), 10) == native.lonlat_to_cell(point, 10)

    def test_cell_to_lonlat(self):
        for resolution in RESOLUTIONS:
            for cell in _cells(resolution):
                _assert_lonlat_close(
                    [native.cell_to_lonlat(cell)],
                    [py_cell.cell_to_lonlat(cell)],
                    (cell, resolution),
                )

    def test_cell_to_lonlat_round_trips_through_world_cell(self):
        assert native.cell_to_lonlat(py_serialization.WORLD_CELL) == py_cell.cell_to_lonlat(
            py_serialization.WORLD_CELL
        )

    @pytest.mark.parametrize(
        'options',
        [
            None,
            {},
            {'closed_ring': True},
            {'closed_ring': False},
            {'segments': 1},
            {'segments': 10},
            {'segments': 'auto'},
            {'segments': None},
            {'segments': 4, 'closed_ring': False},
        ],
    )
    def test_cell_to_boundary(self, options):
        for resolution in (0, 2, 6, 12, 20, 30):
            for cell in _cells(resolution, 20):
                _assert_lonlat_close(
                    rs_cell_to_boundary(cell, options),
                    py_cell.cell_to_boundary(cell, options),
                    (cell, resolution, options),
                )

    def test_hex_round_trip(self):
        # Raw a5-rs bindings, on well-formed input.
        for resolution in RESOLUTIONS:
            for cell in _cells(resolution, 20):
                hex_str = py_hex.u64_to_hex(cell)
                assert native.u64_to_hex(cell) == hex_str
                assert native.hex_to_u64(hex_str) == py_hex.hex_to_u64(hex_str)

    @pytest.mark.parametrize(
        'text', ['1f', '0x1f', '1_f', ' 1f ', 'FF', 'ffffffffffffffff', '10000000000000000']
    )
    def test_public_hex_accepts_the_same_inputs_on_both_backends(self, text):
        # a5-rs parses with u64::from_str_radix and rejects everything except a
        # bare in-range hex string, so the public API serves these from pure
        # Python. See a5/_native.py.
        from a5._native import hex_to_u64

        assert hex_to_u64(text) == py_hex.hex_to_u64(text)


class TestHierarchy:
    def test_get_resolution(self):
        for resolution in RESOLUTIONS:
            for cell in _cells(resolution, 30):
                assert native.get_resolution(cell) == py_serialization.get_resolution(cell)

    def test_cell_to_parent(self):
        for resolution in RESOLUTIONS:
            for cell in _cells(resolution, 20):
                assert native.cell_to_parent(cell) == py_serialization.cell_to_parent(cell)
                # Bound the sweep by the cell's own resolution, not the requested
                # one: res-30 cells normalise to the res-29 layout, so asking for
                # a parent at 30 would be out of domain. See
                # test_known_divergences for what the two ports do there.
                actual_resolution = py_serialization.get_resolution(cell)
                for parent_resolution in range(-1, actual_resolution + 1):
                    assert native.cell_to_parent(
                        cell, parent_resolution
                    ) == py_serialization.cell_to_parent(cell, parent_resolution), (
                        cell,
                        parent_resolution,
                    )

    def test_cell_to_children(self):
        for resolution in (0, 1, 2, 5, 12, 29):
            for cell in _cells(resolution, 10):
                assert native.cell_to_children(cell) == py_serialization.cell_to_children(cell)
                for child_resolution in range(resolution, min(resolution + 4, 31)):
                    assert native.cell_to_children(
                        cell, child_resolution
                    ) == py_serialization.cell_to_children(cell, child_resolution), (
                        cell,
                        child_resolution,
                    )

    def test_get_res0_cells(self):
        from a5._native import get_res0_cells

        assert native.get_res0_cells() == py_serialization.get_res0_cells()
        assert get_res0_cells() == py_serialization.get_res0_cells()

    def test_get_res0_cells_returns_a_fresh_list(self):
        # The result is cached on both backends; a caller mutating it must not
        # corrupt the next call.
        from a5._native import get_res0_cells

        first = get_res0_cells()
        first.append(0)
        assert get_res0_cells() == py_serialization.get_res0_cells()

    def test_cell_area_and_edge_length(self):
        for resolution in range(-1, 31):
            assert native.cell_area(resolution) == py_cell_info.cell_area(resolution), resolution
            assert native.cell_edge_length_avg(resolution) == py_cell_info.cell_edge_length_avg(
                resolution
            ), resolution

    def test_public_cell_counts_agree_across_backends(self):
        # `a5.get_num_cells` / `a5.get_num_children` must return the same values
        # whichever backend is selected, so these are checked against the public
        # surface rather than the raw bindings. See a5/_native.py.
        from a5._native import get_num_cells, get_num_children

        for resolution in range(-1, 31):
            assert get_num_cells(resolution) == py_cell_info.get_num_cells(resolution), resolution
        for parent in range(-1, 31):
            for child in range(-1, 31):
                assert get_num_children(parent, child) == py_cell_info.get_num_children(
                    parent, child
                ), (parent, child)

    def test_get_num_cells(self):
        # Raw a5-rs binding. Resolutions 28+ are a known upstream divergence,
        # see test_known_divergences.
        for resolution in range(-1, 28):
            assert native.get_num_cells(resolution) == py_cell_info.get_num_cells(
                resolution
            ), resolution

    def test_get_num_children(self):
        # Raw a5-rs binding; see test_get_num_cells.
        for parent in range(-1, 31):
            for child in range(-1, 31):
                # get_num_children only consults get_num_cells below the first
                # Hilbert resolution; above it the aperture-4 shortcut is exact
                # in both ports, so only that corner inherits the divergence.
                if parent < py_cell_info.FIRST_HILBERT_RESOLUTION and child >= 28:
                    continue
                assert native.get_num_children(parent, child) == py_cell_info.get_num_children(
                    parent, child
                ), (parent, child)


class TestCompaction:
    def test_uncompact_then_compact(self):
        for resolution in (2, 5, 9, 14):
            cells = _cells(resolution, 30)
            for target in range(resolution, min(resolution + 4, 31)):
                expanded = py_compact.uncompact(cells, target)
                assert native.uncompact(cells, target) == expanded, (resolution, target)
                assert native.compact(expanded) == py_compact.compact(expanded), (
                    resolution,
                    target,
                )

    def test_compact_of_complete_sibling_sets(self):
        for resolution in (3, 7, 11):
            for cell in _cells(resolution, 10):
                children = py_serialization.cell_to_children(cell, resolution + 2)
                assert native.compact(children) == py_compact.compact(children)

    def test_empty_input(self):
        assert native.compact([]) == py_compact.compact([])
        assert native.uncompact([], 5) == py_compact.uncompact([], 5)


class TestTraversal:
    @pytest.mark.parametrize('k', [0, 1, 2, 3])
    def test_grid_disk(self, k):
        for resolution in (2, 5, 10, 18):
            for cell in _cells(resolution, 8):
                assert native.grid_disk(cell, k) == py_grid_disk(cell, k), (cell, k)
                assert native.grid_disk_vertex(cell, k) == py_grid_disk_vertex(cell, k), (cell, k)

    def test_grid_disk_negative_k(self):
        # `k` is usize in Rust and int in Python; the bindings clamp rather than
        # raising OverflowError, matching the pure-Python BFS whose ring loop
        # simply never runs.
        cell = _cells(8, 1)[0]
        for k in (-1, -5):
            assert native.grid_disk(cell, k) == py_grid_disk(cell, k)
            assert native.grid_disk_vertex(cell, k) == py_grid_disk_vertex(cell, k)

    def test_spherical_cap(self):
        for resolution in (5, 9, 14):
            for cell in _cells(resolution, 6):
                for radius in (500.0, 5000.0, 50000.0, 400000.0):
                    assert native.spherical_cap(cell, radius) == py_cap.spherical_cap(
                        cell, radius
                    ), (cell, radius)

    def test_line_string_to_cells_from_fixtures(self):
        with open(FIXTURES / 'traversal' / 'fixtures' / 'line.json') as f:
            cases = json.load(f)['lineSegment']
        for case in cases:
            waypoints = [case['start'], case['end']]
            resolution = case['resolution']
            assert native.line_string_to_cells(
                waypoints, resolution
            ) == py_line.line_string_to_cells(waypoints, resolution), case['name']

    def test_line_string_to_cells_multi_waypoint(self):
        for resolution in (3, 6, 9):
            for start in range(0, 40, 10):
                waypoints = POINTS[start:start + 6]
                assert native.line_string_to_cells(
                    waypoints, resolution
                ) == py_line.line_string_to_cells(waypoints, resolution), (start, resolution)

    def test_line_string_edge_cases(self):
        assert native.line_string_to_cells([], 5) == py_line.line_string_to_cells([], 5)
        single = [POINTS[0]]
        assert native.line_string_to_cells(single, 5) == py_line.line_string_to_cells(single, 5)


class TestRegions:
    @staticmethod
    def _countries():
        with open(FIXTURES / 'regions' / 'fixtures' / 'polygon.json') as f:
            return json.load(f)['country']

    @pytest.mark.parametrize('options', [None, {'containment': 'center'}, {'containment': 'overlapping'}])
    def test_polygon_to_cells_countries(self, options):
        for country in self._countries():
            for resolution in (3, 5, 7):
                assert rs_polygon_to_cells(
                    country['polygon'], resolution, options
                ) == py_polygon.polygon_to_cells(country['polygon'], resolution, options), (
                    country['name'],
                    resolution,
                    options,
                )

    def test_polygon_to_cells_bare_ring_shorthand(self):
        ring = [(0.0, 50.0), (5.0, 50.0), (5.0, 53.0), (0.0, 53.0)]
        for resolution in (4, 6, 8):
            assert rs_polygon_to_cells(ring, resolution) == py_polygon.polygon_to_cells(
                ring, resolution
            ), resolution
            # Closed ring (first vertex repeated) must give the same answer.
            assert rs_polygon_to_cells(
                ring + [ring[0]], resolution
            ) == py_polygon.polygon_to_cells(ring + [ring[0]], resolution), resolution

    def test_polygon_to_cells_degenerate_input(self):
        # Empty input, a ring with too few distinct vertices in both the nested
        # and bare-ring spellings, and a hole dropped for the same reason.
        degenerate = [
            [],
            [(0.0, 0.0), (1.0, 1.0)],
            [[(0.0, 0.0), (1.0, 1.0)]],
            [[(0.0, 50.0), (5.0, 50.0), (5.0, 53.0), (0.0, 53.0)], [(1.0, 51.0), (2.0, 51.0)]],
        ]
        for polygon in degenerate:
            assert rs_polygon_to_cells(polygon, 5) == py_polygon.polygon_to_cells(
                polygon, 5
            ), polygon


class TestErrors:
    """Both backends must raise the same exception type with the same message."""

    def _assert_same_error(self, call_native, call_python, *args):
        with pytest.raises(ValueError) as py_exc:
            call_python(*args)
        with pytest.raises(ValueError) as native_exc:
            call_native(*args)
        assert str(native_exc.value) == str(py_exc.value), args

    def test_cell_to_children_beyond_max_resolution(self):
        cell = _cells(30, 1)[0]
        self._assert_same_error(
            native.cell_to_children, py_serialization.cell_to_children, cell, None
        )

    def test_cell_to_parent_out_of_range(self):
        cell = _cells(10, 1)[0]
        self._assert_same_error(
            native.cell_to_parent, py_serialization.cell_to_parent, cell, 31
        )

    def test_uncompact_to_lower_resolution(self):
        cells = _cells(10, 3)
        self._assert_same_error(native.uncompact, py_compact.uncompact, cells, 4)

    def test_hex_to_u64_rejects_garbage(self):
        # Message text differs (Rust reports the parse failure, Python the
        # ValueError from int()), so only the type is asserted here.
        for backend_fn in (native.hex_to_u64, py_hex.hex_to_u64):
            with pytest.raises(ValueError):
                backend_fn('not-hex')


def test_constants_agree():
    """`a5` serves these from pure Python on both backends -- check that is safe.

    MAX_RESOLUTION and WORLD_CELL carry no computation, so a5/__init__.py imports
    them unconditionally from a5.core.serialization rather than branching. That
    is only correct while a5-rs agrees on their values.
    """
    assert native.MAX_RESOLUTION == py_serialization.MAX_RESOLUTION
    assert native.WORLD_CELL == py_serialization.WORLD_CELL


def test_known_divergences():
    """Pin the one place the two ports disagree, so it cannot spread unnoticed.

    a5-rs hard-codes the JavaScript double-rounded values for
    `get_num_cells(28..30)` to match the TypeScript `number` overload, even
    though it returns u64. Python implements the exact `bigint` overload, which
    is what the shared fixture records as `countBigInt` and what a5-py's own
    fixture test asserts. Python is right here; a5-rs needs the fix.

    Consequences are confined to `get_num_cells` and to `get_num_children` with
    a sub-Hilbert parent: `cell_area` divides by the count, and both integers
    round to the same f64, so areas are unaffected.
    """
    for resolution in (28, 29, 30):
        exact = py_cell_info.get_num_cells(resolution)
        lossy = native.get_num_cells(resolution)
        assert exact != lossy, (
            'a5-rs get_num_cells({}) now agrees with the exact value -- the '
            'upstream fix has landed, so drop this test and widen '
            'test_get_num_cells / test_get_num_children.'.format(resolution)
        )
        # The divergence is pure float rounding, not a different formula.
        assert float(exact) == float(lossy), resolution
