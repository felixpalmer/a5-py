# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

import json
import os

import pytest

from a5.coverings.compact import compact, uncompact
from a5.coverings.measures import area, count
from a5.coverings.set_operations import contains, difference, intersect, overlaps, union
from a5.coverings.resolution import covering_resolution
from a5.core.compaction_marker import is_compaction_marker
from a5.core.cell import cell_to_boundary
from a5.core.hex import hex_to_u64
from a5.core.serialization import is_valid_cell

fixtures_path = os.path.join(os.path.dirname(__file__), '../fixtures/covering.json')
with open(fixtures_path, 'r') as f:
    fixtures = json.load(f)


def to_cells(hexes):
    return [hex_to_u64(h) for h in hexes]


class TestSetOperations:
    @pytest.mark.parametrize('f', fixtures['setOperations'], ids=lambda f: f['name'])
    def test_fixtures(self, f):
        a = to_cells(f['a'])
        b = to_cells(f['b'])
        assert union(a, b) == to_cells(f['union'])
        assert intersect(a, b) == to_cells(f['intersect'])
        assert difference(a, b) == to_cells(f['difference'])
        assert overlaps(a, b) == f['overlaps']
        assert overlaps(b, a) == f['overlaps']
        assert covering_resolution(union(a, b)) == f['resolution']

    @pytest.mark.parametrize('f', fixtures['mismatchedResolutions'], ids=lambda f: f['name'])
    def test_refuse_mismatched_resolutions(self, f):
        a = to_cells(f['a'])
        b = to_cells(f['b'])
        for operation in (union, intersect, difference, overlaps):
            with pytest.raises(ValueError):
                operation(a, b)

    def test_input_out_of_curve_order(self):
        # Sorted input is merged as given; anything else is detected and sorted first
        for f in fixtures['setOperations']:
            a = to_cells(f['a'])[::-1]
            b = to_cells(f['b'])[::-1]
            assert union(a, b) == to_cells(f['union'])
            assert intersect(a, b) == to_cells(f['intersect'])
            assert difference(a, b) == to_cells(f['difference'])


class TestMeasures:
    @pytest.mark.parametrize('f', fixtures['measures'], ids=lambda f: f['name'])
    def test_fixtures(self, f):
        cells = to_cells(f['cells'])
        assert covering_resolution(cells) == f['resolution']
        assert count(cells) == int(f['count'])
        assert abs(area(cells) - f['area']) <= 1e-10 * f['area']

    def test_overlapping_input(self):
        # Every cell given is counted, including duplicates; union merges them
        for f in fixtures['measures']:
            cells = to_cells(f['cells'])
            doubled = cells + cells
            assert count(doubled) == 2 * int(f['count'])
            assert abs(area(doubled) - 2 * f['area']) <= 1e-10 * f['area']
            assert count(union(cells, cells)) == int(f['count'])


class TestContains:
    @pytest.mark.parametrize('f', fixtures['contains'], ids=lambda f: f['name'])
    def test_fixtures(self, f):
        cells = to_cells(f['cells'])
        for probe in f['probes']:
            assert contains(cells, hex_to_u64(probe['cell'])) == probe['expected']

    @pytest.mark.parametrize('f', fixtures['mismatchedProbes'], ids=lambda f: f['name'])
    def test_refuse_cells_at_another_resolution(self, f):
        cells = to_cells(f['cells'])
        for probe in f['probes']:
            with pytest.raises(ValueError):
                contains(cells, hex_to_u64(probe))


class TestIsCompactionMarker:
    def test_recognizes_compaction_markers_only(self):
        for f in fixtures['isCompactionMarker']:
            assert is_compaction_marker(hex_to_u64(f['value'])) == f['expected'], f['value']

    def test_records_resolution(self):
        for f in fixtures['isCompactionMarker']:
            if f['expected']:
                assert covering_resolution([hex_to_u64(f['value'])]) == f['resolution']

    def test_empty_boundary(self):
        for f in fixtures['isCompactionMarker']:
            if f['expected']:
                assert cell_to_boundary(hex_to_u64(f['value'])) == []


class TestIsValidCell:
    def test_recognizes_cells_only(self):
        for f in fixtures['isValidCell']:
            assert is_valid_cell(hex_to_u64(f['value'])) == f['expected'], f['value']

    def test_refuses_values_outside_64_bits(self):
        assert not is_valid_cell(-1)
        assert not is_valid_cell(-(1 << 3))
        assert not is_valid_cell(1 << 64)
        assert not is_valid_cell((1 << 64) | 1)


INVALID_CELL_OPERATIONS = {
    'compact': lambda cells: compact(cells),
    'uncompact': lambda cells: uncompact(cells),
    'count': lambda cells: count(cells),
    'area': lambda cells: area(cells),
    'union': lambda cells: union(cells, cells),
    'contains': lambda cells: contains(cells, cells[0]),
}


class TestInvalidCells:
    @pytest.mark.parametrize('name', INVALID_CELL_OPERATIONS)
    def test_refuse_values_that_are_not_cells(self, name):
        operation = INVALID_CELL_OPERATIONS[name]
        for value in fixtures['invalidCells']:
            with pytest.raises(ValueError):
                operation([hex_to_u64(value)])

    @pytest.mark.parametrize('name', INVALID_CELL_OPERATIONS)
    def test_accept_valid_cells_at_the_edges_of_the_encoding(self, name):
        operation = INVALID_CELL_OPERATIONS[name]
        for value in fixtures['validEdgeCells']:
            operation([hex_to_u64(value)])
