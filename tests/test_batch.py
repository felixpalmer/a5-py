# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

"""Batch operations must be indistinguishable from looping the scalar ones.

Runs against whichever backend is selected, so the suite covers both the
compiled implementations and the pure-Python comprehensions.
"""

import pytest

import a5
from a5 import batch

CELLS = [a5.lonlat_to_cell(point, 12) for point in [
    (-0.1276, 51.5072),
    (2.3522, 48.8566),
    (139.6917, 35.6895),
    (-74.0060, 40.7128),
    (151.2093, -33.8688),
    (18.4241, -33.9249),
    (-43.1729, -22.9068),
    (0.0, 0.0),
    (179.9, 66.5),
    (-179.9, -66.5),
]]


class TestParity:
    def test_get_resolution(self):
        assert batch.get_resolution(CELLS) == [a5.get_resolution(c) for c in CELLS]

    def test_cell_to_parent_default(self):
        assert batch.cell_to_parent(CELLS) == [a5.cell_to_parent(c) for c in CELLS]

    @pytest.mark.parametrize('parent_resolution', [-1, 0, 1, 5, 11, 12])
    def test_cell_to_parent_explicit(self, parent_resolution):
        assert batch.cell_to_parent(CELLS, parent_resolution) == [
            a5.cell_to_parent(c, parent_resolution) for c in CELLS
        ]

    def test_cell_to_children_default(self):
        assert batch.cell_to_children(CELLS) == [a5.cell_to_children(c) for c in CELLS]

    @pytest.mark.parametrize('child_resolution', [12, 13, 14])
    def test_cell_to_children_explicit(self, child_resolution):
        assert batch.cell_to_children(CELLS, child_resolution) == [
            a5.cell_to_children(c, child_resolution) for c in CELLS
        ]

    def test_cell_area(self):
        resolutions = list(range(-1, 31))
        assert batch.cell_area(resolutions) == [a5.cell_area(r) for r in resolutions]


class TestEdgeCases:
    def test_empty_input(self):
        assert batch.get_resolution([]) == []
        assert batch.cell_to_parent([]) == []
        assert batch.cell_to_children([]) == []
        assert batch.cell_area([]) == []

    def test_accepts_tuples_as_well_as_lists(self):
        assert batch.get_resolution(tuple(CELLS)) == batch.get_resolution(CELLS)

    def test_keyword_arguments(self):
        assert batch.cell_to_parent(CELLS, parent_resolution=5) == batch.cell_to_parent(CELLS, 5)
        assert batch.cell_to_children(CELLS, child_resolution=13) == batch.cell_to_children(
            CELLS, 13
        )

    def test_error_matches_the_scalar_operation(self):
        # A failing element must raise, not be silently skipped or defaulted.
        max_res_cells = [a5.lonlat_to_cell((1.0, 51.0), 30)]
        with pytest.raises(ValueError) as scalar_exc:
            a5.cell_to_children(max_res_cells[0])
        with pytest.raises(ValueError) as batch_exc:
            batch.cell_to_children(max_res_cells)
        assert str(batch_exc.value) == str(scalar_exc.value)

    def test_order_is_preserved(self):
        reversed_cells = list(reversed(CELLS))
        assert batch.get_resolution(reversed_cells) == list(
            reversed(batch.get_resolution(CELLS))
        )
