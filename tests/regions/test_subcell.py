# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

import json
from pathlib import Path

import pytest

from a5 import (
    cell_to_subcell, cell_to_supercell, count, covering_resolution, get_resolution, hex_to_u64, u64_to_hex,
    uncompact,
)


def load_fixtures():
    fixture_path = Path(__file__).parent / "fixtures" / "subcell.json"
    with open(fixture_path, 'r') as f:
        return json.load(f)


FIXTURES = load_fixtures()
SUBCELL = FIXTURES["subcell"]
SUPERCELL = FIXTURES["supercell"]


class TestCellToSubcell:
    @pytest.mark.parametrize("f", SUBCELL, ids=lambda f: f'{f["cell"]}-res{f["resolution"]}')
    def test_subcell_fixtures(self, f):
        result = cell_to_subcell(hex_to_u64(f["cell"]), f["resolution"])
        assert [u64_to_hex(c) for c in result] == f["cells"]
        assert count(result) == f["count"]

    def test_maps_every_subcell_back_to_its_cell(self):
        for f in SUBCELL:
            cell = hex_to_u64(f["cell"])
            cell_resolution = get_resolution(cell)
            for subcell in uncompact(cell_to_subcell(cell, f["resolution"])):
                assert cell_to_supercell(subcell, cell_resolution) == cell

    def test_own_resolution_returns_cell(self):
        cell = hex_to_u64(SUBCELL[5]["cell"])
        result = cell_to_subcell(cell, get_resolution(cell))
        assert list(uncompact(result)) == [cell]
        assert covering_resolution(result) == get_resolution(cell)

    def test_coarser_resolution_raises(self):
        cell = hex_to_u64(SUBCELL[5]["cell"])
        with pytest.raises(ValueError):
            cell_to_subcell(cell, get_resolution(cell) - 1)


class TestCellToSupercell:
    def test_supercell_fixtures(self):
        for f in SUPERCELL:
            assert u64_to_hex(cell_to_supercell(hex_to_u64(f["cell"]), f["resolution"])) == f["supercell"]

    def test_finer_resolution_raises(self):
        cell = hex_to_u64(SUPERCELL[0]["cell"])
        with pytest.raises(ValueError):
            cell_to_supercell(cell, get_resolution(cell) + 1)
