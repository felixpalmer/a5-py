# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

import json
from pathlib import Path

from a5.lattice import (
    Triple,
    s_to_cell, s_to_triple, triple_to_s, triple_parity, triple_in_bounds, ij_to_s,
)


def load_fixtures():
    fixture_path = Path(__file__).parent / "fixtures" / "curve.json"
    with open(fixture_path, "r") as f:
        return json.load(f)


class TestCurve:
    """The canonical (engine) curve, via the top-level lattice API."""

    def test_s_to_cell(self):
        for f in load_fixtures()["sToCell"]:
            cell = s_to_cell(f["s"], f["resolution"], f["orientation"])
            assert cell.triple.x == f["x"]
            assert cell.triple.y == f["y"]
            assert cell.triple.z == f["z"]
            assert cell.flavor == f["flavor"]

    def test_s_to_triple(self):
        for f in load_fixtures()["sToCell"]:
            triple = s_to_triple(f["s"], f["resolution"], f["orientation"])
            assert triple == Triple(f["x"], f["y"], f["z"])

    def test_triple_parity(self):
        for f in load_fixtures()["sToCell"]:
            triple = Triple(f["x"], f["y"], f["z"])
            assert triple_parity(triple) == f["parity"]

    def test_triple_to_s(self):
        for f in load_fixtures()["sToCell"]:
            triple = Triple(f["x"], f["y"], f["z"])
            s = triple_to_s(triple, f["resolution"], f["orientation"])
            assert s == f["s"]

    def test_ij_to_s(self):
        for f in load_fixtures()["IJToS"]:
            s = ij_to_s((f["i"], f["j"]), f["resolution"], f["orientation"])
            assert s == f["s"]

    def test_triple_in_bounds(self):
        for f in load_fixtures()["tripleInBounds"]:
            triple = Triple(f["x"], f["y"], f["z"])
            assert triple_in_bounds(triple, f["maxRow"]) == f["expected"]
