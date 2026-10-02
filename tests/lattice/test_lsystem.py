# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

import json
from pathlib import Path

from a5.lattice import Triple
from a5.lattice.lsystem import s_to_cell, s_to_triple, triple_to_s_lattice
from a5.lattice.curve import ij_to_s


def load_fixtures():
    fixture_path = Path(__file__).parent / "fixtures" / "lsystem.json"
    with open(fixture_path, "r") as f:
        return json.load(f)


class TestLSystem:
    """The non-self-intersecting L-system curve (planned FUTURE canonical curve)."""

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

    def test_triple_to_s_lattice(self):
        for f in load_fixtures()["sToCell"]:
            triple = Triple(f["x"], f["y"], f["z"])
            s = triple_to_s_lattice(triple, f["resolution"], f["orientation"])
            assert s == f["s"]

    def test_ij_to_s(self):
        for f in load_fixtures()["IJToS"]:
            s = ij_to_s((f["i"], f["j"]), f["resolution"], f["orientation"])
            assert s == f["s"]
