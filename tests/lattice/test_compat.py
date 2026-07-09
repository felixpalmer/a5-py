# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

import json
from pathlib import Path

from a5.lattice import (
    Triple,
    compat_s_to_cell, compat_s_to_triple, compat_triple_to_s, compat_ij_to_s,
)


def load_fixtures():
    fixture_path = Path(__file__).parent / "fixtures" / "compat.json"
    with open(fixture_path, "r") as f:
        return json.load(f)


class TestCompat:
    """The compat curve reproduces the ORIGINAL (pre-L-system) A5 curve bit-for-bit."""

    def test_compat_s_to_cell(self):
        for f in load_fixtures()["sToCell"]:
            cell = compat_s_to_cell(f["s"], f["resolution"], f["orientation"])
            assert cell.triple.x == f["x"]
            assert cell.triple.y == f["y"]
            assert cell.triple.z == f["z"]
            assert cell.flavor == f["flavor"]

    def test_compat_s_to_triple(self):
        for f in load_fixtures()["sToCell"]:
            triple = compat_s_to_triple(f["s"], f["resolution"], f["orientation"])
            assert triple == Triple(f["x"], f["y"], f["z"])

    def test_compat_triple_to_s(self):
        for f in load_fixtures()["sToCell"]:
            triple = Triple(f["x"], f["y"], f["z"])
            s = compat_triple_to_s(triple, f["resolution"], f["orientation"])
            assert s == f["s"]

    def test_compat_ij_to_s(self):
        for f in load_fixtures()["IJToS"]:
            s = compat_ij_to_s((f["i"], f["j"]), f["resolution"], f["orientation"])
            assert s == f["s"]
