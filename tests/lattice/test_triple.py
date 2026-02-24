# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

import json
from pathlib import Path
from a5.lattice import (
    s_to_anchor, anchor_to_triple, triple_parity,
    triple_to_s, triple_in_bounds, triple_to_anchor, Triple
)


def load_fixtures():
    fixture_path = Path(__file__).parent / "fixtures" / "triple.json"
    with open(fixture_path, 'r') as f:
        return json.load(f)


class TestAnchorToTriple:
    def test_anchor_to_triple_fixtures(self):
        fixtures = load_fixtures()
        for case in fixtures["anchorToTriple"]:
            anchor = s_to_anchor(case["s"], case["resolution"], case["orientation"])
            triple = anchor_to_triple(anchor)
            assert triple.x == case["x"], f's={case["s"]}: x {triple.x} != {case["x"]}'
            assert triple.y == case["y"], f's={case["s"]}: y {triple.y} != {case["y"]}'
            assert triple.z == case["z"], f's={case["s"]}: z {triple.z} != {case["z"]}'
            assert triple_parity(triple) == case["parity"], \
                f's={case["s"]}: parity {triple_parity(triple)} != {case["parity"]}'


class TestTripleToS:
    def test_triple_to_s_roundtrip(self):
        fixtures = load_fixtures()
        for case in fixtures["anchorToTriple"]:
            triple = Triple(case["x"], case["y"], case["z"])
            s = triple_to_s(triple, case["resolution"], case["orientation"])
            assert s == case["s"], f'triple ({case["x"]},{case["y"]},{case["z"]}): s {s} != {case["s"]}'


class TestTripleInBounds:
    def test_triple_in_bounds_fixtures(self):
        fixtures = load_fixtures()
        if "tripleInBounds" in fixtures:
            for case in fixtures["tripleInBounds"]:
                triple = Triple(case["x"], case["y"], case["z"])
                result = triple_in_bounds(triple, case["maxRow"])
                assert result == case["expected"], \
                    f'triple ({case["x"]},{case["y"]},{case["z"]}): {result} != {case["expected"]}'
