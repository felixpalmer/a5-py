# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

import json
from pathlib import Path
from a5.lattice import s_to_anchor, anchor_to_s


def load_fixtures():
    fixture_path = Path(__file__).parent / "fixtures" / "hilbert.json"
    with open(fixture_path, 'r') as f:
        return json.load(f)


class TestSToAnchor:
    def test_s_to_anchor_fixtures(self):
        fixtures = load_fixtures()
        for case in fixtures["sToAnchor"]:
            anchor = s_to_anchor(case["s"], case["resolution"], case["orientation"])
            assert anchor.q == case["q"], f's={case["s"]}: q {anchor.q} != {case["q"]}'
            assert list(anchor.offset) == case["offset"], f's={case["s"]}: offset {list(anchor.offset)} != {case["offset"]}'
            assert list(anchor.flips) == case["flips"], f's={case["s"]}: flips {list(anchor.flips)} != {case["flips"]}'


class TestAnchorToS:
    def test_anchor_to_s_roundtrip(self):
        fixtures = load_fixtures()
        for case in fixtures["sToAnchor"]:
            anchor = s_to_anchor(case["s"], case["resolution"], case["orientation"])
            s_back = anchor_to_s(anchor, case["resolution"], case["orientation"])
            assert s_back == case["s"], f'roundtrip failed: {s_back} != {case["s"]}'
