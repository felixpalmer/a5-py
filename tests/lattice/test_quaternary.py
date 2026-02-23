# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

import json
from pathlib import Path
from a5.lattice import quaternary_to_kj, quaternary_to_flips, ij_to_quaternary


def load_fixtures():
    fixture_path = Path(__file__).parent / "fixtures" / "quaternary.json"
    with open(fixture_path, 'r') as f:
        return json.load(f)


class TestIJToQuaternary:
    def test_ij_to_quaternary_fixtures(self):
        fixtures = load_fixtures()
        for case in fixtures["IJToQuaternary"]:
            result = ij_to_quaternary(tuple(case["ij"]), tuple(case["flips"]))
            assert result == case["digit"], f'ij={case["ij"]}: {result} != {case["digit"]}'


class TestQuaternaryToKJ:
    def test_quaternary_to_kj_fixtures(self):
        fixtures = load_fixtures()
        for case in fixtures["quaternaryToKJ"]:
            result = quaternary_to_kj(case["q"], tuple(case["flips"]))
            assert list(result) == case["kj"], f'q={case["q"]}: {list(result)} != {case["kj"]}'


class TestQuaternaryToFlips:
    def test_quaternary_to_flips_fixtures(self):
        fixtures = load_fixtures()
        for case in fixtures["quaternaryToFlips"]:
            result = quaternary_to_flips(case["q"])
            assert list(result) == case["flips"], f'q={case["q"]}: {list(result)} != {case["flips"]}'
