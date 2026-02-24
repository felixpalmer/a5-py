# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

import json
from pathlib import Path
from a5.traversal.quintant_neighbors import get_cell_neighbors


def load_fixtures():
    fixture_path = Path(__file__).parent / "fixtures" / "quintant-neighbors.json"
    with open(fixture_path, 'r') as f:
        return json.load(f)


class TestGetCellNeighbors:
    def test_quintant_neighbors_fixtures(self):
        fixtures = load_fixtures()
        for case in fixtures:
            inp = case["input"]
            expected = case["output"]["neighbors"]
            result = get_cell_neighbors(inp["s"], inp["resolution"], inp["orientation"])
            assert result == expected, \
                f's={inp["s"]}, res={inp["resolution"]}, orient={inp["orientation"]}: {result} != {expected}'
