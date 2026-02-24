# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

import json
from pathlib import Path
from a5.traversal.global_neighbors import get_global_cell_neighbors


def load_fixtures():
    fixture_path = Path(__file__).parent / "fixtures" / "global-neighbors.json"
    with open(fixture_path, 'r') as f:
        return json.load(f)


def hex_to_int(hex_str: str) -> int:
    return int(hex_str, 16)


class TestGetGlobalCellNeighbors:
    def test_global_neighbors_fixtures(self):
        fixtures = load_fixtures()
        for case in fixtures:
            cell_id = hex_to_int(case["input"]["cellId"])
            expected = sorted(hex_to_int(h) for h in case["output"]["neighbors"])
            result = get_global_cell_neighbors(cell_id)
            assert result == expected, \
                f'cellId={case["input"]["cellId"]}: got {[hex(c) for c in result]}, expected {[hex(c) for c in expected]}'

    def test_global_edge_neighbors_fixtures(self):
        fixtures = load_fixtures()
        for case in fixtures:
            cell_id = hex_to_int(case["input"]["cellId"])
            expected = sorted(hex_to_int(h) for h in case["output"]["edgeNeighbors"])
            result = get_global_cell_neighbors(cell_id, edge_only=True)
            assert result == expected, \
                f'cellId={case["input"]["cellId"]}: got {[hex(c) for c in result]}, expected {[hex(c) for c in expected]}'
