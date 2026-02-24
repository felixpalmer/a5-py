# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

import json
from pathlib import Path
from a5.traversal.grid_disk import grid_disk, grid_disk_vertex
from a5.core.serialization import get_resolution
from a5.core.compact import uncompact


def load_fixtures():
    fixture_path = Path(__file__).parent / "fixtures" / "grid-disk.json"
    with open(fixture_path, 'r') as f:
        return json.load(f)


def hex_to_int(hex_str: str) -> int:
    return int(hex_str, 16)


class TestGridDisk:
    def test_grid_disk_fixtures(self):
        fixtures = load_fixtures()
        for case in fixtures:
            cell_id = hex_to_int(case["cellId"])
            k = case["k"]
            target_res = get_resolution(cell_id)
            expected = sorted(hex_to_int(h) for h in case["cells"])
            result = sorted(uncompact(grid_disk(cell_id, k), target_res))
            assert result == expected, \
                f'cellId={case["cellId"]}, k={k}: got {len(result)} cells, expected {len(expected)}'


class TestGridDiskVertex:
    def test_grid_disk_vertex_fixtures(self):
        fixtures = load_fixtures()
        for case in fixtures:
            cell_id = hex_to_int(case["cellId"])
            k = case["k"]
            target_res = get_resolution(cell_id)
            # grid_disk_vertex returns edge + vertex cells
            extra = [hex_to_int(h) for h in case.get("extraVertexCells", [])]
            expected_edge = [hex_to_int(h) for h in case["cells"]]
            expected = sorted(set(expected_edge + extra))
            result = sorted(uncompact(grid_disk_vertex(cell_id, k), target_res))
            assert result == expected, \
                f'cellId={case["cellId"]}, k={k}: got {len(result)} cells, expected {len(expected)}'
