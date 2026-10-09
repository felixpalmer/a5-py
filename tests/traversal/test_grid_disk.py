# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

import json
from pathlib import Path
from a5.traversal.grid_disk import grid_disk, grid_disk_vertex
from a5.coverings.compact import uncompact
from a5.core.compaction_marker import is_compaction_marker


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
            expected = sorted(hex_to_int(h) for h in case["cells"])
            result = sorted(uncompact(grid_disk(cell_id, k)))
            assert result == expected, \
                f'cellId={case["cellId"]}, k={k}: got {len(result)} cells, expected {len(expected)}'


    def test_k0_returns_only_center_cell(self):
        cell_id = hex_to_int(load_fixtures()[0]["cellId"])
        result = grid_disk(cell_id, 0)
        # The cell itself, then the compaction marker recording its resolution
        assert len(result) == 2
        assert result[0] == cell_id
        assert is_compaction_marker(result[1])
        assert uncompact(result) == [cell_id]


class TestGridDiskVertex:
    def test_grid_disk_vertex_fixtures(self):
        fixtures = load_fixtures()
        for case in fixtures:
            cell_id = hex_to_int(case["cellId"])
            k = case["k"]
            # grid_disk_vertex returns edge + vertex cells
            extra = [hex_to_int(h) for h in case.get("extraVertexCells", [])]
            expected_edge = [hex_to_int(h) for h in case["cells"]]
            expected = sorted(set(expected_edge + extra))
            result = sorted(uncompact(grid_disk_vertex(cell_id, k)))
            assert result == expected, \
                f'cellId={case["cellId"]}, k={k}: got {len(result)} cells, expected {len(expected)}'

    def test_k0_returns_only_center_cell(self):
        cell_id = hex_to_int(load_fixtures()[0]["cellId"])
        result = grid_disk_vertex(cell_id, 0)
        # The cell itself, then the compaction marker recording its resolution
        assert len(result) == 2
        assert result[0] == cell_id
        assert is_compaction_marker(result[1])
        assert uncompact(result) == [cell_id]
