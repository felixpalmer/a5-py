# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

import json
from pathlib import Path

import pytest

from a5 import line_string_to_cells, u64_to_hex


def load_fixtures():
    fixture_path = Path(__file__).parent / "fixtures" / "line.json"
    with open(fixture_path, 'r') as f:
        return json.load(f)


class TestLineStringToCells:
    def test_line_segment_fixtures(self):
        fixtures = load_fixtures()
        for f in fixtures["lineSegment"]:
            start = (f["start"][0], f["start"][1])
            end = (f["end"][0], f["end"][1])
            result = line_string_to_cells([start, end], f["resolution"])
            sorted_cells = sorted(result)
            result_hex = [u64_to_hex(c) for c in sorted_cells]
            assert result_hex == f["cells"], f'fixture {f["name"]}'

    def test_empty_waypoints(self):
        assert line_string_to_cells([], 5) == []

    def test_single_waypoint(self):
        result = line_string_to_cells([(10.0, 50.0)], 5)
        assert len(result) == 1

    def test_deduplicate_at_junctions(self):
        waypoints = [(0.0, 50.0), (10.0, 50.0), (10.0, 45.0)]
        result = line_string_to_cells(waypoints, 3)
        assert len(result) == len(set(result))
