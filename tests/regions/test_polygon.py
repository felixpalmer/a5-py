# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

import json
from pathlib import Path

from a5 import polygon_to_cells, uncompact, u64_to_hex


def load_fixtures():
    fixture_path = Path(__file__).parent / "fixtures" / "polygon.json"
    with open(fixture_path, 'r') as f:
        return json.load(f)


class TestPolygonToCells:
    def test_polygon_fixtures(self):
        fixtures = load_fixtures()
        for f in fixtures["polygon"]:
            ring = [(p[0], p[1]) for p in f["ring"]]
            result = polygon_to_cells(ring, f["resolution"])
            expanded = uncompact(result, f["resolution"])
            sorted_cells = sorted(expanded)
            result_hex = [u64_to_hex(c) for c in sorted_cells]
            assert result_hex == f["cells"], f'fixture {f["name"]}'

    def test_returns_empty_for_less_than_3_vertices(self):
        assert polygon_to_cells([], 5) == []
        assert polygon_to_cells([(0.0, 0.0), (1.0, 1.0)], 5) == []

    def test_country_fixtures(self):
        fixtures = load_fixtures()
        country_cases = fixtures.get("country", [])
        for f in country_cases:
            ring = [(p[0], p[1]) for p in f["ring"]]
            result = polygon_to_cells(ring, f["resolution"])
            expanded = uncompact(result, f["resolution"])
            unique = set(expanded)
            assert len(unique) == f["cellCount"], f'fixture {f["name"]}'
