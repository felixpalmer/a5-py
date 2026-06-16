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


def to_rings(polygon):
    return [[(p[0], p[1]) for p in ring] for ring in polygon]


class TestPolygonToCells:
    def test_polygon_fixtures(self):
        fixtures = load_fixtures()
        for f in fixtures["polygon"]:
            rings = to_rings(f["polygon"])
            result = polygon_to_cells(rings, f["resolution"])
            expanded = uncompact(result, f["resolution"])
            sorted_cells = sorted(expanded)
            result_hex = [u64_to_hex(c) for c in sorted_cells]
            assert result_hex == f["cells"], f'fixture {f["name"]}'

    def test_returns_empty_for_less_than_3_vertices(self):
        assert polygon_to_cells([], 5) == []
        assert polygon_to_cells([(0.0, 0.0), (1.0, 1.0)], 5) == []
        # Nested form with a degenerate outer ring
        assert polygon_to_cells([[(0.0, 0.0), (1.0, 1.0)]], 5) == []
        # Closed ring with only 2 distinct vertices
        assert polygon_to_cells([(0.0, 0.0), (1.0, 1.0), (0.0, 0.0)], 5) == []

    def test_accepts_geojson_style_closed_rings(self):
        ring = [(-5.0, 54.0), (15.0, 54.0), (15.0, 44.0), (-5.0, 44.0)]
        hole = [(2.0, 51.0), (8.0, 51.0), (8.0, 47.0), (2.0, 47.0)]

        def closed(r):
            return r + [r[0]]

        assert polygon_to_cells([closed(ring), closed(hole)], 6) == polygon_to_cells([ring, hole], 6)

    def test_treats_flat_ring_as_polygon_without_holes(self):
        ring = [(-5.0, 54.0), (15.0, 54.0), (15.0, 44.0), (-5.0, 44.0)]
        assert polygon_to_cells(ring, 5) == polygon_to_cells([ring], 5)

    def test_ignores_degenerate_holes(self):
        ring = [(-5.0, 54.0), (15.0, 54.0), (15.0, 44.0), (-5.0, 44.0)]
        degenerate_hole = [(2.0, 50.0), (3.0, 49.0)]
        assert polygon_to_cells([ring, degenerate_hole], 5) == polygon_to_cells([ring], 5)

    def test_country_fixtures(self):
        fixtures = load_fixtures()
        country_cases = fixtures.get("country", [])
        for f in country_cases:
            rings = to_rings(f["polygon"])
            result = polygon_to_cells(rings, f["resolution"])
            expanded = uncompact(result, f["resolution"])
            unique = set(expanded)
            assert len(unique) == f["cellCount"], f'fixture {f["name"]}'
