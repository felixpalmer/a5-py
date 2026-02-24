# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

import json
from pathlib import Path
from a5.traversal.cap import meters_to_h, estimate_cell_radius, pick_coarse_resolution, spherical_cap
from a5.core.serialization import get_resolution
from a5.core.compact import uncompact


def load_fixtures():
    fixture_path = Path(__file__).parent / "fixtures" / "cap.json"
    with open(fixture_path, 'r') as f:
        return json.load(f)


def hex_to_int(hex_str: str) -> int:
    return int(hex_str, 16)


class TestMetersToH:
    def test_meters_to_h_fixtures(self):
        fixtures = load_fixtures()
        for case in fixtures["helpers"]["metersToH"]:
            result = meters_to_h(case["meters"])
            # Allow 1 ULP tolerance: Python's math.sin may round differently from JS Math.sin
            assert abs(result - case["expectedH"]) < 1e-15, \
                f'meters={case["meters"]}: {result} != {case["expectedH"]}'


class TestEstimateCellRadius:
    def test_estimate_cell_radius_fixtures(self):
        fixtures = load_fixtures()
        for case in fixtures["helpers"]["estimateCellRadius"]:
            result = estimate_cell_radius(case["resolution"])
            assert result == case["expectedMeters"], \
                f'res={case["resolution"]}: {result} != {case["expectedMeters"]}'

    def test_decreases_with_resolution(self):
        fixtures = load_fixtures()
        cases = fixtures["helpers"]["estimateCellRadius"]
        for i in range(1, len(cases)):
            assert cases[i]["expectedMeters"] < cases[i - 1]["expectedMeters"]


class TestPickCoarseResolution:
    def test_pick_coarse_resolution_fixtures(self):
        fixtures = load_fixtures()
        for case in fixtures["helpers"]["pickCoarseResolution"]:
            result = pick_coarse_resolution(case["radius"], case["targetRes"])
            assert result == case["expectedCoarseRes"], \
                f'radius={case["radius"]}, targetRes={case["targetRes"]}: {result} != {case["expectedCoarseRes"]}'

    def test_never_exceeds_target_res(self):
        fixtures = load_fixtures()
        for case in fixtures["helpers"]["pickCoarseResolution"]:
            result = pick_coarse_resolution(case["radius"], case["targetRes"])
            assert result <= case["targetRes"]


class TestSphericalCap:
    def test_spherical_cap_fixtures(self):
        fixtures = load_fixtures()
        for case in fixtures["sphericalCap"]:
            cell_id = hex_to_int(case["cellId"])
            radius = case["radius"]
            target_res = get_resolution(cell_id)
            expected = sorted(hex_to_int(h) for h in case["cells"])
            result = sorted(uncompact(spherical_cap(cell_id, radius), target_res))
            assert result == expected, \
                f'cellId={case["cellId"]}, radius={radius}: got {len(result)} cells, expected {len(expected)}'

    def test_spherical_cap_compact_fixtures(self):
        fixtures = load_fixtures()
        for case in fixtures["sphericalCapCompact"]:
            cell_id = hex_to_int(case["cellId"])
            radius = case["radius"]
            expected = sorted(hex_to_int(h) for h in case["compactedCells"])
            result = sorted(spherical_cap(cell_id, radius))
            assert result == expected, \
                f'cellId={case["cellId"]}, radius={radius}: got {len(result)} cells, expected {len(expected)}'
