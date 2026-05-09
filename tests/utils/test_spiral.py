# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

import json
from pathlib import Path

from a5.utils.spiral import Spiral, SPIRAL_SAMPLE_COUNT

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from matchers import is_close_array


def load_fixtures():
    fixture_path = Path(__file__).parent / "fixtures" / "spiral.json"
    with open(fixture_path, 'r') as f:
        return json.load(f)


class TestSpiral:
    def test_sample_count_matches_fixture(self):
        fixtures = load_fixtures()
        assert SPIRAL_SAMPLE_COUNT == fixtures["sampleCount"]

    def test_spiral_fixtures(self):
        fixtures = load_fixtures()
        for f in fixtures["spiral"]:
            center = tuple(f["center"])
            spiral = Spiral(center, f["scaleRad"])
            out = [0.0, 0.0, 0.0]
            assert f["sampleCount"] == SPIRAL_SAMPLE_COUNT
            for i in range(SPIRAL_SAMPLE_COUNT):
                spiral.sample(out, i)
                assert is_close_array(list(out), f["samples"][i], tolerance=6), \
                    f'sample {i} for {f["name"]}'
