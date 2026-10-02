# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

import json
from pathlib import Path

from a5.utils.great_circle import sample_great_circle_arc, great_circle_distance

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from matchers import is_close, is_close_array


def load_fixtures():
    fixture_path = Path(__file__).parent / "fixtures" / "great-circle.json"
    with open(fixture_path, 'r') as f:
        return json.load(f)


class TestGreatCircleDistancePrecision:
    def test_near_degenerate_distances(self):
        """
        Near-degenerate cases with analytically-known distances (Kahan section 12):
        acos(a.b) would fail these -- it returns 0 for separations below ~1e-8 rad
        and errs by ~0.1 m near the antipode. The stable 2*atan2 form must match
        the analytic value to within 1e-10 relative (or 1e-9 m absolute near zero).
        """
        fixtures = load_fixtures()
        for f in fixtures["distances"]:
            d = great_circle_distance(tuple(f["aVec"]), tuple(f["bVec"]))
            tolerance = max(1e-9, 1e-10 * f["distance"])
            assert abs(d - f["distance"]) <= tolerance, f'distance for {f["name"]}'


class TestSampleGreatCircleArc:
    def test_sample_great_circle_arc_fixtures(self):
        fixtures = load_fixtures()
        for f in fixtures["sampleGreatCircleArc"]:
            a_vec = tuple(f["aVec"])
            b_vec = tuple(f["bVec"])

            # Distance check is the cheapest signal that endpoints round-tripped.
            distance = great_circle_distance(a_vec, b_vec)
            assert abs(distance - f["distance"]) < 1e-6, f'distance for {f["name"]}'

            samples = sample_great_circle_arc(a_vec, b_vec, f["sampleInterval"])
            assert len(samples) == f["sampleCount"], f'sample count for {f["name"]}'
            for i in range(len(samples)):
                assert is_close_array(list(samples[i]), f["samples"][i], tolerance=6), \
                    f'sample {i} for {f["name"]}'
