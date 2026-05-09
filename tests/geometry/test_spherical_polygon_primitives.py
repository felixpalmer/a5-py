# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

import json
import math
from pathlib import Path

from a5.geometry.spherical_polygon import point_in_spherical_polygon, ring_winding_sign


def load_fixtures():
    fixture_path = Path(__file__).parent / "fixtures" / "spherical-polygon-primitives.json"
    with open(fixture_path, 'r') as f:
        return json.load(f)


DEG_TO_RAD = math.pi / 180


def ll_to_vec(ll):
    lat = ll[1] * DEG_TO_RAD
    lon = ll[0] * DEG_TO_RAD
    cos_lat = math.cos(lat)
    return (cos_lat * math.cos(lon), cos_lat * math.sin(lon), math.sin(lat))


class TestPointInSphericalPolygon:
    def test_pip_fixtures(self):
        fixtures = load_fixtures()
        for f in fixtures["pointInSphericalPolygon"]:
            ring_vecs = [ll_to_vec(p) for p in f["ring"]]
            for p in f["points"]:
                vec = tuple(p["vec"])
                result = point_in_spherical_polygon(vec, ring_vecs)
                assert result == p["inside"], \
                    f'fixture {f["name"]}, point {p["lonLat"]}'


class TestRingWindingSign:
    def test_ring_winding_sign_fixtures(self):
        fixtures = load_fixtures()
        for f in fixtures["ringWindingSign"]:
            ring_vecs = [ll_to_vec(p) for p in f["ring"]]
            assert ring_winding_sign(ring_vecs) == f["sign"], \
                f'fixture {f["name"]}'
