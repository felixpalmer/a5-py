# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

import json
from pathlib import Path

from a5.traversal.triple_cells import triple_children, triple_parent


def load_fixtures():
    fixture_path = Path(__file__).parent / "fixtures" / "triple-hierarchy.json"
    with open(fixture_path, 'r') as f:
        return json.load(f)


class TestTripleHierarchy:
    def test_triple_children(self):
        for f in load_fixtures()["cases"]:
            out = []
            triple_children(0, 0, *f["triple"], (1 << f["hilbertRes"]) - 1, out)
            children = sorted(out[c + 2:c + 5] for c in range(0, 20, 5))
            assert children == f["children"], f'{f["triple"]} @ {f["hilbertRes"]}'

    def test_triple_parent(self):
        for f in load_fixtures()["cases"]:
            for child in f["children"]:
                out = []
                triple_parent(0, 0, *child, (1 << f["hilbertRes"]) - 1, out)
                assert out[2:] == f["triple"], f'{child} @ {f["hilbertRes"] + 1}'
