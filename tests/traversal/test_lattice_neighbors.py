# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

import json
from pathlib import Path

from a5.traversal.lattice_neighbors import get_lattice_neighbors
from a5.core.hex import hex_to_u64, u64_to_hex


def load_fixtures():
    fixture_path = Path(__file__).parent / "fixtures" / "lattice-neighbors.json"
    with open(fixture_path, 'r') as f:
        return json.load(f)


class TestGetLatticeNeighbors:
    def test_lattice_neighbors_fixtures(self):
        fixtures = load_fixtures()
        for f in fixtures["cases"]:
            cell = hex_to_u64(f["cell"])

            edge = sorted(u64_to_hex(c) for c in get_lattice_neighbors(cell, True))
            assert edge == f["edgeOnlyNeighbors"], \
                f'edge_only for cell {f["cell"]} (res {f["resolution"]})'

            superset = sorted(u64_to_hex(c) for c in get_lattice_neighbors(cell, False))
            assert superset == f["supersetNeighbors"], \
                f'superset for cell {f["cell"]} (res {f["resolution"]})'
