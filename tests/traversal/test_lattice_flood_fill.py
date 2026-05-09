# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

import json
from pathlib import Path

from a5.traversal.lattice_flood_fill import triple_space_flood_fill
from a5.core.hex import hex_to_u64, u64_to_hex


def load_fixtures():
    fixture_path = Path(__file__).parent / "fixtures" / "lattice-flood-fill.json"
    with open(fixture_path, 'r') as f:
        return json.load(f)


class TestTripleSpaceFloodFill:
    def test_lattice_flood_fill_fixtures(self):
        fixtures = load_fixtures()
        for f in fixtures["cases"]:
            seeds = [hex_to_u64(c) for c in f["seedCells"]]
            firewall = set(hex_to_u64(c) for c in f["firewallCells"])

            result = triple_space_flood_fill(firewall, seeds, f["resolution"], f.get("maxLayers"))
            interior = sorted(u64_to_hex(c) for c in result['interior_cells'])
            frontier = sorted(u64_to_hex(c) for c in result['frontier_cell_ids'])

            assert interior == f["interiorCells"], f'interior for {f["name"]}'
            assert frontier == f["frontierCells"], f'frontier for {f["name"]}'
