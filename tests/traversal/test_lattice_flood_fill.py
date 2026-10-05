# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

import json
from pathlib import Path

from a5.traversal.lattice_flood_fill import triple_space_flood_fill
from a5.traversal.triple_cells import cell_ids_to_triples, triple_cells_to_ids
from a5.core.serialization import FIRST_HILBERT_RESOLUTION
from a5.core.hex import hex_to_u64, u64_to_hex


def load_fixtures():
    fixture_path = Path(__file__).parent / "fixtures" / "lattice-flood-fill.json"
    with open(fixture_path, 'r') as f:
        return json.load(f)


class TestTripleSpaceFloodFill:
    def test_lattice_flood_fill_fixtures(self):
        fixtures = load_fixtures()
        for f in fixtures["cases"]:
            seeds = cell_ids_to_triples(hex_to_u64(c) for c in f["seedCells"])
            firewall = cell_ids_to_triples(hex_to_u64(c) for c in f["firewallCells"])

            result = triple_space_flood_fill(firewall, seeds, f["resolution"], f.get("maxLayers"))
            hilbert_res = f["resolution"] - FIRST_HILBERT_RESOLUTION + 1
            interior = sorted(u64_to_hex(c) for c in triple_cells_to_ids(result['interior'], hilbert_res, f["resolution"]))
            frontier = sorted(u64_to_hex(c) for c in triple_cells_to_ids(result['frontier'], hilbert_res, f["resolution"]))

            assert interior == f["interiorCells"], f'interior for {f["name"]}'
            assert frontier == f["frontierCells"], f'frontier for {f["name"]}'
