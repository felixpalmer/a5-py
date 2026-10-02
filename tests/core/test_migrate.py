# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

import json
import os

from a5.core.migrate import migrate
from a5.core.hex import hex_to_u64

# Load fixtures
fixtures_path = os.path.join(os.path.dirname(__file__), 'fixtures/migrate.json')
with open(fixtures_path, 'r') as f:
    migrate_fixtures = json.load(f)


class TestMigrate:
    def test_maps_v0_to_v1(self):
        """Test that v0 cell ids map to v1 cell ids."""
        for fixture in migrate_fixtures:
            assert migrate(hex_to_u64(fixture['v0'])) == hex_to_u64(fixture['v1'])
