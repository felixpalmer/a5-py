# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

import json
import os

from a5.collections.compact import compact, uncompact
from a5.collections.resolution import get_compaction_resolution
from a5.core.hex import hex_to_u64
from a5.core.serialization import get_resolution

# Load fixtures
fixtures_path = os.path.join(os.path.dirname(__file__), '../fixtures/compact.json')
with open(fixtures_path, 'r') as f:
    compact_fixtures = json.load(f)


class TestUncompact:
    def test_all_fixture_cases(self):
        """Test all uncompact fixture test cases."""
        for test_case in compact_fixtures['uncompact']:
            input_cells = [hex_to_u64(h) for h in test_case['input']]
            result = uncompact(input_cells)

            assert len(result) == test_case['expectedCount'], test_case['name']
            assert result == [hex_to_u64(h) for h in test_case['expectedCells']], test_case['name']
            assert get_compaction_resolution(input_cells) == test_case['expectedResolution'], test_case['name']

            # All results should be at the collection's resolution
            for cell in result:
                assert get_resolution(cell) == test_case['expectedResolution'], test_case['name']


class TestCompact:
    def test_all_fixture_cases(self):
        """Test all compact fixture test cases."""
        for test_case in compact_fixtures['compact']:
            input_cells = [hex_to_u64(h) for h in test_case['input']]
            expected = [hex_to_u64(h) for h in test_case['expectedOutput']]

            # Output is canonical: cells in curve order, then the compaction marker
            assert compact(input_cells) == expected, test_case['name']


class TestRoundTrip:
    def test_all_roundtrip_cases(self):
        """Test all round-trip fixture test cases."""
        for test_case in compact_fixtures['roundTrip']:
            initial_cells = [hex_to_u64(h) for h in test_case['initialCells']]
            after_compact = [hex_to_u64(h) for h in test_case['afterCompact']]

            # Verify compact result matches fixture
            assert compact(initial_cells) == after_compact, test_case['name']

            # Verify uncompact restores coverage, at the resolution the compaction marker records
            uncompact_result = uncompact(after_compact)
            assert len(uncompact_result) == test_case['expectedFinalCount'], test_case['name']
            for cell in uncompact_result:
                assert get_resolution(cell) == test_case['resolution'], test_case['name']
