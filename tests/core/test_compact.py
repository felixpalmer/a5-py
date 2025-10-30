# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

import json
import os
import pytest

from a5.core.compact import compact, uncompact
from a5.core.hex import hex_to_u64
from a5.core.serialization import deserialize

# Load fixtures
fixtures_path = os.path.join(os.path.dirname(__file__), '../fixtures/compact.json')
with open(fixtures_path, 'r') as f:
    compact_fixtures = json.load(f)


class TestUncompact:
    def test_all_fixture_cases(self):
        """Test all uncompact fixture test cases."""
        for test_case in compact_fixtures['uncompact']:
            # Skip error test cases - handle separately
            if test_case.get('expectedError'):
                continue

            input_cells = [hex_to_u64(h) for h in test_case['input']]
            result = uncompact(input_cells, test_case['targetResolution'])

            assert len(result) == test_case['expectedCount'], \
                f"Failed test case: {test_case['name']}"

            # All results should be at target resolution
            for cell in result:
                cell_data = deserialize(cell)
                assert cell_data['resolution'] == test_case['targetResolution'], \
                    f"Failed test case: {test_case['name']}"

    def test_error_on_lower_resolution(self):
        """Test that uncompact throws error when trying to uncompact to lower resolution."""
        error_cases = [tc for tc in compact_fixtures['uncompact'] if tc.get('expectedError')]
        if error_cases:
            error_case = error_cases[0]
            input_cells = [hex_to_u64(h) for h in error_case['input']]
            with pytest.raises(ValueError):
                uncompact(input_cells, error_case['targetResolution'])


class TestCompact:
    def test_all_fixture_cases(self):
        """Test all compact fixture test cases."""
        for test_case in compact_fixtures['compact']:
            input_cells = [hex_to_u64(h) for h in test_case['input']]
            expected = sorted([hex_to_u64(h) for h in test_case['expectedOutput']])
            result = compact(input_cells)

            assert sorted(result) == expected, \
                f"Failed test case: {test_case['name']}"


class TestRoundTrip:
    def test_all_roundtrip_cases(self):
        """Test all round-trip fixture test cases."""
        for test_case in compact_fixtures['roundTrip']:
            initial_cells = [hex_to_u64(h) for h in test_case['initialCells']]
            after_compact_expected = sorted([hex_to_u64(h) for h in test_case['afterCompact']])

            # Verify compact result matches fixture
            compact_result = compact(initial_cells)
            assert sorted(compact_result) == after_compact_expected, \
                f"Failed compact in test case: {test_case['name']}"

            # Verify uncompact restores coverage
            uncompact_result = uncompact(after_compact_expected, test_case['targetResolution'])

            if 'expectedCount' in test_case:
                assert len(uncompact_result) == test_case['expectedCount'], \
                    f"Failed uncompact count in test case: {test_case['name']}"

            if 'expectedFinalCount' in test_case:
                assert len(uncompact_result) == test_case['expectedFinalCount'], \
                    f"Failed final count in test case: {test_case['name']}"

            # All results should be at target resolution
            for cell in uncompact_result:
                cell_data = deserialize(cell)
                assert cell_data['resolution'] == test_case['targetResolution'], \
                    f"Failed resolution check in test case: {test_case['name']}"
