# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

import json
from pathlib import Path
from a5.lattice import shift_digits, PATTERN, PATTERN_FLIPPED, PATTERN_REVERSED, PATTERN_FLIPPED_REVERSED


def load_fixtures():
    fixture_path = Path(__file__).parent / "fixtures" / "shift-digits.json"
    with open(fixture_path, 'r') as f:
        return json.load(f)


PATTERN_MAP = {
    "PATTERN": PATTERN,
    "PATTERN_FLIPPED": PATTERN_FLIPPED,
    "PATTERN_REVERSED": PATTERN_REVERSED,
    "PATTERN_FLIPPED_REVERSED": PATTERN_FLIPPED_REVERSED,
}


class TestShiftDigits:
    def test_shift_digits_fixtures(self):
        fixtures = load_fixtures()
        for case in fixtures["shiftDigits"]:
            digits = list(case["digitsBefore"])
            flips = list(case["flips"])
            pattern = PATTERN_MAP[case["patternName"]]
            shift_digits(digits, case["i"], flips, case["invertJ"], pattern)
            assert digits == case["digitsAfter"], \
                f'digits {case["digitsBefore"]} -> {digits} != {case["digitsAfter"]}'
