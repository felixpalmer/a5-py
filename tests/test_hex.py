"""
Tests for hex conversion utilities.
"""

import pytest
from a5.core.hex import hex_to_u64, u64_to_hex


def test_hex_to_u64():
    """Test hex to u64 conversion."""
    assert hex_to_u64("1a2b3c") == 1715004
    assert hex_to_u64("0") == 0
    assert hex_to_u64("ff") == 255
    assert hex_to_u64("ffffffff") == 4294967295


def test_u64_to_hex():
    """Test u64 to hex conversion."""
    assert u64_to_hex(1715004) == "1a2b3c"
    assert u64_to_hex(0) == "0"
    assert u64_to_hex(255) == "ff"
    assert u64_to_hex(4294967295) == "ffffffff"


def test_round_trip():
    """Test that converting back and forth gives the same result."""
    test_values = ["1a2b3c", "0", "ff", "ffffffff"]
    for hex_str in test_values:
        u64_value = hex_to_u64(hex_str)
        result = u64_to_hex(u64_value)
        assert result == hex_str 