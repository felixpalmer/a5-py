"""
Tests for hex conversion utilities.
"""

import pytest
from a5.core.hex import hex_to_bigint, bigint_to_hex


def test_hex_to_big_int():
    """Test hex to big int conversion."""
    assert hex_to_bigint("1a2b3c") == 1715004
    assert hex_to_bigint("0") == 0
    assert hex_to_bigint("ff") == 255
    assert hex_to_bigint("ffffffff") == 4294967295


def test_big_int_to_hex():
    """Test big int to hex conversion."""
    assert bigint_to_hex(1715004) == "1a2b3c"
    assert bigint_to_hex(0) == "0"
    assert bigint_to_hex(255) == "ff"
    assert bigint_to_hex(4294967295) == "ffffffff"


def test_round_trip():
    """Test that converting back and forth gives the same result."""
    test_values = ["1a2b3c", "0", "ff", "ffffffff"]
    for hex_str in test_values:
        big_int = hex_to_bigint(hex_str)
        result = bigint_to_hex(big_int)
        assert result == hex_str 