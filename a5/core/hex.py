# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

def hex_to_bigint(hex_str: str) -> int:
    """Convert hex string to big integer."""
    return int(hex_str, 16)

def bigint_to_hex(index: int) -> str:
    """Convert big integer to hex string."""
    return hex(index)[2:]  # Remove '0x' prefix 