# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

def hex_to_u64(hex_str: str) -> int:
    """Convert hex string to u64."""
    return int(hex_str, 16)

def u64_to_hex(value: int) -> str:
    """Convert u64 to hex string."""
    return hex(value)[2:]  # Remove '0x' prefix 