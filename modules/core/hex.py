"""
Hex conversion utilities for A5.
"""

def hex_to_big_int(hex_str: str) -> int:
    """
    Convert a hexadecimal string to a big integer.
    
    Args:
        hex_str (str): The hexadecimal string to convert
        
    Returns:
        int: The big integer value
    """
    return int(hex_str, 16)


def big_int_to_hex(index: int) -> str:
    """
    Convert a big integer to a hexadecimal string.
    
    Args:
        index (int): The big integer to convert
        
    Returns:
        str: The hexadecimal string representation
    """
    return hex(index)[2:]  # Remove '0x' prefix 