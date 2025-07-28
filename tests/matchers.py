"""
Utility functions for test files - Python equivalent of tests/utils/matchers.ts
"""
from typing import List, Union

# Default tolerance matching TypeScript version
DEFAULT_TOLERANCE = 13


def is_close_array(actual: Union[List, tuple], expected: List, tolerance: int = DEFAULT_TOLERANCE) -> bool:
    """
    Python equivalent of toBeCloseToArray matcher from TypeScript tests.
    Check if arrays are close within specified decimal places tolerance.
    """
    if len(actual) != len(expected):
        return False
    
    tolerance_value = 10**(-tolerance)
    return all(abs(a - e) < tolerance_value for a, e in zip(actual, expected))


def is_close(actual: float, expected: float, tolerance: int = DEFAULT_TOLERANCE) -> bool:
    """Helper function to check if values are close within tolerance"""
    tolerance_value = 10**(-tolerance)
    return abs(actual - expected) < tolerance_value