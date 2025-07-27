"""
A5 Math module - gl-matrix style vector and matrix operations
"""

from . import vec2
from . import vec3
from .vector_utils import (
    vector_difference, 
    triple_product, 
    quadruple_product, 
    slerp,
    dot_product,
    cross_product,
    vector_magnitude,
    vector_difference_basic
)

__all__ = [
    'vec2',
    'vec3', 
    'vector_difference',
    'triple_product',
    'quadruple_product',
    'slerp',
    'dot_product',
    'cross_product',
    'vector_magnitude',
    'vector_difference_basic'
]