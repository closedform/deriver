"""
types - Custom Mathematical Types

Allows defining custom mathematical types with associated rules and operations.
Supports defining operations, methods, and custom display formatting.

Example:
    >>> Quaternion = DefineType('Quaternion', ['r', 'i', 'j', 'k'])
    >>> q1 = Quaternion(1, 2, 3, 4)
    >>> q2 = Quaternion(5, 6, 7, 8)
    >>> # Define addition rule
    >>> Quaternion.define_op('__add__', lambda a, b: Quaternion(
    ...     a.r + b.r, a.i + b.i, a.j + b.j, a.k + b.k))
"""

from derive.types.base import CustomType, DefineType
from derive.types.builtin import ComplexNumber, Quaternion, Vector3D

__all__ = [
    'CustomType',
    'DefineType',
    'ComplexNumber',
    'Quaternion',
    'Vector3D',
]
