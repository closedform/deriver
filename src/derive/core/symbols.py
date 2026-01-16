"""
symbols.py - Symbol and Function definitions.

Provides Symbol, symbols, and Function for creating symbolic variables.
Also exports common sympy types so users don't need to import sympy directly.
"""

from sympy import (
    Symbol, symbols, Function, Rational, Integer, Float,
    # Common types users might need
    Array, ImmutableDenseNDimArray, MutableDenseNDimArray,
    Heaviside, DiracDelta,
)

__all__ = [
    'Symbol',
    'symbols',
    'Function',
    'Rational',
    'Integer',
    'Float',
    # Common types
    'Array',
    'ImmutableDenseNDimArray',
    'MutableDenseNDimArray',
    'Heaviside',
    'DiracDelta',
]
