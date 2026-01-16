"""
Core module - Symbols and Constants.

This module provides the foundational elements: symbols, functions, and mathematical constants.
"""

from derive.core.symbols import (
    Symbol, symbols, Function,
    Rational, Integer, Float,
    # Common types
    Array, ImmutableDenseNDimArray, MutableDenseNDimArray,
    Heaviside, DiracDelta,
)
from derive.core.constants import (
    Pi, E, I, Infinity, oo, ComplexInfinity, Indeterminate,
)
from derive.core.numbers import (
    rationalize, to_rational, exact, numerical,
    is_exact_mode, is_numerical_mode, get_numerical_precision,
    auto_rational, SmartNumber, S, expr,
    float_to_rational, ensure_rational,
    R, Half, Third, Quarter, TwoThirds, ThreeQuarters,
)

__all__ = [
    # Symbols
    'Symbol', 'symbols', 'Function',
    'Rational', 'Integer', 'Float',
    # Common types
    'Array', 'ImmutableDenseNDimArray', 'MutableDenseNDimArray',
    'Heaviside', 'DiracDelta',
    # Constants
    'Pi', 'E', 'I', 'Infinity', 'oo', 'ComplexInfinity', 'Indeterminate',
    # Smart number handling
    'rationalize', 'to_rational', 'exact', 'numerical',
    'is_exact_mode', 'is_numerical_mode', 'get_numerical_precision',
    'auto_rational', 'SmartNumber', 'S', 'expr',
    'float_to_rational', 'ensure_rational',
    # Rational shortcuts
    'R', 'Half', 'Third', 'Quarter', 'TwoThirds', 'ThreeQuarters',
]
