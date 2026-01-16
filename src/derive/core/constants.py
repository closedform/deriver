"""
constants.py - Mathematical Constants.

Provides constants: Pi, E, I, Infinity, etc.
"""

from sympy import pi, E as sympy_E, I as sympy_I, oo, zoo, nan

# constants
Pi = pi
E = sympy_E
I = sympy_I
Infinity = oo
oo = oo  # Also export as oo for convenience
ComplexInfinity = zoo
Indeterminate = nan

__all__ = [
    'Pi',
    'E',
    'I',
    'Infinity',
    'oo',
    'ComplexInfinity',
    'Indeterminate',
]
