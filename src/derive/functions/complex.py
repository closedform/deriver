"""
complex.py - Complex Number Functions.

Provides functions for working with complex numbers: Re, Im, Conjugate, Arg, Abs.
"""

from sympy import re, im, conjugate, arg, Abs as sympy_abs

# Complex number functions
Re = re
Im = im
Conjugate = conjugate
Arg = arg
Abs = sympy_abs

__all__ = [
    'Re',
    'Im',
    'Conjugate',
    'Arg',
    'Abs',
]
