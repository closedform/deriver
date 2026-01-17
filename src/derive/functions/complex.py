"""
complex.py - Complex Number Functions.

Provides functions for working with complex numbers: Re, Im, Conjugate, Arg, Abs.
"""

from sympy import re, im, conjugate, arg, Abs as sympy_abs
from derive.functions.utils import alias_function

# Complex number functions
Re = alias_function('Re', re)
Im = alias_function('Im', im)
Conjugate = alias_function('Conjugate', conjugate)
Arg = alias_function('Arg', arg)
Abs = alias_function('Abs', sympy_abs)

__all__ = [
    'Re',
    'Im',
    'Conjugate',
    'Arg',
    'Abs',
]
