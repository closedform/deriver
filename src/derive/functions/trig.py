"""
trig.py - Trigonometric Functions.

Provides trigonometric functions with CamelCase naming.
"""

from sympy import (
    sin, cos, tan, cot, sec, csc,
    sinh, cosh, tanh, coth, sech, csch,
    asin, acos, atan, acot, asec, acsc,
    asinh, acosh, atanh, acoth, asech, acsch,
)
from derive.functions.utils import alias_function

# Basic trigonometric functions
Sin = alias_function('Sin', sin)
Cos = alias_function('Cos', cos)
Tan = alias_function('Tan', tan)
Cot = alias_function('Cot', cot)
Sec = alias_function('Sec', sec)
Csc = alias_function('Csc', csc)

# Hyperbolic functions
Sinh = alias_function('Sinh', sinh)
Cosh = alias_function('Cosh', cosh)
Tanh = alias_function('Tanh', tanh)
Coth = alias_function('Coth', coth)
Sech = alias_function('Sech', sech)
Csch = alias_function('Csch', csch)

# Inverse trigonometric functions
ArcSin = alias_function('ArcSin', asin)
ArcCos = alias_function('ArcCos', acos)
ArcTan = alias_function('ArcTan', atan)
ArcCot = alias_function('ArcCot', acot)
ArcSec = alias_function('ArcSec', asec)
ArcCsc = alias_function('ArcCsc', acsc)

# Inverse hyperbolic functions
ArcSinh = alias_function('ArcSinh', asinh)
ArcCosh = alias_function('ArcCosh', acosh)
ArcTanh = alias_function('ArcTanh', atanh)
ArcCoth = alias_function('ArcCoth', acoth)
ArcSech = alias_function('ArcSech', asech)
ArcCsch = alias_function('ArcCsch', acsch)

__all__ = [
    # Basic trig
    'Sin', 'Cos', 'Tan', 'Cot', 'Sec', 'Csc',
    # Hyperbolic
    'Sinh', 'Cosh', 'Tanh', 'Coth', 'Sech', 'Csch',
    # Inverse trig
    'ArcSin', 'ArcCos', 'ArcTan', 'ArcCot', 'ArcSec', 'ArcCsc',
    # Inverse hyperbolic
    'ArcSinh', 'ArcCosh', 'ArcTanh', 'ArcCoth', 'ArcSech', 'ArcCsch',
]
