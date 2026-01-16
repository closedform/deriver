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

# Basic trigonometric functions
Sin = sin
Cos = cos
Tan = tan
Cot = cot
Sec = sec
Csc = csc

# Hyperbolic functions
Sinh = sinh
Cosh = cosh
Tanh = tanh
Coth = coth
Sech = sech
Csch = csch

# Inverse trigonometric functions
ArcSin = asin
ArcCos = acos
ArcTan = atan
ArcCot = acot
ArcSec = asec
ArcCsc = acsc

# Inverse hyperbolic functions
ArcSinh = asinh
ArcCosh = acosh
ArcTanh = atanh
ArcCoth = acoth
ArcSech = asech
ArcCsch = acsch

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
