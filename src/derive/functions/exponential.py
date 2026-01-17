"""
exponential.py - Exponential and Logarithmic Functions.

Provides Exp, Log, Ln, Sqrt, and Power functions.
"""

from typing import Any
from sympy import exp, log, sqrt
from derive.functions.utils import alias_function

# Direct aliases
Exp = alias_function('Exp', exp)
Log = alias_function('Log', log)
Ln = alias_function('Ln', log)  # Natural log alias
Sqrt = alias_function('Sqrt', sqrt)


def Power(base: Any, exponent: Any) -> Any:
    """
    Power function.

    Power[x, y] - x raised to power y

    Args:
        base: The base of the power
        exponent: The exponent

    Returns:
        base raised to the power of exponent

    Examples:
        >>> Power(2, 3)
        8
        >>> x = Symbol('x')
        >>> Power(x, 2)
        x**2
    """
    return base ** exponent


__all__ = [
    'Exp',
    'Log',
    'Ln',
    'Sqrt',
    'Power',
]
