"""
differentiation.py - Differentiation Operations.

Provides differentiation with the D function.
"""

from typing import Any, Union, Tuple
from sympy import diff


def D(expr: Any, *args: Union[Any, Tuple[Any, int]]) -> Any:
    """
    differentiation.

    D[f, x] - differentiate f with respect to x
    D[f, {x, n}] - differentiate f n times with respect to x
    D[f, x, y] - differentiate f with respect to x, then y

    Args:
        expr: Expression to differentiate
        *args: Variables to differentiate with respect to.
               Can be single symbols or tuples (var, n) for nth derivative.

    Returns:
        The derivative of the expression

    Examples:
        >>> x = Symbol('x')
        >>> D(x**3, x)
        3*x**2
        >>> D(Sin(x), x)
        cos(x)
        >>> D(x**5, (x, 3))
        60*x**2
    """
    if len(args) == 0:
        raise ValueError("D requires at least one differentiation variable")

    result = expr
    for arg in args:
        if isinstance(arg, (list, tuple)) and len(arg) == 2:
            var, n = arg
            result = diff(result, var, n)
        else:
            result = diff(result, arg)
    return result


__all__ = ['D']
