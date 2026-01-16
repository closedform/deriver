"""
series.py - Series, Sums, and Products.

Provides series expansions and summations.
"""

from typing import Any, Tuple
from sympy import series, summation, product as sympy_product


def Series(expr: Any, var_point_order: Tuple[Any, Any, int]) -> Any:
    """
    series expansion.

    Series[f, {x, x0, n}] - expand f around x=x0 to order n

    Args:
        expr: Expression to expand
        var_point_order: Tuple (var, x0, n) specifying variable, expansion point, and order

    Returns:
        The series expansion

    Examples:
        >>> x = Symbol('x')
        >>> Series(Exp(x), (x, 0, 5))
        1 + x + x**2/2 + x**3/6 + x**4/24 + x**5/120 + O(x**6)
    """
    if isinstance(var_point_order, (list, tuple)) and len(var_point_order) == 3:
        var, x0, n = var_point_order
        return series(expr, var, x0, n + 1)
    raise ValueError("Series requires format: Series(expr, (var, x0, n))")


def Sum(expr: Any, *args: Tuple[Any, Any, Any]) -> Any:
    """
    summation.

    Sum[f, {i, a, b}] - sum f for i from a to b
    Sum[f, {i, a, b}, {j, c, d}] - multiple sums

    Args:
        expr: Expression to sum
        *args: Summation bounds as tuples (var, start, end)

    Returns:
        The sum

    Examples:
        >>> i = Symbol('i')
        >>> Sum(i, (i, 1, 10))
        55
        >>> n = Symbol('n')
        >>> Sum(i, (i, 1, n))
        n*(n + 1)/2
    """
    result = expr
    for arg in args:
        if isinstance(arg, (list, tuple)) and len(arg) == 3:
            var, a, b = arg
            result = summation(result, (var, a, b))
        else:
            raise ValueError("Sum requires format: Sum(expr, (var, a, b))")
    return result


def Product(expr: Any, *args: Tuple[Any, Any, Any]) -> Any:
    """
    product.

    Product[f, {i, a, b}] - product of f for i from a to b

    Args:
        expr: Expression to multiply
        *args: Product bounds as tuples (var, start, end)

    Returns:
        The product

    Examples:
        >>> i = Symbol('i')
        >>> Product(i, (i, 1, 5))
        120
        >>> n = Symbol('n')
        >>> Product(i, (i, 1, n))
        factorial(n)
    """
    result = expr
    for arg in args:
        if isinstance(arg, (list, tuple)) and len(arg) == 3:
            var, a, b = arg
            result = sympy_product(result, (var, a, b))
        else:
            raise ValueError("Product requires format: Product(expr, (var, a, b))")
    return result


__all__ = ['Series', 'Sum', 'Product']
