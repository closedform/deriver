"""
simplify.py - Expression Simplification and Transformation.

Provides functions for simplifying and transforming symbolic expressions.
"""

from typing import Any
import sympy as sp
from sympy import (
    simplify, expand, factor, collect, cancel, apart, together,
    trigsimp, powsimp, logcombine,
)

# Direct aliases
Simplify = simplify
Expand = expand
Factor = factor
Collect = collect
Cancel = cancel
Apart = apart
Together = together
TrigSimplify = trigsimp
PowerSimplify = powsimp
LogSimplify = logcombine


def TrigExpand(expr: Any) -> Any:
    """
    Expand trigonometric expressions.

    Args:
        expr: Expression containing trigonometric functions

    Returns:
        Expanded expression

    Examples:
        >>> x = Symbol('x')
        >>> TrigExpand(Sin(2*x))
        2*sin(x)*cos(x)
    """
    return sp.expand_trig(expr)


def TrigReduce(expr: Any) -> Any:
    """
    Reduce trigonometric expressions.

    Args:
        expr: Expression containing trigonometric functions

    Returns:
        Reduced expression

    Examples:
        >>> x = Symbol('x')
        >>> TrigReduce(Sin(x)*Cos(x))
        sin(2*x)/2
    """
    return trigsimp(expr)


__all__ = [
    'Simplify',
    'Expand',
    'Factor',
    'Collect',
    'Cancel',
    'Apart',
    'Together',
    'TrigSimplify',
    'TrigExpand',
    'TrigReduce',
    'PowerSimplify',
    'LogSimplify',
]
