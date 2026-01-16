"""
logic.py - Logic and Control Flow.

Provides comparison operators, boolean operators, and control flow functions.
"""

from typing import Any, Optional
from sympy import (
    Eq, Ne, Lt, Le, Gt, Ge,
    And as sympy_And, Or as sympy_Or, Not as sympy_Not,
    Xor as sympy_Xor, Nand as sympy_Nand, Nor as sympy_Nor,
    Piecewise as sympy_Piecewise, Max as sympy_Max, Min as sympy_Min,
)

# Comparison operators
Equal = Eq
Unequal = Ne
Less = Lt
LessEqual = Le
Greater = Gt
GreaterEqual = Ge

# Boolean operators
And = sympy_And
Or = sympy_Or
Not = sympy_Not
Xor = sympy_Xor
Nand = sympy_Nand
Nor = sympy_Nor

# Conditional
Piecewise = sympy_Piecewise
Max = sympy_Max
Min = sympy_Min


def If(cond: Any, true_val: Any, false_val: Optional[Any] = None) -> Any:
    """
    If.

    Args:
        cond: Condition to evaluate
        true_val: Value if condition is true
        false_val: Value if condition is false (default None)

    Returns:
        true_val if condition is true, else false_val
    """
    if cond:
        return true_val
    return false_val


def Which(*args: Any) -> Optional[Any]:
    """
    Which (like cond in Lisp).

    Which[cond1, val1, cond2, val2, ...]

    Args:
        *args: Alternating conditions and values

    Returns:
        First value whose condition is true

    Examples:
        >>> x = 5
        >>> Which(x < 0, "negative", x == 0, "zero", x > 0, "positive")
        "positive"
    """
    for i in range(0, len(args), 2):
        if i + 1 < len(args) and args[i]:
            return args[i + 1]
    return None


def Switch(expr: Any, *cases: Any) -> Optional[Any]:
    """
    Switch.

    Switch[expr, pat1, val1, pat2, val2, ...]

    Args:
        expr: Expression to match against
        *cases: Alternating patterns and values

    Returns:
        Value for the first matching pattern

    Examples:
        >>> Switch(2, 1, "one", 2, "two", 3, "three")
        "two"
    """
    for i in range(0, len(cases), 2):
        if i + 1 < len(cases) and expr == cases[i]:
            return cases[i + 1]
    return None


__all__ = [
    'Equal', 'Unequal', 'Less', 'LessEqual', 'Greater', 'GreaterEqual',
    'And', 'Or', 'Not', 'Xor', 'Nand', 'Nor',
    'If', 'Which', 'Switch', 'Piecewise', 'Max', 'Min',
]
