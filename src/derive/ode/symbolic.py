"""
symbolic.py - Symbolic ODE Solver.

Provides DSolve for symbolic differential equation solving.
"""

from typing import Any, List
from sympy import dsolve


def DSolve(eq: Any, func: Any, var: Any) -> List:
    """
    ODE solver.

    DSolve[eqn, y[x], x] - solve ODE for y as function of x

    Args:
        eq: Differential equation to solve
        func: Function to solve for (e.g., y(x))
        var: Independent variable

    Returns:
        List of solutions

    Examples:
        >>> x = Symbol('x')
        >>> y = Function('y')
        >>> DSolve(Eq(y(x).diff(x), y(x)), y(x), x)
        [Eq(y(x), C1*exp(x))]
    """
    return dsolve(eq, func)


__all__ = ['DSolve']
