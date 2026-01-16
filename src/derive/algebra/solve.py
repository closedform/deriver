"""
solve.py - Equation Solving.

Provides equation solvers: Solve, NSolve, FindRoot.
"""

from typing import Any, List, Dict, Tuple
import sympy as sp
from sympy import solve
from scipy import optimize as scipy_optimize


def Solve(eqns: Any, vars: Any, **kwargs) -> List[Dict]:
    """
    equation solver.

    Solve[eqn, x] - solve equation for x
    Solve[{eqn1, eqn2}, {x, y}] - solve system of equations

    Args:
        eqns: Equation(s) to solve (can be single or list)
        vars: Variable(s) to solve for (can be single or list)
        **kwargs: Additional options passed to SymPy

    Returns:
        List of solution dictionaries

    Examples:
        >>> x = Symbol('x')
        >>> Solve(x**2 - 4, x)
        [{x: -2}, {x: 2}]
    """
    result = solve(eqns, vars, dict=True)
    return result


def NSolve(eqns: Any, vars: Any, **kwargs) -> List[Dict]:
    """
    Numerical equation solver.

    Returns numerical solutions to equations.

    Args:
        eqns: Equation(s) to solve
        vars: Variable(s) to solve for
        **kwargs: Additional options

    Returns:
        List of numerical solution dictionaries

    Examples:
        >>> x = Symbol('x')
        >>> NSolve(x**2 - 2, x)
        [{x: -1.4142135623730951}, {x: 1.4142135623730951}]
    """
    solutions = solve(eqns, vars, dict=True)
    numerical = []
    for sol in solutions:
        num_sol = {k: complex(v).real if v.is_real else complex(v) for k, v in sol.items()}
        numerical.append(num_sol)
    return numerical


def FindRoot(expr: Any, var_guess: Tuple[Any, float]) -> Dict:
    """
    numerical root finding.

    FindRoot[f, {x, x0}] - find root starting from x0

    Args:
        expr: Expression to find root of (f(x) = 0)
        var_guess: Tuple (var, x0) specifying variable and initial guess

    Returns:
        Dictionary with the root

    Examples:
        >>> x = Symbol('x')
        >>> FindRoot(x**2 - 2, (x, 1))
        {x: 1.4142135623730951}
    """
    if isinstance(var_guess, (list, tuple)) and len(var_guess) == 2:
        var, x0 = var_guess
        f = sp.lambdify(var, expr, modules=['numpy'])

        result = scipy_optimize.fsolve(f, float(x0))
        return {var: result[0]}

    raise ValueError("FindRoot requires format: FindRoot(expr, (var, x0))")


__all__ = ['Solve', 'NSolve', 'FindRoot']
