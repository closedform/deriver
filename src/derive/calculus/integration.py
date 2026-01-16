"""
integration.py - Integration Operations.

Provides symbolic and numerical integration.
"""

from typing import Any, Union, Tuple
import numpy as np
import sympy as sp
from sympy import integrate, oo, Abs, Integral
from sympy.solvers import solve
from scipy import integrate as scipy_integrate

from derive.calculus.differentiation import D
from derive.algebra import Simplify


def Integrate(expr: Any, *args: Union[Any, Tuple[Any, Any, Any]], **kwargs) -> Any:
    """
    integration.

    Integrate[f, x] - indefinite integral
    Integrate[f, {x, a, b}] - definite integral from a to b

    Args:
        expr: Expression to integrate
        *args: Integration specifications. Either a single variable (indefinite)
               or tuples (var, a, b) for definite integrals.
        **kwargs: Additional options passed to SymPy

    Returns:
        The integral of the expression

    Examples:
        >>> x = Symbol('x')
        >>> Integrate(x**2, x)
        x**3/3
        >>> Integrate(Sin(x), (x, 0, Pi))
        2
    """
    if len(args) == 0:
        raise ValueError("Integrate requires at least one integration variable")

    result = expr
    for arg in args:
        if isinstance(arg, (list, tuple)) and len(arg) == 3:
            var, a, b = arg
            result = integrate(result, (var, a, b))
        else:
            result = integrate(result, arg)
    return result


def NIntegrate(expr: Any, *args: Tuple[Any, Any, Any], **kwargs) -> float:
    """
    Numerical integration.

    NIntegrate[f, {x, a, b}] - numerical definite integral

    Args:
        expr: Expression to integrate
        *args: Integration bounds as tuples (var, a, b)
        **kwargs: Additional options (currently unused)

    Returns:
        Numerical value of the integral

    Examples:
        >>> x = Symbol('x')
        >>> NIntegrate(Exp(-x**2), (x, 0, Infinity))
        0.8862269254527579  # sqrt(pi)/2
    """
    if len(args) == 0:
        raise ValueError("NIntegrate requires integration bounds")

    # Convert symbolic expression to numerical function
    if len(args) == 1 and isinstance(args[0], (list, tuple)) and len(args[0]) == 3:
        var, a, b = args[0]

        # Convert bounds to float (handle Infinity)
        a_val = float(a) if a != oo and a != -oo else (np.inf if a == oo else -np.inf)
        b_val = float(b) if b != oo and b != -oo else (np.inf if b == oo else -np.inf)

        # Create numerical function
        f = sp.lambdify(var, expr, modules=['numpy', 'scipy'])

        result, error = scipy_integrate.quad(f, a_val, b_val)
        return result

    raise ValueError("NIntegrate requires format: NIntegrate(expr, (var, a, b))")


def ChangeVariables(
    integral_or_expr: Any,
    old_var: Any,
    new_var: Any,
    substitution: Any,
    bounds: Tuple[Any, Any] = None
) -> Any:
    """
    Change variables in an integral (u-substitution).

    Transforms ∫f(x)dx to ∫f(g(u))|g'(u)|du with appropriate bounds.

    Args:
        integral_or_expr: The integrand expression or an Integral object
        old_var: The original integration variable (e.g., x)
        new_var: The new variable (e.g., u)
        substitution: Expression for old_var in terms of new_var (x = g(u))
        bounds: Optional tuple (a, b) for definite integrals. If provided,
                bounds are transformed automatically.

    Returns:
        Transformed integral or integrand with new variable

    Examples:
        >>> x, u = symbols('x u')
        >>> # Change x to u where x = u^2
        >>> ChangeVariables(x**2, x, u, u**2)
        2*u**5  # (u^2)^2 * |2u| = u^4 * 2u = 2u^5

        >>> # Definite integral with bounds transformation
        >>> ChangeVariables(Sqrt(x), x, u, u**2, bounds=(0, 4))
        (Integral(2*u**2, (u, 0, 2)), {0: 0, 4: 2})
    """
    # Compute the Jacobian: dx/du = g'(u)
    jacobian = D(substitution, new_var)

    # Substitute old_var -> substitution in the expression
    new_integrand = integral_or_expr.subs(old_var, substitution)

    # Multiply by |Jacobian| (absolute value for proper measure)
    # For symbolic work, we often assume positive Jacobian
    transformed = Simplify(new_integrand * Abs(jacobian))

    if bounds is not None:
        a, b = bounds
        # Solve for new bounds: find u such that substitution = a, substitution = b
        new_a_solutions = solve(substitution - a, new_var)
        new_b_solutions = solve(substitution - b, new_var)

        # Take the first real solution (could be improved with assumptions)
        new_a = new_a_solutions[0] if new_a_solutions else a
        new_b = new_b_solutions[0] if new_b_solutions else b

        # Return integral with new bounds and mapping info
        result_integral = Integral(transformed, (new_var, new_a, new_b))
        bounds_map = {a: new_a, b: new_b}
        return result_integral, bounds_map

    return transformed


def IntegrateWithSubstitution(
    expr: Any,
    var: Any,
    new_var: Any,
    substitution: Any,
    bounds: Tuple[Any, Any] = None
) -> Any:
    """
    Perform integration using a specified substitution.

    This combines ChangeVariables with evaluation of the integral.

    Args:
        expr: Expression to integrate
        var: Original variable
        new_var: Substitution variable
        substitution: Expression for var in terms of new_var
        bounds: Optional (a, b) for definite integral

    Returns:
        Result of the integral after substitution

    Examples:
        >>> x, u = symbols('x u')
        >>> # ∫ sin(x^2) * 2x dx with u = x^2
        >>> IntegrateWithSubstitution(Sin(x**2) * 2*x, x, u, Sqrt(u))
        -Cos(u)  # which is -cos(x^2)
    """
    if bounds is not None:
        result, _ = ChangeVariables(expr, var, new_var, substitution, bounds)
        return result.doit()
    else:
        transformed = ChangeVariables(expr, var, new_var, substitution)
        return Integrate(transformed, new_var)


__all__ = ['Integrate', 'NIntegrate', 'ChangeVariables', 'IntegrateWithSubstitution']
