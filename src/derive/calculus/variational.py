"""
variational.py - Variational Calculus

Provides variational derivatives for deriving equations of motion
from Lagrangian densities.
"""

import sympy as sp
from sympy import Symbol, symbols, Function, Rational
from typing import List

# Use derive's own APIs for self-consistency
from derive.calculus.differentiation import D
from derive.algebra import Simplify


def VariationalDerivative(lagrangian: sp.Expr, field: Function, coords: List[Symbol]) -> sp.Expr:
    """
    Compute the variational (functional) derivative of a Lagrangian.

    This gives the Euler-Lagrange equations: δL/δφ = 0

    For L(φ, ∂_μ φ), the variational derivative is:
    δL/δφ = ∂L/∂φ - ∂_μ(∂L/∂(∂_μ φ))

    Args:
        lagrangian: The Lagrangian density expression
        field: The field function (e.g., phi(x, t))
        coords: Coordinate variables

    Returns:
        The Euler-Lagrange equation (set to 0 for equation of motion)

    Examples:
        >>> x, t = symbols('x t')
        >>> phi = Function('phi')(x, t)
        >>> # Klein-Gordon Lagrangian: L = (1/2)(∂_t φ)^2 - (1/2)(∂_x φ)^2 - (1/2)m^2 φ^2
        >>> m = Symbol('m')
        >>> L = Rational(1,2)*diff(phi, t)**2 - Rational(1,2)*diff(phi, x)**2 - Rational(1,2)*m**2*phi**2
        >>> eq = VariationalDerivative(L, phi, [x, t])
        >>> # Should give: ∂_t^2 φ - ∂_x^2 φ + m^2 φ = 0
    """
    # Get the field and its derivatives
    result = D(lagrangian, field)

    for coord in coords:
        # ∂L/∂(∂_μ φ)
        d_field = D(field, coord)
        partial_L = D(lagrangian, d_field)
        # ∂_μ(∂L/∂(∂_μ φ))
        result -= D(partial_L, coord)

    return Simplify(result)


def EulerLagrangeEquation(action_density: sp.Expr, field: Function,
                          coords: List[Symbol]) -> sp.Expr:
    """
    Derive the Euler-Lagrange equation from an action density.

    This is an alias for VariationalDerivative, named to match standard convention.

    Args:
        action_density: The Lagrangian density L
        field: The field to vary
        coords: Spacetime coordinates

    Returns:
        The Euler-Lagrange equation (= 0 for equations of motion)
    """
    return VariationalDerivative(action_density, field, coords)


__all__ = [
    'VariationalDerivative', 'EulerLagrangeEquation',
]
