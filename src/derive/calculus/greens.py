"""
greens.py - Green's Functions for PDEs

Provides Green's functions for common differential operators.
These are solutions to equations of the form L[G] = δ.
"""

import sympy as sp
from sympy import (
    Symbol, symbols, Piecewise, Heaviside, Abs, exp, pi, I,
)
from typing import Dict, Optional


def GreenFunction(operator: sp.Expr, var: Symbol, source_var: Symbol,
                  boundary_conditions: Optional[Dict] = None) -> sp.Expr:
    """
    Compute the Green's function for a linear differential operator.

    The Green's function G(x, x') satisfies:
    L[G(x, x')] = δ(x - x')

    Args:
        operator: The differential operator L acting on a function
        var: The independent variable x
        source_var: The source point x'
        boundary_conditions: Optional boundary conditions

    Returns:
        The Green's function G(x, x')

    Examples:
        >>> x, xp = symbols('x xp')
        >>> # Green's function for d²/dx² with Dirichlet BCs at x=0, x=1
        >>> # G(x, x') = x_<(1 - x_>) where x_< = min(x,x'), x_> = max(x,x')
    """
    # For the common case of -d²/dx² on [0, L]
    # Green's function is: G(x, x') = (x_<)(L - x_>) / L
    # where x_< = min(x, x'), x_> = max(x, x')

    # For simplicity, provide a symbolic representation
    # Real implementation would solve the BVP

    # Create a piecewise Green's function for simple cases
    # This is a placeholder that represents the structure
    L = symbols('L', positive=True)

    G = Piecewise(
        (var * (L - source_var) / L, var < source_var),
        (source_var * (L - var) / L, True)
    )

    return G


def GreenFunctionPoisson1D(x: Symbol, xp: Symbol, L: sp.Expr) -> sp.Expr:
    """
    Green's function for 1D Poisson equation -d²u/dx² = f on [0, L].

    With homogeneous Dirichlet BCs: u(0) = u(L) = 0

    G(x, x') = { x(L-x')/L  if x < x'
              { x'(L-x)/L  if x > x'

    Args:
        x: Evaluation point
        xp: Source point
        L: Domain length

    Returns:
        Green's function G(x, x')
    """
    return Piecewise(
        (x * (L - xp) / L, x < xp),
        (xp * (L - x) / L, True)
    )


def GreenFunctionHelmholtz1D(x: Symbol, xp: Symbol, k: Symbol) -> sp.Expr:
    """
    Green's function for 1D Helmholtz equation (d²/dx² + k²)u = f.

    G(x, x') = -(i/2k) * exp(ik|x - x'|)

    For the outgoing wave solution.

    Args:
        x: Evaluation point
        xp: Source point
        k: Wave number

    Returns:
        Green's function G(x, x')
    """
    return -I / (2*k) * exp(I * k * Abs(x - xp))


def GreenFunctionLaplacian3D(r: sp.Expr) -> sp.Expr:
    """
    Green's function for 3D Laplacian ∇²G = δ³(r).

    G(r) = -1/(4πr)

    Args:
        r: Distance from source (can be expression like sqrt(x² + y² + z²))

    Returns:
        Green's function G(r)
    """
    return -1 / (4 * pi * r)


def GreenFunctionWave1D(x: Symbol, t: Symbol, xp: Symbol, tp: Symbol, c: Symbol) -> sp.Expr:
    """
    Green's function for 1D wave equation (∂²/∂t² - c²∂²/∂x²)u = f.

    G(x, t; x', t') = (1/2c) * H(t - t' - |x - x'|/c)

    where H is the Heaviside step function.

    Args:
        x, t: Field point
        xp, tp: Source point
        c: Wave speed

    Returns:
        Green's function (retarded)
    """
    return Heaviside(t - tp - Abs(x - xp)/c) / (2*c)


__all__ = [
    'GreenFunction', 'GreenFunctionPoisson1D', 'GreenFunctionHelmholtz1D',
    'GreenFunctionLaplacian3D', 'GreenFunctionWave1D',
]
