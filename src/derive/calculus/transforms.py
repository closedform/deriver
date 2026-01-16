"""
transforms.py - Integral Transforms

Provides Fourier, Laplace, and related integral transforms.
"""

import sympy as sp
from sympy import (
    fourier_transform, inverse_fourier_transform,
    laplace_transform, inverse_laplace_transform,
    Symbol, oo, exp, I, pi, sqrt,
)
from typing import Any, Optional, Tuple


def FourierTransform(expr: Any, t: Symbol, omega: Symbol, **kwargs) -> Any:
    """
    Compute the Fourier transform of an expression.

    F[f(t)](ω) = ∫_{-∞}^{∞} f(t) e^{-iωt} dt

    Args:
        expr: Expression to transform (function of t)
        t: Time/space variable
        omega: Frequency variable
        **kwargs: Additional options passed to SymPy

    Returns:
        Fourier transform of the expression

    Examples:
        >>> t, omega = symbols('t omega')
        >>> FourierTransform(exp(-t**2), t, omega)
        sqrt(pi)*exp(-omega**2/4)
    """
    return fourier_transform(expr, t, omega, **kwargs)


def InverseFourierTransform(expr: Any, omega: Symbol, t: Symbol, **kwargs) -> Any:
    """
    Compute the inverse Fourier transform.

    f(t) = (1/2π) ∫_{-∞}^{∞} F(ω) e^{iωt} dω

    Args:
        expr: Expression to inverse transform (function of omega)
        omega: Frequency variable
        t: Time/space variable
        **kwargs: Additional options passed to SymPy

    Returns:
        Inverse Fourier transform

    Examples:
        >>> t, omega = symbols('t omega')
        >>> InverseFourierTransform(sqrt(pi)*exp(-omega**2/4), omega, t)
        exp(-t**2)
    """
    return inverse_fourier_transform(expr, omega, t, **kwargs)


def LaplaceTransform(expr: Any, t: Symbol, s: Symbol, **kwargs) -> Tuple[Any, Any, Any]:
    """
    Compute the Laplace transform of an expression.

    L[f(t)](s) = ∫_0^{∞} f(t) e^{-st} dt

    Args:
        expr: Expression to transform (function of t)
        t: Time variable
        s: Complex frequency variable
        **kwargs: Additional options (noconds=True to skip convergence conditions)

    Returns:
        Tuple of (transform, convergence_plane, conditions) or just transform if noconds=True

    Examples:
        >>> t, s = symbols('t s')
        >>> LaplaceTransform(exp(-t), t, s, noconds=True)
        1/(s + 1)
        >>> LaplaceTransform(sin(t), t, s, noconds=True)
        1/(s**2 + 1)
    """
    return laplace_transform(expr, t, s, **kwargs)


def InverseLaplaceTransform(expr: Any, s: Symbol, t: Symbol, **kwargs) -> Any:
    """
    Compute the inverse Laplace transform.

    f(t) = (1/2πi) ∫_{γ-i∞}^{γ+i∞} F(s) e^{st} ds

    Args:
        expr: Expression to inverse transform (function of s)
        s: Complex frequency variable
        t: Time variable
        **kwargs: Additional options passed to SymPy

    Returns:
        Inverse Laplace transform

    Examples:
        >>> t, s = symbols('t s')
        >>> InverseLaplaceTransform(1/(s + 1), s, t)
        exp(-t)*Heaviside(t)
    """
    return inverse_laplace_transform(expr, s, t, **kwargs)


def Convolve(f: Any, g: Any, t: Symbol, tau: Optional[Symbol] = None) -> Any:
    """
    Compute the convolution of two functions.

    (f * g)(t) = ∫_{-∞}^{∞} f(τ) g(t - τ) dτ

    Args:
        f: First function
        g: Second function
        t: Variable
        tau: Integration variable (auto-generated if not provided)

    Returns:
        Convolution integral

    Examples:
        >>> t = Symbol('t')
        >>> from sympy import Heaviside
        >>> # Convolution of two step functions
        >>> Convolve(Heaviside(t), Heaviside(t), t)
    """
    if tau is None:
        tau = Symbol('tau')

    from sympy import Integral, oo

    # Substitute t -> tau in f, and t -> (t - tau) in g
    f_sub = f.subs(t, tau)
    g_sub = g.subs(t, t - tau)

    return Integral(f_sub * g_sub, (tau, -oo, oo))


__all__ = [
    'FourierTransform',
    'InverseFourierTransform',
    'LaplaceTransform',
    'InverseLaplaceTransform',
    'Convolve',
]
