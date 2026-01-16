"""
special.py - Special Mathematical Functions.

Provides special functions commonly used in physics and mathematics:
- Bessel functions
- Orthogonal polynomials
- Elliptic integrals
- Error functions
- Hypergeometric functions
- And more
"""

from typing import List, Any
import sympy as sp
from sympy import (
    factorial, binomial, gamma, beta,
    besselj, bessely, besseli, besselk,
    hankel1, hankel2,
    legendre, assoc_legendre,
    chebyshevt, chebyshevu,
    hermite, laguerre, assoc_laguerre,
    gegenbauer, jacobi,
    Ynm,  # Spherical harmonics
    zeta, polylog,
    erf, erfc, erfi,
    Ei, Si, Ci, li,
    fresnels, fresnelc,
    airyai, airybi,
    elliptic_k, elliptic_e, elliptic_f, elliptic_pi,
    hyper, meijerg,
    sqrt, Rational,
)

# Basic special functions
Factorial = factorial
Binomial = binomial
Gamma = gamma
Beta = beta

# Bessel functions
BesselJ = besselj
BesselY = bessely
BesselI = besseli
BesselK = besselk
HankelH1 = hankel1
HankelH2 = hankel2


def SphericalBesselJ(n: Any, x: Any) -> Any:
    """
    Spherical Bessel function of the first kind j_n(x).

    Args:
        n: Order of the function
        x: Argument

    Returns:
        j_n(x) = sqrt(pi/(2x)) * J_{n+1/2}(x)
    """
    return sqrt(sp.pi / (2 * x)) * BesselJ(n + Rational(1, 2), x)


def SphericalBesselY(n: Any, x: Any) -> Any:
    """
    Spherical Bessel function of the second kind y_n(x).

    Args:
        n: Order of the function
        x: Argument

    Returns:
        y_n(x) = sqrt(pi/(2x)) * Y_{n+1/2}(x)
    """
    return sqrt(sp.pi / (2 * x)) * BesselY(n + Rational(1, 2), x)


# Orthogonal polynomials
LegendreP = legendre
AssociatedLegendreP = assoc_legendre
ChebyshevT = chebyshevt
ChebyshevU = chebyshevu
HermiteH = hermite
LaguerreL = laguerre
AssociatedLaguerreL = assoc_laguerre
GegenbauerC = gegenbauer
JacobiP = jacobi

# Spherical harmonics
SphericalHarmonicY = Ynm

# Elliptic integrals
EllipticK = elliptic_k
EllipticE = elliptic_e
EllipticF = elliptic_f
EllipticPi = elliptic_pi

# Error functions
Erf = erf
Erfc = erfc
Erfi = erfi

# Exponential integrals
ExpIntegralEi = Ei
SinIntegral = Si
CosIntegral = Ci
LogIntegral = li

# Fresnel integrals
FresnelS = fresnels
FresnelC = fresnelc

# Airy functions
AiryAi = airyai
AiryBi = airybi

# Zeta and polylog
Zeta = zeta
PolyLog = polylog

# Hypergeometric functions
Hypergeometric2F1 = hyper


def HypergeometricPFQ(a_list: List, b_list: List, z: Any) -> Any:
    """
    Generalized hypergeometric function pFq.

    HypergeometricPFQ[{a1, ..., ap}, {b1, ..., bq}, z]

    Args:
        a_list: Upper parameters
        b_list: Lower parameters
        z: Argument

    Returns:
        The generalized hypergeometric function pFq(a; b; z)
    """
    return hyper(a_list, b_list, z)


def MeijerG(upper_lists: List, lower_lists: List, z: Any) -> Any:
    """
    Meijer G-function (very general special function).

    MeijerG[{{a1,...}, {ap+1,...}}, {{b1,...}, {bq+1,...}}, z]

    Args:
        upper_lists: Upper parameters as [[], []] or similar
        lower_lists: Lower parameters as [[], []] or similar
        z: Argument

    Returns:
        The Meijer G-function
    """
    return meijerg(upper_lists, lower_lists, z)


__all__ = [
    # Basic special functions
    'Factorial', 'Binomial', 'Gamma', 'Beta',
    # Bessel functions
    'BesselJ', 'BesselY', 'BesselI', 'BesselK',
    'HankelH1', 'HankelH2',
    'SphericalBesselJ', 'SphericalBesselY',
    # Orthogonal polynomials
    'LegendreP', 'AssociatedLegendreP',
    'ChebyshevT', 'ChebyshevU',
    'HermiteH', 'LaguerreL', 'AssociatedLaguerreL',
    'GegenbauerC', 'JacobiP',
    # Spherical harmonics
    'SphericalHarmonicY',
    # Elliptic integrals
    'EllipticK', 'EllipticE', 'EllipticF', 'EllipticPi',
    # Error functions
    'Erf', 'Erfc', 'Erfi',
    # Exponential integrals
    'ExpIntegralEi', 'SinIntegral', 'CosIntegral', 'LogIntegral',
    # Fresnel integrals
    'FresnelS', 'FresnelC',
    # Airy functions
    'AiryAi', 'AiryBi',
    # Number theory & QFT
    'Zeta', 'PolyLog',
    # Hypergeometric functions
    'Hypergeometric2F1', 'HypergeometricPFQ', 'MeijerG',
]
