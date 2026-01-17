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
from derive.functions.utils import alias_function

# Basic special functions
Factorial = alias_function('Factorial', factorial)
Binomial = alias_function('Binomial', binomial)
Gamma = alias_function('Gamma', gamma)
Beta = alias_function('Beta', beta)

# Bessel functions
BesselJ = alias_function('BesselJ', besselj)
BesselY = alias_function('BesselY', bessely)
BesselI = alias_function('BesselI', besseli)
BesselK = alias_function('BesselK', besselk)
HankelH1 = alias_function('HankelH1', hankel1)
HankelH2 = alias_function('HankelH2', hankel2)


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
LegendreP = alias_function('LegendreP', legendre)
AssociatedLegendreP = alias_function('AssociatedLegendreP', assoc_legendre)
ChebyshevT = alias_function('ChebyshevT', chebyshevt)
ChebyshevU = alias_function('ChebyshevU', chebyshevu)
HermiteH = alias_function('HermiteH', hermite)
LaguerreL = alias_function('LaguerreL', laguerre)
AssociatedLaguerreL = alias_function('AssociatedLaguerreL', assoc_laguerre)
GegenbauerC = alias_function('GegenbauerC', gegenbauer)
JacobiP = alias_function('JacobiP', jacobi)

# Spherical harmonics
SphericalHarmonicY = alias_function('SphericalHarmonicY', Ynm)

# Elliptic integrals
EllipticK = alias_function('EllipticK', elliptic_k)
EllipticE = alias_function('EllipticE', elliptic_e)
EllipticF = alias_function('EllipticF', elliptic_f)
EllipticPi = alias_function('EllipticPi', elliptic_pi)

# Error functions
Erf = alias_function('Erf', erf)
Erfc = alias_function('Erfc', erfc)
Erfi = alias_function('Erfi', erfi)

# Exponential integrals
ExpIntegralEi = alias_function('ExpIntegralEi', Ei)
SinIntegral = alias_function('SinIntegral', Si)
CosIntegral = alias_function('CosIntegral', Ci)
LogIntegral = alias_function('LogIntegral', li)

# Fresnel integrals
FresnelS = alias_function('FresnelS', fresnels)
FresnelC = alias_function('FresnelC', fresnelc)

# Airy functions
AiryAi = alias_function('AiryAi', airyai)
AiryBi = alias_function('AiryBi', airybi)

# Zeta and polylog
Zeta = alias_function('Zeta', zeta)
PolyLog = alias_function('PolyLog', polylog)

# Hypergeometric functions
Hypergeometric2F1 = alias_function('Hypergeometric2F1', hyper)


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
