"""
Calculus module - Differentiation, Integration, Limits, Series, Variational Calculus.

Provides calculus operations.
"""

from derive.calculus.differentiation import D
from derive.calculus.integration import (
    Integrate, NIntegrate, ChangeVariables, IntegrateWithSubstitution
)
from derive.calculus.limits import Limit
from derive.calculus.series import Series, Sum, Product
from derive.calculus.variational import VariationalDerivative, EulerLagrangeEquation
from derive.calculus.greens import (
    GreenFunction, GreenFunctionPoisson1D, GreenFunctionHelmholtz1D,
    GreenFunctionLaplacian3D, GreenFunctionWave1D,
)
from derive.calculus.transforms import (
    FourierTransform, InverseFourierTransform,
    LaplaceTransform, InverseLaplaceTransform,
    Convolve,
)

__all__ = [
    'D',
    'Integrate',
    'NIntegrate',
    'ChangeVariables',
    'IntegrateWithSubstitution',
    'Limit',
    'Series',
    'Sum',
    'Product',
    # Variational calculus
    'VariationalDerivative',
    'EulerLagrangeEquation',
    # Green's functions
    'GreenFunction',
    'GreenFunctionPoisson1D',
    'GreenFunctionHelmholtz1D',
    'GreenFunctionLaplacian3D',
    'GreenFunctionWave1D',
    # Integral transforms
    'FourierTransform',
    'InverseFourierTransform',
    'LaplaceTransform',
    'InverseLaplaceTransform',
    'Convolve',
]
