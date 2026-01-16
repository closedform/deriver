"""
optimize - Convex Optimization Interface

A thin wrapper around cvxpy for convex optimization problems.
Provides a simple interface while letting cvxpy handle solver backends.
"""

from derive.optimize.core import (
    OptVar, OptimizationProblem, Minimize, Maximize,
    Norm, Sum, Quad, PositiveSemidefinite,
)

__all__ = [
    'OptVar',
    'OptimizationProblem',
    'Minimize',
    'Maximize',
    'Norm',
    'Sum',
    'Quad',
    'PositiveSemidefinite',
]
