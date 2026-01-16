"""
Algebra module - Equation solving, simplification, and linear algebra.

Provides algebraic operations.
"""

from derive.algebra.solve import Solve, NSolve, FindRoot
from derive.algebra.simplify import (
    Simplify, Expand, Factor, Collect,
    Cancel, Apart, Together, TrigSimplify, TrigExpand, TrigReduce,
    PowerSimplify, LogSimplify,
)
from derive.algebra.linear import (
    Matrix, Dot, Transpose, Inverse, Det,
    Eigenvalues, Eigenvectors,
    IdentityMatrix, DiagonalMatrix, ZeroMatrix,
    Tr, MatrixRank, NullSpace, RowReduce,
    ConjugateTranspose, MatrixExp, CharacteristicPolynomial,
)

__all__ = [
    # Solving
    'Solve', 'NSolve', 'FindRoot',
    # Simplification
    'Simplify', 'Expand', 'Factor', 'Collect',
    'Cancel', 'Apart', 'Together', 'TrigSimplify', 'TrigExpand', 'TrigReduce',
    'PowerSimplify', 'LogSimplify',
    # Linear algebra
    'Matrix', 'Dot', 'Transpose', 'Inverse', 'Det',
    'Eigenvalues', 'Eigenvectors',
    'IdentityMatrix', 'DiagonalMatrix', 'ZeroMatrix',
    'Tr', 'MatrixRank', 'NullSpace', 'RowReduce',
    'ConjugateTranspose', 'MatrixExp', 'CharacteristicPolynomial',
]
