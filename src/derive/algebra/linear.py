"""
linear.py - Linear Algebra Operations.

Provides linear algebra functions.
"""

from typing import Any, List, Optional
import sympy as sp
from sympy import Matrix, eye, zeros, diag, Symbol

# Re-export Matrix
Matrix = Matrix


def Dot(*args: Any) -> Any:
    """
    dot product / matrix multiplication.

    Dot[a, b] or a.b - matrix multiplication or dot product

    Args:
        *args: Two or more matrices/vectors to multiply

    Returns:
        The product

    Examples:
        >>> Dot(Matrix([[1, 2], [3, 4]]), Matrix([[5], [6]]))
        Matrix([[17], [39]])
    """
    if len(args) < 2:
        raise ValueError("Dot requires at least 2 arguments")

    result = args[0]
    for arg in args[1:]:
        if isinstance(result, Matrix) and isinstance(arg, Matrix):
            result = result * arg
        elif hasattr(result, '__matmul__'):
            result = result @ arg
        else:
            result = result * arg
    return result


def Transpose(m: Any) -> Any:
    """
    Matrix transpose.

    Args:
        m: Matrix to transpose

    Returns:
        Transposed matrix
    """
    if isinstance(m, Matrix):
        return m.T
    return sp.transpose(m)


def Inverse(m: Any) -> Any:
    """
    Matrix inverse.

    Args:
        m: Matrix to invert

    Returns:
        Inverse matrix
    """
    if isinstance(m, Matrix):
        return m.inv()
    return m**(-1)


def Det(m: Any) -> Any:
    """
    Matrix determinant.

    Args:
        m: Matrix

    Returns:
        Determinant
    """
    if isinstance(m, Matrix):
        return m.det()
    return sp.det(m)


def Eigenvalues(m: Any) -> dict:
    """
    Compute eigenvalues of a matrix.

    Args:
        m: Matrix

    Returns:
        Dictionary mapping eigenvalues to their multiplicities

    Examples:
        >>> Eigenvalues(Matrix([[1, 2], [2, 1]]))
        {-1: 1, 3: 1}
    """
    if isinstance(m, Matrix):
        return m.eigenvals()
    return sp.Matrix(m).eigenvals()


def Eigenvectors(m: Any) -> List:
    """
    Compute eigenvectors of a matrix.

    Args:
        m: Matrix

    Returns:
        List of (eigenvalue, multiplicity, [eigenvectors])
    """
    if isinstance(m, Matrix):
        return m.eigenvects()
    return sp.Matrix(m).eigenvects()


def IdentityMatrix(n: int) -> Matrix:
    """
    n x n identity matrix.

    Args:
        n: Size of the matrix

    Returns:
        n x n identity matrix
    """
    return eye(n)


def DiagonalMatrix(lst: List) -> Matrix:
    """
    Create diagonal matrix from list.

    Args:
        lst: List of diagonal elements

    Returns:
        Diagonal matrix
    """
    return diag(*lst)


def ZeroMatrix(m: int, n: Optional[int] = None) -> Matrix:
    """
    m x n zero matrix (m x m if n not given).

    Args:
        m: Number of rows
        n: Number of columns (default: m)

    Returns:
        m x n zero matrix
    """
    if n is None:
        n = m
    return zeros(m, n)


def Tr(m: Any) -> Any:
    """
    Trace of a matrix.

    Tr[m] - sum of diagonal elements

    Args:
        m: Matrix

    Returns:
        Trace

    Examples:
        >>> Tr(Matrix([[1, 2], [3, 4]]))
        5
    """
    if isinstance(m, Matrix):
        return m.trace()
    return sp.Matrix(m).trace()


def MatrixRank(m: Any) -> int:
    """
    Rank of a matrix.

    Args:
        m: Matrix

    Returns:
        Rank

    Examples:
        >>> MatrixRank(Matrix([[1, 2], [2, 4]]))
        1
    """
    if isinstance(m, Matrix):
        return m.rank()
    return sp.Matrix(m).rank()


def NullSpace(m: Any) -> List[Matrix]:
    """
    Null space (kernel) of a matrix.

    Args:
        m: Matrix

    Returns:
        List of basis vectors for the null space

    Examples:
        >>> NullSpace(Matrix([[1, 2], [2, 4]]))
        [Matrix([[-2], [1]])]
    """
    if isinstance(m, Matrix):
        return m.nullspace()
    return sp.Matrix(m).nullspace()


def RowReduce(m: Any) -> Matrix:
    """
    Row reduce to echelon form.

    Args:
        m: Matrix

    Returns:
        Row reduced matrix

    Examples:
        >>> RowReduce(Matrix([[1, 2, 3], [4, 5, 6]]))
    """
    if isinstance(m, Matrix):
        return m.rref()[0]
    return sp.Matrix(m).rref()[0]


def ConjugateTranspose(m: Any) -> Matrix:
    """
    Conjugate transpose (Hermitian adjoint) of a matrix.

    Args:
        m: Matrix

    Returns:
        Hermitian conjugate

    Examples:
        >>> ConjugateTranspose(Matrix([[1, I], [2, 3]]))
    """
    if isinstance(m, Matrix):
        return m.H
    return sp.Matrix(m).H


def MatrixExp(m: Any) -> Matrix:
    """
    Matrix exponential exp(m).

    Args:
        m: Matrix

    Returns:
        Matrix exponential

    Examples:
        >>> MatrixExp(Matrix([[0, 1], [-1, 0]]))  # Rotation matrix
    """
    if isinstance(m, Matrix):
        return m.exp()
    return sp.Matrix(m).exp()


def CharacteristicPolynomial(m: Any, x: Optional[Symbol] = None) -> Any:
    """
    Characteristic polynomial of a matrix.

    Args:
        m: Matrix
        x: Variable for the polynomial (default: creates 'x')

    Returns:
        Characteristic polynomial

    Examples:
        >>> x = Symbol('x')
        >>> CharacteristicPolynomial(Matrix([[1, 2], [3, 4]]), x)
    """
    if x is None:
        x = sp.Symbol('x')
    if isinstance(m, Matrix):
        return m.charpoly(x).as_expr()
    return sp.Matrix(m).charpoly(x).as_expr()


__all__ = [
    'Matrix',
    'Dot',
    'Transpose',
    'Inverse',
    'Det',
    'Eigenvalues',
    'Eigenvectors',
    'IdentityMatrix',
    'DiagonalMatrix',
    'ZeroMatrix',
    'Tr',
    'MatrixRank',
    'NullSpace',
    'RowReduce',
    'ConjugateTranspose',
    'MatrixExp',
    'CharacteristicPolynomial',
]
