"""
Standard test cases for common symbolic math operations.

These tests verify that derive functions produce correct results
for typical symbolic computation examples.
"""

import pytest
from derive import *


class TestIntegrate:
    """Tests for Integrate function."""

    def test_integrate_x_squared(self):
        """Integrate[x^2, x] -> x^3/3"""
        x = Symbol("x")
        result = Integrate(x**2, x)
        assert result == x**3/3

    def test_integrate_sin(self):
        """Integrate[Sin[x], x] -> -Cos[x]"""
        x = Symbol("x")
        result = Integrate(Sin(x), x)
        assert result == -Cos(x)

    def test_integrate_definite(self):
        """Integrate[x, {x, 0, 1}] -> 1/2"""
        x = Symbol("x")
        result = Integrate(x, (x, 0, 1))
        from sympy import Rational
        assert result == Rational(1, 2)

    def test_integrate_gaussian(self):
        """Integrate[Exp[-x^2], {x, -Infinity, Infinity}] -> Sqrt[Pi]"""
        x = Symbol("x")
        result = Integrate(Exp(-x**2), (x, -Infinity, Infinity))
        assert result == Sqrt(Pi)


class TestD:
    """Tests for D (differentiation) function."""

    def test_d_x_cubed(self):
        """D[x^3, x] -> 3*x^2"""
        x = Symbol("x")
        result = D(x**3, x)
        assert result == 3*x**2

    def test_d_sin(self):
        """D[Sin[x], x] -> Cos[x]"""
        x = Symbol("x")
        result = D(Sin(x), x)
        assert result == Cos(x)

    def test_d_second_derivative(self):
        """D[x^5, {x, 2}] -> 20*x^3"""
        x = Symbol("x")
        result = D(x**5, (x, 2))
        assert result == 20*x**3


class TestSolve:
    """Tests for Solve function."""

    def test_solve_quadratic(self):
        """Solve[x^2 - 4 == 0, x] -> x = -2, 2"""
        x = Symbol("x")
        result = Solve(x**2 - 4, x)
        solutions = [sol[x] for sol in result]
        assert -2 in solutions
        assert 2 in solutions

    def test_solve_perfect_square(self):
        """Solve[x^2 + 2*x + 1 == 0, x] -> x = -1"""
        x = Symbol("x")
        result = Solve(x**2 + 2*x + 1, x)
        solutions = [sol[x] for sol in result]
        assert -1 in solutions


class TestSum:
    """Tests for Sum function."""

    def test_sum_1_to_10(self):
        """Sum[i, {i, 1, 10}] -> 55"""
        i = Symbol("i")
        result = Sum(i, (i, 1, 10))
        assert result == 55

    def test_sum_squares_symbolic(self):
        """Sum[i^2, {i, 1, n}] -> n*(n+1)*(2*n+1)/6"""
        i, n = symbols("i n")
        result = Sum(i**2, (i, 1, n))
        from sympy import simplify
        expected = n*(n+1)*(2*n+1)/6
        assert simplify(result - expected) == 0


class TestSeries:
    """Tests for Series function."""

    def test_series_exp(self):
        """Series[Exp[x], {x, 0, 5}] - Taylor expansion of e^x"""
        x = Symbol("x")
        result = Series(Exp(x), (x, 0, 5))
        # Check the first few terms
        truncated = result.removeO()
        assert truncated.coeff(x, 0) == 1  # constant term
        assert truncated.coeff(x, 1) == 1  # x term
        assert truncated.coeff(x, 2) == Rational(1, 2)  # x^2/2 term

    def test_series_sin(self):
        """Series[Sin[x], {x, 0, 5}] - Taylor expansion of sin(x)"""
        x = Symbol("x")
        result = Series(Sin(x), (x, 0, 5))
        truncated = result.removeO()
        assert truncated.coeff(x, 0) == 0  # no constant
        assert truncated.coeff(x, 1) == 1  # x term
        assert truncated.coeff(x, 3) == Rational(-1, 6)  # -x^3/6


class TestLimit:
    """Tests for Limit function."""

    def test_limit_sinc(self):
        """Limit[Sin[x]/x, x -> 0] -> 1"""
        x = Symbol("x")
        result = Limit(Sin(x)/x, x, 0)
        assert result == 1

    def test_limit_e_definition(self):
        """Limit[(1 + 1/n)^n, n -> Infinity] -> e"""
        n = Symbol("n")
        result = Limit((1 + 1/n)**n, n, Infinity)
        assert result == E


class TestDSolve:
    """Tests for DSolve function."""

    def test_dsolve_exponential(self):
        """DSolve[y'[x] == y[x], y[x], x] -> y[x] = C1*e^x"""
        x = Symbol("x")
        y = Function("y")
        from sympy import Eq, exp
        eq = Eq(y(x).diff(x), y(x))
        result = DSolve(eq, y(x), x)
        # Result contains exp(x)
        assert result.has(exp(x))

    def test_dsolve_harmonic(self):
        """DSolve[y''[x] + y[x] == 0, y[x], x] -> trig solution"""
        x = Symbol("x")
        y = Function("y")
        from sympy import Eq, sin, cos
        eq = Eq(y(x).diff(x, 2) + y(x), 0)
        result = DSolve(eq, y(x), x)
        # Result contains sin and cos
        assert result.has(sin(x)) or result.has(cos(x))


class TestEigenvalues:
    """Tests for Eigenvalues function."""

    def test_eigenvalues_symmetric(self):
        """Eigenvalues[{{1, 2}, {2, 1}}] -> {3, -1}"""
        m = Matrix([[1, 2], [2, 1]])
        result = Eigenvalues(m)
        # Result is dict {eigenvalue: multiplicity}
        assert 3 in result
        assert -1 in result


class TestDet:
    """Tests for Det function."""

    def test_det_2x2(self):
        """Det[{{1, 2}, {3, 4}}] -> -2"""
        m = Matrix([[1, 2], [3, 4]])
        result = Det(m)
        assert result == -2


class TestInverse:
    """Tests for Inverse function."""

    def test_inverse_2x2(self):
        """Inverse[{{1, 2}, {3, 4}}]"""
        m = Matrix([[1, 2], [3, 4]])
        result = Inverse(m)
        # m * inverse = identity
        product = m * result
        assert product == IdentityMatrix(2)
