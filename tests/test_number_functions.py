"""
Tests for number manipulation functions.
"""
import pytest
from sympy import Symbol, symbols, pi, sqrt, Rational
from derive import *


class TestPower:
    """Tests for Power function."""

    def test_power_integer(self):
        """Power[2, 3] = 8"""
        assert Power(2, 3) == 8

    def test_power_symbolic(self):
        """Power[x, 2] = x^2"""
        x = Symbol('x')
        assert Power(x, 2) == x**2


class TestN:
    """Tests for numerical evaluation."""

    def test_n_pi(self):
        """N[Pi] returns numeric pi."""
        result = N(Pi)
        assert abs(float(result) - 3.14159265358979) < 1e-10

    def test_n_sqrt2(self):
        """N[Sqrt[2]] = 1.414..."""
        result = N(Sqrt(2))
        assert abs(float(result) - 1.4142135623730951) < 1e-10


class TestRound:
    """Tests for Round function."""

    def test_round_integer(self):
        """Round[3.7] = 4"""
        assert Round(3.7) == 4

    def test_round_step(self):
        """Round[3.14159, 0.01] = 3.14"""
        assert abs(Round(3.14159, 0.01) - 3.14) < 1e-10


class TestMod:
    """Tests for Mod function."""

    def test_mod_basic(self):
        """Mod[17, 5] = 2"""
        result = Mod(17, 5)
        assert result == 2


class TestGCD:
    """Tests for GCD function."""

    def test_gcd_basic(self):
        """GCD[12, 18] = 6"""
        assert GCD(12, 18) == 6

    def test_gcd_multiple(self):
        """GCD[12, 18, 24] = 6"""
        assert GCD(12, 18, 24) == 6


class TestLCM:
    """Tests for LCM function."""

    def test_lcm_basic(self):
        """LCM[4, 6] = 12"""
        assert LCM(4, 6) == 12

    def test_lcm_multiple(self):
        """LCM[4, 6, 8] = 24"""
        assert LCM(4, 6, 8) == 24


class TestPrimeQ:
    """Tests for PrimeQ function."""

    def test_primeq_true(self):
        """PrimeQ[7] = True"""
        assert PrimeQ(7) == True

    def test_primeq_false(self):
        """PrimeQ[8] = False"""
        assert PrimeQ(8) == False


class TestPrime:
    """Tests for Prime function."""

    def test_prime_1(self):
        """Prime[1] = 2"""
        assert Prime(1) == 2

    def test_prime_10(self):
        """Prime[10] = 29"""
        assert Prime(10) == 29


class TestFactorInteger:
    """Tests for FactorInteger function."""

    def test_factor_integer_60(self):
        """FactorInteger[60] = {{2, 2}, {3, 1}, {5, 1}}"""
        result = FactorInteger(60)
        assert result == [[2, 2], [3, 1], [5, 1]]


class TestMatrixFunctions:
    """Tests for new matrix functions."""

    def test_tr_basic(self):
        """Tr[{{1, 2}, {3, 4}}] = 5"""
        m = Matrix([[1, 2], [3, 4]])
        assert Tr(m) == 5

    def test_matrix_rank(self):
        """MatrixRank of rank-1 matrix."""
        m = Matrix([[1, 2], [2, 4]])
        assert MatrixRank(m) == 1

    def test_null_space(self):
        """NullSpace of rank-deficient matrix."""
        m = Matrix([[1, 2], [2, 4]])
        ns = NullSpace(m)
        assert len(ns) == 1

    def test_row_reduce(self):
        """RowReduce produces echelon form."""
        m = Matrix([[1, 2], [3, 4]])
        result = RowReduce(m)
        # Should be in reduced row echelon form
        assert result[0, 0] == 1
        assert result[1, 1] == 1

    def test_characteristic_polynomial(self):
        """CharacteristicPolynomial of 2x2 matrix."""
        x = Symbol('x')
        m = Matrix([[1, 2], [3, 4]])
        poly = CharacteristicPolynomial(m, x)
        # lambda^2 - 5*lambda - 2
        # Roots are eigenvalues
        from sympy import solve
        roots = solve(poly, x)
        eigenvals = list(Eigenvalues(m).keys())
        for root in roots:
            assert any(abs(float(root) - float(ev)) < 1e-10 for ev in eigenvals)
