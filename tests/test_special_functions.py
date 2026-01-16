"""
Tests for special functions (Bessel, Legendre, Spherical harmonics, etc.)

These functions are important for physics and differential geometry applications.
"""
import pytest
from sympy import Symbol, symbols, simplify, N as numerical_eval
from sympy import pi, oo, E, I, Rational, sqrt, sin, cos, exp
from derive import *


x, y, z, n, m = symbols('x y z n m')


class TestBesselFunctions:
    """Tests for Bessel functions."""

    def test_besselj_0_0(self):
        """BesselJ[0, 0] = 1"""
        result = BesselJ(0, 0)
        assert result == 1

    def test_besselj_numeric(self):
        """BesselJ[0, 1] numerical value."""
        result = float(BesselJ(0, 1).evalf())
        # J_0(1) ≈ 0.7651976865579666
        assert abs(result - 0.7651976865579666) < 1e-10

    def test_bessely_numeric(self):
        """BesselY[0, 1] numerical value."""
        result = float(BesselY(0, 1).evalf())
        # Y_0(1) ≈ 0.0882569642156769
        assert abs(result - 0.0882569642156769) < 1e-10

    def test_besseli_0_0(self):
        """BesselI[0, 0] = 1"""
        result = BesselI(0, 0)
        assert result == 1

    def test_hankel_relation(self):
        """H1 + H2 = 2*BesselJ (for real x), verified numerically."""
        # SymPy doesn't auto-simplify this identity, so verify numerically
        val = 1.5  # Test at a specific point
        result = complex(HankelH1(0, val).evalf() + HankelH2(0, val).evalf())
        expected = 2 * float(BesselJ(0, val).evalf())
        assert abs(result.real - expected) < 1e-10
        assert abs(result.imag) < 1e-10  # Should be real for real x


class TestLegendrePolynomials:
    """Tests for Legendre polynomials."""

    def test_legendre_0(self):
        """P_0(x) = 1"""
        result = LegendreP(0, x)
        assert result == 1

    def test_legendre_1(self):
        """P_1(x) = x"""
        result = LegendreP(1, x)
        assert result == x

    def test_legendre_2(self):
        """P_2(x) = (3x^2 - 1)/2"""
        result = LegendreP(2, x)
        expected = (3*x**2 - 1) / 2
        assert simplify(result - expected) == 0

    def test_associated_legendre(self):
        """Test associated Legendre P_1^1(x)."""
        result = AssociatedLegendreP(1, 1, x)
        # P_1^1(x) = -sqrt(1-x^2)
        expected = -sqrt(1 - x**2)
        assert simplify(result - expected) == 0


class TestChebyshevPolynomials:
    """Tests for Chebyshev polynomials."""

    def test_chebyshev_t_0(self):
        """T_0(x) = 1"""
        result = ChebyshevT(0, x)
        assert result == 1

    def test_chebyshev_t_1(self):
        """T_1(x) = x"""
        result = ChebyshevT(1, x)
        assert result == x

    def test_chebyshev_u_0(self):
        """U_0(x) = 1"""
        result = ChebyshevU(0, x)
        assert result == 1


class TestHermitePolynomials:
    """Tests for Hermite polynomials (quantum mechanics)."""

    def test_hermite_0(self):
        """H_0(x) = 1"""
        result = HermiteH(0, x)
        assert result == 1

    def test_hermite_1(self):
        """H_1(x) = 2x"""
        result = HermiteH(1, x)
        assert result == 2*x


class TestLaguerrePolynomials:
    """Tests for Laguerre polynomials (quantum mechanics)."""

    def test_laguerre_0(self):
        """L_0(x) = 1"""
        result = LaguerreL(0, x)
        assert result == 1

    def test_laguerre_1(self):
        """L_1(x) = -x + 1"""
        result = LaguerreL(1, x)
        expected = -x + 1
        assert simplify(result - expected) == 0


class TestEllipticIntegrals:
    """Tests for elliptic integrals (classical mechanics, string theory)."""

    def test_elliptic_k_0(self):
        """K(0) = pi/2"""
        result = EllipticK(0)
        expected = pi / 2
        assert simplify(result - expected) == 0

    def test_elliptic_e_0(self):
        """E(0) = pi/2"""
        result = EllipticE(0)
        expected = pi / 2
        assert simplify(result - expected) == 0


class TestErrorFunctions:
    """Tests for error functions."""

    def test_erf_0(self):
        """erf(0) = 0"""
        result = Erf(0)
        assert result == 0

    def test_erfc_0(self):
        """erfc(0) = 1"""
        result = Erfc(0)
        assert result == 1

    def test_erf_plus_erfc(self):
        """erf(x) + erfc(x) = 1, verified numerically."""
        # SymPy doesn't auto-simplify this identity
        for val in [0.5, 1.0, 2.0]:
            result = float((Erf(val) + Erfc(val)).evalf())
            assert abs(result - 1.0) < 1e-10


class TestAiryFunctions:
    """Tests for Airy functions (quantum mechanics, optics)."""

    def test_airy_ai_0(self):
        """Ai(0) = 1/(3^(2/3)*Gamma(2/3))"""
        result = float(AiryAi(0).evalf())
        # Ai(0) ≈ 0.35502805388781724
        assert abs(result - 0.35502805388781724) < 1e-10


class TestZetaFunction:
    """Tests for Riemann zeta function (QFT, string theory)."""

    def test_zeta_2(self):
        """zeta(2) = pi^2/6"""
        result = Zeta(2)
        expected = pi**2 / 6
        assert simplify(result - expected) == 0

    def test_zeta_4(self):
        """zeta(4) = pi^4/90"""
        result = Zeta(4)
        expected = pi**4 / 90
        assert simplify(result - expected) == 0


class TestHypergeometricFunctions:
    """Tests for hypergeometric functions."""

    def test_hypergeometric_exp(self):
        """Exp is a special case of hypergeometric."""
        # 0F0(z) = exp(z), but we test a simpler case
        result = HypergeometricPFQ([], [], x)
        expected = exp(x)
        assert simplify(result - expected) == 0


class TestNDSolveOptions:
    """Tests for extended NDSolve options."""

    def test_ndsolve_bdf_method(self):
        """Test NDSolve with BDF method (stiff solver)."""
        sol = NDSolve(lambda t, y: -y, 1.0, ('t', 0, 5), Method='BDF')
        assert sol.success
        # y(5) ≈ exp(-5)
        assert abs(sol.y[0][-1] - exp(-5).evalf()) < 0.01

    def test_ndsolve_radau_method(self):
        """Test NDSolve with implicit Runge-Kutta (Radau)."""
        sol = NDSolve(lambda t, y: -y, 1.0, ('t', 0, 5), Method='ImplicitRungeKutta')
        assert sol.success
        assert abs(sol.y[0][-1] - exp(-5).evalf()) < 0.01

    def test_ndsolve_accuracy_goal(self):
        """Test NDSolve with AccuracyGoal option."""
        sol = NDSolve(lambda t, y: -y, 1.0, ('t', 0, 5), AccuracyGoal=10)
        assert sol.success

    def test_ndsolve_max_step_size(self):
        """Test NDSolve with MaxStepSize option."""
        sol = NDSolve(lambda t, y: -y, 1.0, ('t', 0, 5), MaxStepSize=0.1)
        assert sol.success
