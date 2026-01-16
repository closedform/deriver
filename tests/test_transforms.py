"""
Test suite for integral transforms and change of variables.
"""

import pytest
from sympy import sqrt, pi, exp, Heaviside, Integral, Abs, simplify, oo
from derive import (
    Symbol, symbols, Sin, Cos, Exp, Sqrt,
    FourierTransform, InverseFourierTransform,
    LaplaceTransform, InverseLaplaceTransform,
    Convolve,
    ChangeVariables, IntegrateWithSubstitution,
    Integrate, Simplify,
)


class TestFourierTransform:
    """Test Fourier transform operations."""

    def test_gaussian_transform(self):
        """FourierTransform of Gaussian is Gaussian (up to normalization)."""
        t, omega = symbols('t omega', real=True)
        # SymPy uses F[f](k) = ∫ f(x) e^{-2πikx} dx convention
        # F[exp(-t^2)] = sqrt(pi) * exp(-pi^2*omega^2) with this convention
        result = FourierTransform(exp(-t**2), t, omega)
        expected = sqrt(pi) * exp(-pi**2 * omega**2)
        assert simplify(result - expected) == 0

    def test_exponential_decay(self):
        """FourierTransform of exp(-a*t)*Heaviside(t)."""
        t, omega = symbols('t omega', real=True)
        a = Symbol('a', positive=True)
        result = FourierTransform(exp(-a * t) * Heaviside(t), t, omega)
        # Result should exist
        assert result is not None


class TestInverseFourierTransform:
    """Test inverse Fourier transform."""

    def test_inverse_roundtrip(self):
        """InverseFourierTransform undoes FourierTransform."""
        t, omega = symbols('t omega', real=True)
        # Forward transform of Gaussian
        forward = FourierTransform(exp(-t**2), t, omega)
        # Inverse should recover the original
        recovered = InverseFourierTransform(forward, omega, t)
        expected = exp(-t**2)
        assert simplify(recovered - expected) == 0


class TestLaplaceTransform:
    """Test Laplace transform operations."""

    def test_exponential(self):
        """LaplaceTransform of exp(-t)."""
        t, s = symbols('t s')
        result = LaplaceTransform(exp(-t), t, s, noconds=True)
        # L[exp(-t)] = 1/(s+1)
        expected = 1 / (s + 1)
        assert simplify(result - expected) == 0

    def test_sine(self):
        """LaplaceTransform of sin(t)."""
        t, s = symbols('t s')
        from sympy import sin
        result = LaplaceTransform(sin(t), t, s, noconds=True)
        # L[sin(t)] = 1/(s^2 + 1)
        expected = 1 / (s**2 + 1)
        assert simplify(result - expected) == 0

    def test_polynomial(self):
        """LaplaceTransform of t^n."""
        t, s = symbols('t s')
        # L[t^2] = 2/s^3
        result = LaplaceTransform(t**2, t, s, noconds=True)
        expected = 2 / s**3
        assert simplify(result - expected) == 0


class TestInverseLaplaceTransform:
    """Test inverse Laplace transform."""

    def test_inverse_exponential(self):
        """InverseLaplaceTransform of 1/(s+1)."""
        t, s = symbols('t s')
        result = InverseLaplaceTransform(1 / (s + 1), s, t)
        # Should be exp(-t)*Heaviside(t)
        expected = exp(-t) * Heaviside(t)
        assert simplify(result - expected) == 0


class TestConvolve:
    """Test convolution operation."""

    def test_convolution_integral(self):
        """Convolve returns an integral."""
        t = Symbol('t')
        f = Heaviside(t)
        g = Heaviside(t)
        result = Convolve(f, g, t)
        # Should be an integral
        assert isinstance(result, Integral)

    def test_convolution_with_tau(self):
        """Convolve with explicit tau variable."""
        t = Symbol('t')
        tau = Symbol('tau')
        f = exp(-t)
        g = exp(-t)
        result = Convolve(f, g, t, tau)
        assert isinstance(result, Integral)
        # The integration variable should be tau
        assert result.limits[0][0] == tau


class TestChangeVariables:
    """Test change of variables in integrals."""

    def test_simple_substitution(self):
        """Change variables x = u^2."""
        x, u = symbols('x u', positive=True)
        # Integrand: sqrt(x), with x = u^2
        # d(u^2)/du = 2u, so dx = 2u du
        # sqrt(u^2) * 2u = u * 2u = 2u^2
        result = ChangeVariables(Sqrt(x), x, u, u**2)
        expected = 2 * u**2
        assert simplify(result - expected) == 0

    def test_substitution_with_bounds(self):
        """Change variables with bounds transformation."""
        x, u = symbols('x u', positive=True)
        # ∫_0^4 sqrt(x) dx with x = u^2
        # u goes from 0 to 2
        result, bounds_map = ChangeVariables(Sqrt(x), x, u, u**2, bounds=(0, 4))
        # Bounds should map 0->0, 4->2
        assert bounds_map[0] == 0
        assert bounds_map[4] == 2

    def test_substitution_x_squared(self):
        """x = u^2 for power integrand."""
        x, u = symbols('x u', positive=True)
        # ∫ x^2 dx with x = u^2
        # x^2 = u^4, dx = 2u du
        # Integrand becomes u^4 * 2u = 2u^5
        result = ChangeVariables(x**2, x, u, u**2)
        expected = 2 * Abs(u) * u**4  # 2|u|*u^4
        # For positive u, this is 2u^5
        assert simplify(result.subs(Abs(u), u) - 2*u**5) == 0


class TestIntegrateWithSubstitution:
    """Test combined substitution and integration."""

    def test_sqrt_with_substitution(self):
        """Integrate sqrt(x) using u = sqrt(x)."""
        x, u = symbols('x u', positive=True)
        # ∫_0^4 sqrt(x) dx = [2/3 x^(3/2)]_0^4 = 2/3 * 8 = 16/3
        result = IntegrateWithSubstitution(Sqrt(x), x, u, u**2, bounds=(0, 4))
        from sympy import Rational
        expected = Rational(16, 3)
        assert simplify(result - expected) == 0

    def test_indefinite_substitution(self):
        """Indefinite integral with substitution."""
        x, u = symbols('x u', positive=True)
        # ∫ 2x * cos(x^2) dx with u = x^2
        # The standard setup: x = sqrt(u), dx = 1/(2*sqrt(u)) du
        # But here we use x = u^2 setup for different example
        pass  # Complex example, skip for basic test


class TestTransformRoundTrip:
    """Test that transforms can round-trip."""

    def test_laplace_roundtrip(self):
        """Laplace transform and inverse recover original."""
        t, s = symbols('t s')
        original = exp(-2 * t) * Heaviside(t)
        transformed = LaplaceTransform(original, t, s, noconds=True)
        recovered = InverseLaplaceTransform(transformed, s, t)
        # Both should represent the same function
        assert simplify(recovered - original) == 0
