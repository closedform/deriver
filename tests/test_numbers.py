"""
Tests for Smart Number Handling (Phase 13).

Tests rational preservation, context managers, expr() parser,
and N() with multiple backends.
"""

import pytest
from derive import (
    Symbol, Rational, Integer, Float, Pi, Sqrt,
    rationalize, to_rational, exact, numerical,
    is_exact_mode, is_numerical_mode, get_numerical_precision,
    auto_rational, SmartNumber, S, expr,
    float_to_rational, ensure_rational, N,
)


class TestRationalize:
    """Tests for the rationalize function."""

    def test_rationalize_integer(self):
        """Integers should become Integer."""
        result = rationalize(5)
        assert result == Integer(5)

    def test_rationalize_half(self):
        """0.5 should become 1/2."""
        result = rationalize(0.5)
        assert result == Rational(1, 2)

    def test_rationalize_third(self):
        """0.333... should become 1/3."""
        result = rationalize(1/3)
        assert result == Rational(1, 3)

    def test_rationalize_quarter(self):
        """0.25 should become 1/4."""
        result = rationalize(0.25)
        assert result == Rational(1, 4)

    def test_rationalize_zero(self):
        """0.0 should become Integer(0)."""
        result = rationalize(0.0)
        assert result == Integer(0)

    def test_rationalize_already_rational(self):
        """Rationals should pass through unchanged."""
        rat = Rational(3, 7)
        result = rationalize(rat)
        assert result == rat

    def test_rationalize_expression_with_floats(self):
        """Expressions containing floats should have floats rationalized."""
        x = Symbol('x')
        expr = 0.5 * x
        result = rationalize(expr)
        # Should be x/2
        assert result == x / 2


class TestToRational:
    """Tests for the to_rational function."""

    def test_to_rational_basic(self):
        """Basic rational creation."""
        assert to_rational(1, 3) == Rational(1, 3)

    def test_to_rational_integer(self):
        """Creating integer."""
        assert to_rational(5) == Rational(5, 1)

    def test_to_rational_simplifies(self):
        """Rationals should be simplified."""
        assert to_rational(4, 8) == Rational(1, 2)


class TestExactContextManager:
    """Tests for the exact() context manager."""

    def test_exact_mode_flag(self):
        """exact() should set the mode flag."""
        assert not is_exact_mode()
        with exact():
            assert is_exact_mode()
        assert not is_exact_mode()

    def test_exact_rational_arithmetic(self):
        """Rational arithmetic in exact mode."""
        with exact():
            result = Rational(1, 3) + Rational(1, 4)
            assert result == Rational(7, 12)


class TestNumericalContextManager:
    """Tests for the numerical() context manager."""

    def test_numerical_mode_flag(self):
        """numerical() should set the mode flag."""
        assert not is_numerical_mode()
        with numerical(50):
            assert is_numerical_mode()
            assert get_numerical_precision() == 50
        assert not is_numerical_mode()

    def test_numerical_default_precision(self):
        """Default precision should be 15."""
        with numerical():
            assert get_numerical_precision() == 15


class TestSmartNumber:
    """Tests for the SmartNumber class."""

    def test_smart_number_from_int(self):
        """SmartNumber from int should be Integer."""
        n = SmartNumber(5)
        assert n.value == Integer(5)

    def test_smart_number_from_float(self):
        """SmartNumber from float should rationalize."""
        n = SmartNumber(0.5)
        assert n.value == Rational(1, 2)

    def test_smart_number_division(self):
        """Division of SmartNumbers should create Rational."""
        a = SmartNumber(1)
        b = SmartNumber(3)
        result = a / b
        assert result.value == Rational(1, 3)

    def test_smart_number_addition(self):
        """Addition of SmartNumbers."""
        a = SmartNumber(1)
        b = SmartNumber(2)
        result = a + b
        assert result.value == Integer(3)

    def test_smart_number_multiplication(self):
        """Multiplication of SmartNumbers."""
        a = SmartNumber(Rational(1, 2))
        b = SmartNumber(Rational(2, 3))
        result = a * b
        assert result.value == Rational(1, 3)

    def test_smart_number_comparison(self):
        """Comparison of SmartNumbers."""
        a = SmartNumber(Rational(1, 2))
        b = SmartNumber(0.5)
        assert a == b

    def test_smart_number_power(self):
        """Power of SmartNumbers."""
        a = SmartNumber(2)
        b = SmartNumber(3)
        result = a ** b
        assert result.value == Integer(8)


class TestSFunction:
    """Tests for the S() shorthand function."""

    def test_s_creates_smart_number(self):
        """S() should create SmartNumber."""
        n = S(5)
        assert isinstance(n, SmartNumber)

    def test_s_division(self):
        """S(1) / S(3) should give 1/3."""
        result = S(1) / S(3)
        assert result.value == Rational(1, 3)


class TestExprParser:
    """Tests for the expr() string parser."""

    def test_expr_integer_division(self):
        """Integer division should become Rational."""
        result = expr("1/3 + 1/4")
        assert result == Rational(7, 12)

    def test_expr_with_symbols(self):
        """Expression with symbols."""
        result = expr("x^2 + 3*x + 1")
        x = Symbol('x')
        assert result == x**2 + 3*x + 1

    def test_expr_symbol_division(self):
        """Symbol division should stay as division."""
        result = expr("x/3")
        x = Symbol('x')
        assert result == x / 3

    def test_expr_trig_functions(self):
        """Trig functions should be recognized."""
        result = expr("sin(x)")
        x = Symbol('x')
        from sympy import sin
        assert result == sin(x)

    def test_expr_pi(self):
        """Pi should be recognized."""
        from sympy import pi
        result = expr("pi")
        assert result == pi

    def test_expr_sqrt(self):
        """sqrt should be recognized."""
        from sympy import sqrt
        result = expr("sqrt(2)")
        assert result == sqrt(2)

    def test_expr_complex_expression(self):
        """Complex expression with multiple features."""
        result = expr("1/2 * x^2 + sin(x)")
        x = Symbol('x')
        from sympy import sin
        expected = Rational(1, 2) * x**2 + sin(x)
        assert result == expected


class TestFloatToRational:
    """Tests for float_to_rational function."""

    def test_float_to_rational_half(self):
        """0.5 -> 1/2"""
        result = float_to_rational(0.5)
        assert result == Rational(1, 2)

    def test_float_to_rational_third(self):
        """0.333... -> 1/3"""
        result = float_to_rational(1/3)
        assert result == Rational(1, 3)

    def test_float_to_rational_zero(self):
        """0.0 -> 0"""
        result = float_to_rational(0.0)
        assert result == Integer(0)

    def test_float_to_rational_non_simple(self):
        """Non-simple fractions return None."""
        result = float_to_rational(3.14159265358979)
        # Pi is not a simple fraction
        assert result is None


class TestEnsureRational:
    """Tests for ensure_rational function."""

    def test_ensure_rational_integer(self):
        """int -> Integer"""
        result = ensure_rational(5)
        assert result == Integer(5)

    def test_ensure_rational_float_simple(self):
        """Simple float -> Rational"""
        result = ensure_rational(0.25)
        assert result == Rational(1, 4)

    def test_ensure_rational_passthrough_rational(self):
        """Rational passes through."""
        rat = Rational(3, 7)
        result = ensure_rational(rat)
        assert result == rat


class TestNFunction:
    """Tests for the enhanced N() function."""

    def test_n_default(self):
        """N with default precision."""
        result = N(Pi)
        # Should be approximately 3.14159...
        assert abs(float(result) - 3.14159265358979) < 1e-10

    def test_n_high_precision(self):
        """N with higher precision."""
        result = N(Pi, 30)
        # Check we get a reasonable approximation
        assert '3.14159265358979' in str(result)

    def test_n_sympy_method(self):
        """N with explicit sympy method."""
        result = N(Pi, 15, method='sympy')
        assert abs(float(result) - 3.14159265358979) < 1e-10

    def test_n_mpfr_method(self):
        """N with mpfr method for arbitrary precision."""
        result = N(Pi, 30, method='mpfr')
        # mpfr should give us an mpf object
        import mpmath
        assert isinstance(result, mpmath.mpf)

    def test_n_interval_method(self):
        """N with interval method for rigorous bounds."""
        result = N(Pi, 15, method='interval')
        import mpmath
        # Should return an interval (mpi returns an iv.mpf type)
        assert hasattr(result, 'a') and hasattr(result, 'b')  # interval has lower and upper bounds

    def test_n_floating_method(self):
        """N with floating method for standard float."""
        result = N(Pi, 15, method='floating')
        assert isinstance(result, float)

    def test_n_invalid_method(self):
        """N with invalid method raises error."""
        with pytest.raises(ValueError):
            N(Pi, 15, method='invalid')


class TestIntegrationScenarios:
    """Integration tests for typical use cases."""

    def test_rational_in_expression(self):
        """Rationals should work naturally in expressions."""
        x = Symbol('x')
        expr_val = x / 3 + Rational(1, 2)
        # Should simplify to (2x + 3) / 6
        from sympy import simplify
        assert simplify(expr_val - (2*x + 3)/6) == 0

    def test_smart_calculation_chain(self):
        """Chain of SmartNumber calculations."""
        a = S(1) / S(3)
        b = S(1) / S(4)
        result = (a + b).value
        assert result == Rational(7, 12)

    def test_expr_then_evaluate(self):
        """Parse expression then evaluate numerically."""
        parsed = expr("1/3 + 1/4")
        numerical_result = N(parsed, 10)
        assert abs(float(numerical_result) - 0.5833333333) < 1e-8

    def test_symbolic_with_exact_rational(self):
        """Symbolic computation with exact rationals."""
        x = Symbol('x')
        # Create expression with rationals
        from derive import Integrate
        # Integrate x/3 from 0 to 1 = 1/6
        result = Integrate(x/3, (x, 0, 1))
        assert result == Rational(1, 6)


class TestEdgeCases:
    """Edge case tests."""

    def test_rationalize_negative(self):
        """Negative fractions."""
        result = rationalize(-0.5)
        assert result == Rational(-1, 2)

    def test_smart_number_negative_division(self):
        """Negative SmartNumber division."""
        result = S(-1) / S(3)
        assert result.value == Rational(-1, 3)

    def test_expr_negative_fraction(self):
        """expr with negative fraction."""
        result = expr("-1/3")
        assert result == Rational(-1, 3)

    def test_expr_multiple_fractions(self):
        """expr with multiple fractions."""
        result = expr("1/2 + 2/3 - 1/4")
        # 6/12 + 8/12 - 3/12 = 11/12
        assert result == Rational(11, 12)

    def test_nested_context_managers(self):
        """Nested context managers."""
        with exact():
            assert is_exact_mode()
            with numerical(30):
                assert is_numerical_mode()
                assert get_numerical_precision() == 30
            assert not is_numerical_mode()
        assert not is_exact_mode()
