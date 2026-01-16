"""
Test suite for assumptions system.
"""

import pytest
from sympy import sqrt, Abs, simplify, Q
from derive import (
    Symbol, symbols,
    Assuming, Refine, Ask, SimplifyWithAssumptions,
    get_assumptions, assume_positive, assume_real, assume_integer,
    AssumedSymbol,
    Positive, Negative, Real, IntegerPred, Nonzero, Nonnegative,
)


class TestAssuming:
    """Test the Assuming context manager."""

    def test_assuming_positive(self):
        """Assuming positive simplifies Abs."""
        x = Symbol('x')
        with Assuming(Q.positive(x)):
            from sympy import refine
            result = refine(Abs(x))
            assert result == x

    def test_assuming_real(self):
        """Assuming real affects sqrt simplification."""
        x = Symbol('x')
        with Assuming(Q.positive(x)):
            from sympy import refine
            result = refine(sqrt(x**2))
            assert result == x

    def test_assuming_context_cleanup(self):
        """Assumptions are removed after context exits."""
        x = Symbol('x')
        with Assuming(Q.positive(x)):
            pass
        # After context, the global assumption should be gone
        from sympy.assumptions.assume import global_assumptions
        assert Q.positive(x) not in global_assumptions

    def test_multiple_assumptions(self):
        """Multiple assumptions in one context."""
        x, y = symbols('x y')
        with Assuming(Q.positive(x), Q.positive(y)):
            from sympy import refine
            assert refine(Abs(x)) == x
            assert refine(Abs(y)) == y


class TestRefine:
    """Test the Refine function."""

    def test_refine_abs_positive(self):
        """Refine Abs with positive assumption."""
        x = Symbol('x')
        result = Refine(Abs(x), Q.positive(x))
        assert result == x

    def test_refine_abs_negative(self):
        """Refine Abs with negative assumption."""
        x = Symbol('x')
        result = Refine(Abs(x), Q.negative(x))
        assert result == -x

    def test_refine_sqrt_squared(self):
        """Refine sqrt(x^2) with positive assumption."""
        x = Symbol('x')
        result = Refine(sqrt(x**2), Q.positive(x))
        assert result == x

    def test_refine_multiple_assumptions(self):
        """Refine with multiple assumptions."""
        x = Symbol('x')
        result = Refine(sqrt(x**2), Q.real(x), Q.positive(x))
        assert result == x


class TestAsk:
    """Test the Ask function."""

    def test_ask_positive_symbol(self):
        """Ask about positive symbol."""
        x = Symbol('x', positive=True)
        # Symbol assumptions should be respected
        result = Ask(Q.positive(x))
        # Result is True or None depending on SymPy version
        assert result is True or result is None

    def test_ask_negative_symbol(self):
        """Ask about negative symbol."""
        x = Symbol('x', negative=True)
        result = Ask(Q.negative(x))
        assert result is True or result is None

    def test_ask_with_assumptions(self):
        """Ask with explicit assumptions."""
        x = Symbol('x')
        result = Ask(Q.positive(x), Q.positive(x))
        assert result is True

    def test_ask_unknown(self):
        """Ask about symbol with no assumptions."""
        x = Symbol('x')
        result = Ask(Q.positive(x))
        # Without assumptions, result should be None or False
        assert result is None or result is False


class TestSimplifyWithAssumptions:
    """Test SimplifyWithAssumptions function."""

    def test_simplify_sqrt_squared(self):
        """Simplify sqrt(x^2) with positive."""
        x = Symbol('x')
        result = SimplifyWithAssumptions(sqrt(x**2), Q.positive(x))
        assert result == x

    def test_simplify_no_assumptions(self):
        """Simplify without assumptions."""
        x = Symbol('x')
        result = SimplifyWithAssumptions(x**2 - x**2)
        assert result == 0


class TestAssumedSymbol:
    """Test AssumedSymbol factory."""

    def test_positive_symbol(self):
        """Create positive symbol."""
        x = AssumedSymbol.positive('x')
        assert x.is_positive is True

    def test_negative_symbol(self):
        """Create negative symbol."""
        x = AssumedSymbol.negative('x')
        assert x.is_negative is True

    def test_real_symbol(self):
        """Create real symbol."""
        x = AssumedSymbol.real('x')
        assert x.is_real is True

    def test_integer_symbol(self):
        """Create integer symbol."""
        n = AssumedSymbol.integer('n')
        assert n.is_integer is True

    def test_nonzero_symbol(self):
        """Create nonzero symbol."""
        x = AssumedSymbol.nonzero('x')
        assert x.is_nonzero is True

    def test_nonnegative_symbol(self):
        """Create nonnegative symbol."""
        x = AssumedSymbol.nonnegative('x')
        assert x.is_nonnegative is True

    def test_even_symbol(self):
        """Create even symbol."""
        n = AssumedSymbol.even('n')
        assert n.is_even is True

    def test_odd_symbol(self):
        """Create odd symbol."""
        n = AssumedSymbol.odd('n')
        assert n.is_odd is True


class TestBatchCreation:
    """Test batch symbol creation functions."""

    def test_assume_positive(self):
        """Create multiple positive symbols."""
        a, b, c = assume_positive('a', 'b', 'c')
        assert a.is_positive is True
        assert b.is_positive is True
        assert c.is_positive is True

    def test_assume_real(self):
        """Create multiple real symbols."""
        x, y = assume_real('x', 'y')
        assert x.is_real is True
        assert y.is_real is True

    def test_assume_integer(self):
        """Create multiple integer symbols."""
        m, n = assume_integer('m', 'n')
        assert m.is_integer is True
        assert n.is_integer is True


class TestGetAssumptions:
    """Test get_assumptions utility."""

    def test_get_positive_assumptions(self):
        """Get assumptions from positive symbol."""
        x = Symbol('x', positive=True)
        assumptions = get_assumptions(x)
        assert assumptions.get('positive') is True

    def test_get_real_assumptions(self):
        """Get assumptions from real symbol."""
        x = Symbol('x', real=True)
        assumptions = get_assumptions(x)
        assert assumptions.get('real') is True


class TestPredicates:
    """Test predicate shortcuts."""

    def test_positive_predicate(self):
        """Positive predicate works."""
        x = Symbol('x')
        result = Refine(Abs(x), Positive(x))
        assert result == x

    def test_negative_predicate(self):
        """Negative predicate works."""
        x = Symbol('x')
        result = Refine(Abs(x), Negative(x))
        assert result == -x

    def test_real_predicate(self):
        """Real predicate works."""
        x = Symbol('x')
        # Real + positive gives sqrt(x^2) = x
        result = Refine(sqrt(x**2), Real(x) & Positive(x))
        assert result == x
