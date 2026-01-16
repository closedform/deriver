"""Tests for lazy evaluation utilities."""

import pytest
from derive import (
    Symbol, Sin, Cos, Simplify, Expand,
    Lazy, LazyExpr, lazy, force,
)


class TestLazy:
    """Tests for the Lazy class."""

    def test_lazy_delays_evaluation(self):
        """Lazy should not evaluate until value is accessed."""
        counter = [0]

        def expensive():
            counter[0] += 1
            return 42

        result = Lazy(expensive)
        assert counter[0] == 0  # Not evaluated yet
        assert result.value == 42
        assert counter[0] == 1  # Now evaluated

    def test_lazy_caches_result(self):
        """Lazy should only evaluate once."""
        counter = [0]

        def expensive():
            counter[0] += 1
            return 42

        result = Lazy(expensive)
        _ = result.value
        _ = result.value
        assert counter[0] == 1  # Still only evaluated once

    def test_lazy_is_evaluated(self):
        """is_evaluated should track evaluation state."""
        result = Lazy(lambda: 42)
        assert not result.is_evaluated()
        _ = result.value
        assert result.is_evaluated()


class TestLazyExpr:
    """Tests for the LazyExpr class."""

    def test_lazy_expr_with_simplify(self):
        """LazyExpr with auto_simplify should simplify on access."""
        x = Symbol('x')
        expr = LazyExpr(Sin(x)**2 + Cos(x)**2, auto_simplify=True)
        assert expr.value == 1

    def test_lazy_expr_raw(self):
        """raw should return unsimplified expression."""
        x = Symbol('x')
        expr = LazyExpr(Sin(x)**2 + Cos(x)**2, auto_simplify=True)
        raw = expr.raw
        assert raw != 1  # Not simplified


class TestLazyDecorator:
    """Tests for the lazy decorator."""

    def test_lazy_decorator(self):
        """@lazy should make function return Lazy object."""
        @lazy
        def compute(n):
            return sum(range(n))

        result = compute(1000)
        assert isinstance(result, Lazy)
        assert result.value == 499500


class TestForce:
    """Tests for the force function."""

    def test_force_lazy(self):
        """force should evaluate Lazy objects."""
        result = Lazy(lambda: 42)
        assert force(result) == 42

    def test_force_non_lazy(self):
        """force should pass through non-lazy objects."""
        assert force(42) == 42
        assert force("hello") == "hello"
