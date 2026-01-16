"""Tests for composable API utilities."""

import pytest
from derive import (
    Symbol, Sin, Cos, Simplify, Expand,
    Pipe, pipe, compose, thread_first, thread_last,
    Nest, NestList, FixedPoint, FixedPointList,
)


class TestPipe:
    """Tests for the Pipe class."""

    def test_pipe_basic(self):
        """Pipe should apply transformations."""
        x = Symbol('x')
        result = Pipe((x + 1)**2).then(Expand).value
        assert result == x**2 + 2*x + 1

    def test_pipe_chain(self):
        """Pipe should support chaining."""
        x = Symbol('x')
        result = (
            Pipe(Sin(x)**2 + Cos(x)**2)
            .then(Simplify)
            .value
        )
        assert result == 1

    def test_pipe_with_args(self):
        """Pipe.then should pass additional arguments."""
        x = Symbol('x')
        result = (
            Pipe(x**2 + 2*x + 1)
            .then(lambda e, v: e.subs(x, v), 2)
            .value
        )
        assert result == 9

    def test_pipe_operator(self):
        """Pipe should support | operator."""
        x = Symbol('x')
        result = (Pipe(Sin(x)**2 + Cos(x)**2) | Simplify).value
        assert result == 1


class TestPipeFunction:
    """Tests for the pipe function."""

    def test_pipe_function(self):
        """pipe function should apply transformations in order."""
        x = Symbol('x')
        result = pipe(Sin(x)**2 + Cos(x)**2, Simplify)
        assert result == 1

    def test_pipe_multiple(self):
        """pipe should apply multiple transformations."""
        x = Symbol('x')
        result = pipe(
            (x + 1)**2,
            Expand,
            lambda e: e.subs(x, 2)
        )
        assert result == 9


class TestCompose:
    """Tests for the compose function."""

    def test_compose_basic(self):
        """compose should create composed function."""
        double = lambda x: x * 2
        add_one = lambda x: x + 1
        # compose applies right-to-left
        f = compose(double, add_one)
        assert f(5) == 12  # double(add_one(5)) = 12

    def test_compose_single(self):
        """compose with one function should return that function."""
        f = compose(lambda x: x * 2)
        assert f(5) == 10

    def test_compose_empty(self):
        """compose with no functions should return identity."""
        f = compose()
        assert f(5) == 5


class TestThreadFirst:
    """Tests for thread_first."""

    def test_thread_first_basic(self):
        """thread_first should thread value as first argument."""
        result = thread_first(
            5,
            (lambda x, y: x + y, 3),  # 5 + 3 = 8
            (lambda x, y: x * y, 2)   # 8 * 2 = 16
        )
        assert result == 16

    def test_thread_first_callable(self):
        """thread_first should work with simple callables."""
        result = thread_first(
            [1, 2, 3],
            sum
        )
        assert result == 6


class TestThreadLast:
    """Tests for thread_last."""

    def test_thread_last_basic(self):
        """thread_last should thread value as last argument."""
        result = thread_last(
            [1, 2, 3],
            (map, lambda x: x + 1),  # [2, 3, 4]
            list
        )
        assert result == [2, 3, 4]


class TestNest:
    """Tests for the Nest function."""

    def test_nest_basic(self):
        """Nest should apply function n times."""
        result = Nest(lambda x: x * 2, 1, 4)
        assert result == 16  # 1 -> 2 -> 4 -> 8 -> 16

    def test_nest_zero(self):
        """Nest with n=0 should return original value."""
        result = Nest(lambda x: x * 2, 5, 0)
        assert result == 5

    def test_nest_symbolic(self):
        """Nest should work with symbolic expressions."""
        x = Symbol('x')
        result = Nest(lambda e: e + 1, x, 3)
        assert result == x + 3

    def test_nest_squaring(self):
        """Nest should apply squaring repeatedly."""
        result = Nest(lambda x: x**2, 2, 3)
        assert result == 256  # 2 -> 4 -> 16 -> 256


class TestNestList:
    """Tests for the NestList function."""

    def test_nestlist_basic(self):
        """NestList should return list of all intermediate values."""
        result = NestList(lambda x: x * 2, 1, 4)
        assert result == [1, 2, 4, 8, 16]

    def test_nestlist_zero(self):
        """NestList with n=0 should return single-element list."""
        result = NestList(lambda x: x * 2, 5, 0)
        assert result == [5]

    def test_nestlist_squaring(self):
        """NestList should show squaring progression."""
        result = NestList(lambda x: x**2, 2, 3)
        assert result == [2, 4, 16, 256]

    def test_nestlist_symbolic(self):
        """NestList should work with symbolic expressions."""
        x = Symbol('x')
        result = NestList(lambda e: e + 1, x, 2)
        assert result == [x, x + 1, x + 2]


class TestFixedPoint:
    """Tests for the FixedPoint function."""

    def test_fixedpoint_sqrt2(self):
        """FixedPoint should converge to sqrt(2)."""
        result = FixedPoint(lambda x: (x + 2/x) / 2, 1.0, tol=1e-10)
        assert abs(result - 2**0.5) < 1e-9

    def test_fixedpoint_immediate(self):
        """FixedPoint should return immediately if already at fixed point."""
        result = FixedPoint(lambda x: x, 5)
        assert result == 5

    def test_fixedpoint_symbolic(self):
        """FixedPoint should work with symbolic simplification."""
        x = Symbol('x')
        expr = (x + 1) - 1
        result = FixedPoint(Simplify, expr)
        assert result == x


class TestFixedPointList:
    """Tests for the FixedPointList function."""

    def test_fixedpointlist_convergence(self):
        """FixedPointList should show convergence path."""
        result = FixedPointList(lambda x: (x + 2/x) / 2, 1.0, max_iter=5, tol=1e-10)
        assert len(result) >= 2
        assert result[0] == 1.0
        # Should converge toward sqrt(2)
        assert abs(result[-1] - 2**0.5) < 0.01

    def test_fixedpointlist_immediate(self):
        """FixedPointList should return two-element list at fixed point."""
        result = FixedPointList(lambda x: x, 5, max_iter=10)
        assert result == [5, 5]
