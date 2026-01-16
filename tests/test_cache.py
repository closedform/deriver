"""Tests for caching utilities."""

import pytest
from derive import (
    Symbol, Sin, Simplify,
    ExpressionCache, memoize, memoize_method, cached_simplify,
    clear_all_caches,
)


class TestExpressionCache:
    """Tests for the ExpressionCache class."""

    def test_cache_set_get(self):
        """Cache should store and retrieve values."""
        cache = ExpressionCache()
        cache.set('key', 'value')
        assert cache.get('key') == 'value'

    def test_cache_default(self):
        """Cache should return default for missing keys."""
        cache = ExpressionCache()
        assert cache.get('missing') is None
        assert cache.get('missing', 'default') == 'default'

    def test_cache_contains(self):
        """Cache should support 'in' operator."""
        cache = ExpressionCache()
        cache.set('key', 'value')
        assert 'key' in cache
        assert 'missing' not in cache

    def test_cache_clear(self):
        """Cache should clear all entries."""
        cache = ExpressionCache()
        cache.set('key1', 'value1')
        cache.set('key2', 'value2')
        assert len(cache) == 2
        cache.clear()
        assert len(cache) == 0

    def test_cache_maxsize(self):
        """Cache should evict old entries when full."""
        cache = ExpressionCache(maxsize=3)
        cache.set('a', 1)
        cache.set('b', 2)
        cache.set('c', 3)
        cache.set('d', 4)  # Should evict 'a'
        assert len(cache) == 3
        assert 'a' not in cache


class TestMemoize:
    """Tests for the memoize decorator."""

    def test_memoize_caches(self):
        """memoize should cache function results."""
        counter = [0]

        @memoize
        def expensive(n):
            counter[0] += 1
            return n * 2

        assert expensive(5) == 10
        assert expensive(5) == 10
        assert counter[0] == 1  # Only called once

    def test_memoize_different_args(self):
        """memoize should cache different arg combinations separately."""
        counter = [0]

        @memoize
        def add(a, b):
            counter[0] += 1
            return a + b

        assert add(1, 2) == 3
        assert add(3, 4) == 7
        assert add(1, 2) == 3  # From cache
        assert counter[0] == 2


class TestMemoizeMethod:
    """Tests for the memoize_method decorator."""

    def test_memoize_method(self):
        """memoize_method should work with instance methods."""
        class Calculator:
            def __init__(self):
                self.call_count = 0

            @memoize_method
            def compute(self, n):
                self.call_count += 1
                return n ** 2

        calc = Calculator()
        assert calc.compute(5) == 25
        assert calc.compute(5) == 25  # From cache
        assert calc.call_count == 1


class TestCachedSimplify:
    """Tests for cached_simplify."""

    def test_cached_simplify(self):
        """cached_simplify should return simplified result."""
        x = Symbol('x')
        result = cached_simplify(x + x)
        assert result == 2 * x

    def test_clear_all_caches(self):
        """clear_all_caches should not raise errors."""
        clear_all_caches()  # Should not raise
