"""
Caching and memoization utilities for derive.

Provides memoization decorators and caching utilities for expensive
symbolic computations like Christoffel symbols, curvature tensors, etc.
"""

from typing import Any, Callable, Dict, Optional, TypeVar, Hashable
from functools import wraps
import weakref
import hashlib

T = TypeVar('T')


class ExpressionCache:
    """
    A cache for symbolic expressions keyed by their string representation.

    Uses weak references to allow garbage collection of unused values.

    Parameters
    ----------
    maxsize : int, optional
        Maximum number of entries. Default is 1000.

    Examples
    --------
    >>> from derive.utils.cache import ExpressionCache
    >>> cache = ExpressionCache()
    >>> cache.set('key', 'value')
    >>> cache.get('key')
    'value'
    """

    def __init__(self, maxsize: int = 1000) -> None:
        self._cache: Dict[str, Any] = {}
        self._maxsize = maxsize

    def _make_key(self, expr: Any) -> str:
        """Create a cache key from an expression."""
        s = str(expr)
        if len(s) > 256:
            # Use hash for very long expressions
            return hashlib.md5(s.encode()).hexdigest()
        return s

    def get(self, key: Any, default: Any = None) -> Any:
        """
        Get a value from the cache.

        Parameters
        ----------
        key : Any
            The cache key (will be converted to string).
        default : Any, optional
            Value to return if not found.

        Returns
        -------
        Any
            The cached value or default.
        """
        k = self._make_key(key)
        return self._cache.get(k, default)

    def set(self, key: Any, value: Any) -> None:
        """
        Set a value in the cache.

        Parameters
        ----------
        key : Any
            The cache key.
        value : Any
            The value to cache.
        """
        if len(self._cache) >= self._maxsize:
            # Simple FIFO eviction - remove oldest
            oldest = next(iter(self._cache))
            del self._cache[oldest]
        k = self._make_key(key)
        self._cache[k] = value

    def clear(self) -> None:
        """Clear all cached values."""
        self._cache.clear()

    def __contains__(self, key: Any) -> bool:
        return self._make_key(key) in self._cache

    def __len__(self) -> int:
        return len(self._cache)


# Global caches for common computations
_christoffel_cache = ExpressionCache(maxsize=500)
_riemann_cache = ExpressionCache(maxsize=200)
_simplify_cache = ExpressionCache(maxsize=1000)


def memoize(func: Callable[..., T]) -> Callable[..., T]:
    """
    Simple memoization decorator for functions with hashable arguments.

    Parameters
    ----------
    func : Callable
        The function to memoize.

    Returns
    -------
    Callable
        Memoized function.

    Examples
    --------
    >>> from derive.utils.cache import memoize
    >>> @memoize
    ... def fibonacci(n):
    ...     if n < 2:
    ...         return n
    ...     return fibonacci(n-1) + fibonacci(n-2)
    >>> fibonacci(100)  # Computed efficiently
    354224848179261915075
    """
    cache: Dict[Hashable, T] = {}

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        # Create cache key from args and kwargs
        try:
            key = (args, tuple(sorted(kwargs.items())))
            if key not in cache:
                cache[key] = func(*args, **kwargs)
            return cache[key]
        except TypeError:
            # Unhashable args, just call the function
            return func(*args, **kwargs)

    wrapper.cache = cache  # type: ignore
    wrapper.cache_clear = cache.clear  # type: ignore
    return wrapper


def memoize_method(func: Callable[..., T]) -> Callable[..., T]:
    """
    Memoization decorator for instance methods.

    Stores cache per instance using weak references.

    Parameters
    ----------
    func : Callable
        The method to memoize.

    Returns
    -------
    Callable
        Memoized method.
    """
    cache_attr = f'_memoize_cache_{func.__name__}'

    @wraps(func)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> T:
        # Get or create cache for this instance
        if not hasattr(self, cache_attr):
            setattr(self, cache_attr, {})
        cache = getattr(self, cache_attr)

        try:
            key = (args, tuple(sorted(kwargs.items())))
            if key not in cache:
                cache[key] = func(self, *args, **kwargs)
            return cache[key]
        except TypeError:
            return func(self, *args, **kwargs)

    return wrapper


def cached_simplify(expr: Any) -> Any:
    """
    Simplify with caching to avoid repeated work.

    Parameters
    ----------
    expr : Any
        The expression to simplify.

    Returns
    -------
    Any
        The simplified expression.
    """
    from sympy import simplify

    if expr in _simplify_cache:
        return _simplify_cache.get(expr)

    result = simplify(expr)
    _simplify_cache.set(expr, result)
    return result


def get_christoffel_cache() -> ExpressionCache:
    """Get the global Christoffel symbol cache."""
    return _christoffel_cache


def get_riemann_cache() -> ExpressionCache:
    """Get the global Riemann tensor cache."""
    return _riemann_cache


def clear_all_caches() -> None:
    """Clear all global caches."""
    _christoffel_cache.clear()
    _riemann_cache.clear()
    _simplify_cache.clear()


__all__ = [
    'ExpressionCache',
    'memoize',
    'memoize_method',
    'cached_simplify',
    'get_christoffel_cache',
    'get_riemann_cache',
    'clear_all_caches',
]
