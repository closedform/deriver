"""
Lazy evaluation utilities for derive.

Provides lazy evaluation patterns for expensive symbolic computations,
allowing expressions to be built up without immediate evaluation.
"""

from typing import Any, Callable, TypeVar, Generic, Optional
from functools import wraps

T = TypeVar('T')


class Lazy(Generic[T]):
    """
    A lazy evaluation wrapper that delays computation until needed.

    Parameters
    ----------
    thunk : Callable[[], T]
        A zero-argument function that produces the value when called.

    Examples
    --------
    >>> from derive.utils.lazy import Lazy
    >>> expensive = Lazy(lambda: sum(range(1000000)))
    >>> # Not computed yet
    >>> expensive.value  # Now computed
    499999500000
    """

    __slots__ = ('_thunk', '_value', '_evaluated')

    def __init__(self, thunk: Callable[[], T]) -> None:
        self._thunk = thunk
        self._value: Optional[T] = None
        self._evaluated = False

    @property
    def value(self) -> T:
        """
        Get the value, computing it if necessary.

        Returns
        -------
        T
            The computed value.
        """
        if not self._evaluated:
            self._value = self._thunk()
            self._evaluated = True
            # Allow garbage collection of thunk
            self._thunk = None  # type: ignore
        return self._value  # type: ignore

    def is_evaluated(self) -> bool:
        """Check if the value has been computed."""
        return self._evaluated

    def __repr__(self) -> str:
        if self._evaluated:
            return f"Lazy({self._value!r})"
        return "Lazy(<unevaluated>)"


class LazyExpr:
    """
    A lazy symbolic expression that delays simplification.

    Useful for building complex expressions without triggering
    expensive simplification at each step.

    Parameters
    ----------
    expr : Any
        The symbolic expression.
    auto_simplify : bool, optional
        Whether to automatically simplify when value is accessed.
        Default is False.

    Examples
    --------
    >>> from derive import Symbol, Sin, Cos
    >>> from derive.utils.lazy import LazyExpr
    >>> x = Symbol('x')
    >>> expr = LazyExpr(Sin(x)**2 + Cos(x)**2, auto_simplify=True)
    >>> expr.value  # Simplified on access
    1
    """

    __slots__ = ('_expr', '_simplified', '_auto_simplify', '_value')

    def __init__(self, expr: Any, auto_simplify: bool = False) -> None:
        self._expr = expr
        self._auto_simplify = auto_simplify
        self._simplified = False
        self._value: Optional[Any] = None

    @property
    def value(self) -> Any:
        """
        Get the expression value, optionally simplifying.

        Returns
        -------
        Any
            The expression, simplified if auto_simplify was True.
        """
        if self._auto_simplify and not self._simplified:
            from sympy import simplify
            self._value = simplify(self._expr)
            self._simplified = True
            return self._value
        return self._expr

    @property
    def raw(self) -> Any:
        """Get the raw expression without simplification."""
        return self._expr

    def simplify(self) -> Any:
        """Force simplification and return the result."""
        from sympy import simplify
        self._value = simplify(self._expr)
        self._simplified = True
        return self._value

    def __repr__(self) -> str:
        return f"LazyExpr({self._expr})"


def lazy(func: Callable[..., T]) -> Callable[..., Lazy[T]]:
    """
    Decorator that makes a function return a Lazy wrapper.

    Parameters
    ----------
    func : Callable
        The function to wrap.

    Returns
    -------
    Callable
        A function that returns a Lazy object.

    Examples
    --------
    >>> from derive.utils.lazy import lazy
    >>> @lazy
    ... def expensive_computation(n):
    ...     return sum(range(n))
    >>> result = expensive_computation(1000000)  # Not computed
    >>> result.value  # Now computed
    499999500000
    """
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Lazy[T]:
        return Lazy(lambda: func(*args, **kwargs))
    return wrapper


def force(obj: Any) -> Any:
    """
    Force evaluation of a lazy object.

    If the object is not lazy, returns it unchanged.

    Parameters
    ----------
    obj : Any
        The object to force.

    Returns
    -------
    Any
        The evaluated value.
    """
    if isinstance(obj, Lazy):
        return obj.value
    if isinstance(obj, LazyExpr):
        return obj.value
    return obj


__all__ = ['Lazy', 'LazyExpr', 'lazy', 'force']
