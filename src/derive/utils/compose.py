"""
Composable API utilities for derive.

Provides pipe-style operations and function composition for
natural chaining of transformations on symbolic expressions.
"""

from typing import Any, Callable, TypeVar, Union, List
from functools import reduce

T = TypeVar('T')
R = TypeVar('R')


class Pipe:
    """
    A pipe wrapper that enables fluent, chainable operations.

    Parameters
    ----------
    value : Any
        The initial value to pipe through operations.

    Examples
    --------
    >>> from derive import Symbol, Sin, Cos, Simplify, Expand
    >>> from derive.utils.compose import Pipe
    >>> x = Symbol('x')
    >>> result = (Pipe(Sin(x)**2 + Cos(x)**2)
    ...           .then(Simplify)
    ...           .value)
    >>> result
    1

    >>> # Multiple operations
    >>> result = (Pipe((x + 1)**3)
    ...           .then(Expand)
    ...           .then(lambda e: e.subs(x, 2))
    ...           .value)
    >>> result
    27
    """

    __slots__ = ('_value',)

    def __init__(self, value: Any) -> None:
        self._value = value

    @property
    def value(self) -> Any:
        """Get the current value."""
        return self._value

    def then(self, func: Callable[[Any], Any], *args: Any, **kwargs: Any) -> 'Pipe':
        """
        Apply a function to the current value.

        Parameters
        ----------
        func : Callable
            The function to apply.
        *args : Any
            Additional positional arguments to pass to func.
        **kwargs : Any
            Additional keyword arguments to pass to func.

        Returns
        -------
        Pipe
            A new Pipe with the transformed value.
        """
        if args or kwargs:
            # Function with additional arguments
            result = func(self._value, *args, **kwargs)
        else:
            result = func(self._value)
        return Pipe(result)

    def __or__(self, func: Callable[[Any], Any]) -> 'Pipe':
        """
        Pipe operator for chaining.

        Examples
        --------
        >>> result = Pipe(x) | Sin | Simplify
        """
        return self.then(func)

    def __repr__(self) -> str:
        return f"Pipe({self._value})"


def pipe(value: T, *funcs: Callable) -> Any:
    """
    Pipe a value through a sequence of functions.

    Parameters
    ----------
    value : T
        The initial value.
    *funcs : Callable
        Functions to apply in order.

    Returns
    -------
    Any
        The final result after all transformations.

    Examples
    --------
    >>> from derive import Symbol, Sin, Cos, Simplify
    >>> from derive.utils.compose import pipe
    >>> x = Symbol('x')
    >>> pipe(Sin(x)**2 + Cos(x)**2, Simplify)
    1
    """
    return reduce(lambda v, f: f(v), funcs, value)


def compose(*funcs: Callable) -> Callable:
    """
    Compose multiple functions into a single function.

    Functions are applied right-to-left (last argument is applied first).

    Parameters
    ----------
    *funcs : Callable
        Functions to compose.

    Returns
    -------
    Callable
        The composed function.

    Examples
    --------
    >>> from derive import Simplify, Expand
    >>> from derive.utils.compose import compose
    >>> simplify_expanded = compose(Simplify, Expand)
    >>> # Equivalent to Simplify(Expand(expr))
    """
    if not funcs:
        return lambda x: x
    if len(funcs) == 1:
        return funcs[0]

    def composed(x: Any) -> Any:
        result = x
        for f in reversed(funcs):
            result = f(result)
        return result

    return composed


def thread_first(value: T, *forms: Union[Callable, tuple]) -> Any:
    """
    Thread a value through forms, inserting it as the first argument.

    Similar to Clojure's -> macro.

    Parameters
    ----------
    value : T
        The initial value.
    *forms : Union[Callable, tuple]
        Functions or (function, args...) tuples.

    Returns
    -------
    Any
        The final result.

    Examples
    --------
    >>> from derive.utils.compose import thread_first
    >>> # (-> 5 (+ 3) (* 2)) == (* (+ 5 3) 2) == 16
    >>> thread_first(5, (lambda x, y: x + y, 3), (lambda x, y: x * y, 2))
    16
    """
    result = value
    for form in forms:
        if callable(form):
            result = form(result)
        else:
            func, *args = form
            result = func(result, *args)
    return result


def thread_last(value: T, *forms: Union[Callable, tuple]) -> Any:
    """
    Thread a value through forms, inserting it as the last argument.

    Similar to Clojure's ->> macro.

    Parameters
    ----------
    value : T
        The initial value.
    *forms : Union[Callable, tuple]
        Functions or (function, args...) tuples.

    Returns
    -------
    Any
        The final result.

    Examples
    --------
    >>> from derive.utils.compose import thread_last
    >>> # (->> [1,2,3] (map inc) (filter odd?))
    >>> thread_last([1,2,3], (map, lambda x: x+1), (filter, lambda x: x%2==1), list)
    [3]
    """
    result = value
    for form in forms:
        if callable(form):
            result = form(result)
        else:
            func, *args = form
            result = func(*args, result)
    return result


class Chainable:
    """
    Mixin class that adds chainable methods to expressions.

    Inherit from this to add .simplify(), .expand(), etc. methods
    that return new instances for chaining.
    """

    def simplify(self) -> Any:
        """Simplify the expression."""
        from sympy import simplify
        return simplify(self)

    def expand(self) -> Any:
        """Expand the expression."""
        from sympy import expand
        return expand(self)

    def factor(self) -> Any:
        """Factor the expression."""
        from sympy import factor
        return factor(self)

    def collect(self, *syms: Any) -> Any:
        """Collect terms."""
        from sympy import collect
        return collect(self, *syms)


def Nest(f: Callable, x: Any, n: int) -> Any:
    """
    Apply function f to x repeatedly n times.

    Nest[f, x, n] gives f(f(f(...f(x)...))) with f applied n times.

    Parameters
    ----------
    f : Callable
        Function to apply.
    x : Any
        Initial value.
    n : int
        Number of times to apply f.

    Returns
    -------
    Any
        Result after n applications of f.

    Examples
    --------
    >>> Nest(lambda x: x**2, 2, 3)
    256  # ((2^2)^2)^2 = 256
    >>> from derive import Symbol
    >>> x = Symbol('x')
    >>> Nest(lambda e: e + 1, x, 3)
    x + 3
    """
    result = x
    for _ in range(n):
        result = f(result)
    return result


def NestList(f: Callable, x: Any, n: int) -> List:
    """
    Generate list of repeated function applications.

    NestList[f, x, n] gives {x, f(x), f(f(x)), ..., f^n(x)}.

    Parameters
    ----------
    f : Callable
        Function to apply.
    x : Any
        Initial value.
    n : int
        Number of applications.

    Returns
    -------
    List
        List of length n+1 with successive applications.

    Examples
    --------
    >>> NestList(lambda x: x**2, 2, 3)
    [2, 4, 16, 256]
    >>> NestList(lambda x: 2*x, 1, 4)
    [1, 2, 4, 8, 16]
    """
    result = [x]
    current = x
    for _ in range(n):
        current = f(current)
        result.append(current)
    return result


def FixedPoint(f: Callable, x: Any, max_iter: int = 100, tol: float = None) -> Any:
    """
    Apply function repeatedly until a fixed point is reached.

    FixedPoint[f, x] applies f until f(x) == x.

    Parameters
    ----------
    f : Callable
        Function to apply.
    x : Any
        Initial value.
    max_iter : int, optional
        Maximum iterations (default 100).
    tol : float, optional
        Numerical tolerance for convergence (for numeric types).

    Returns
    -------
    Any
        The fixed point value.

    Examples
    --------
    >>> FixedPoint(lambda x: (x + 2/x)/2, 1.0)  # sqrt(2)
    1.414213...
    >>> from derive import Simplify, Expand
    >>> # Symbolic fixed point
    >>> FixedPoint(Simplify, expr)  # Simplify until no change
    """
    current = x
    for i in range(max_iter):
        next_val = f(current)
        # Check for equality
        if tol is not None and isinstance(next_val, (int, float)):
            if abs(next_val - current) < tol:
                return next_val
        elif next_val == current:
            return current
        # For symbolic expressions, try to simplify comparison
        try:
            from sympy import simplify
            if simplify(next_val - current) == 0:
                return current
        except:
            pass
        current = next_val
    return current


def FixedPointList(f: Callable, x: Any, max_iter: int = 100, tol: float = None) -> List:
    """
    Generate list of values until fixed point is reached.

    Parameters
    ----------
    f : Callable
        Function to apply.
    x : Any
        Initial value.
    max_iter : int, optional
        Maximum iterations (default 100).
    tol : float, optional
        Numerical tolerance for convergence.

    Returns
    -------
    List
        List of values from x to the fixed point.

    Examples
    --------
    >>> FixedPointList(lambda x: (x + 2/x)/2, 1.0, max_iter=10)
    [1.0, 1.5, 1.4166..., 1.41421...]
    """
    result = [x]
    current = x
    for i in range(max_iter):
        next_val = f(current)
        result.append(next_val)
        # Check for convergence
        if tol is not None and isinstance(next_val, (int, float)):
            if abs(next_val - current) < tol:
                return result
        elif next_val == current:
            return result
        try:
            from sympy import simplify
            if simplify(next_val - current) == 0:
                return result
        except:
            pass
        current = next_val
    return result


__all__ = [
    'Pipe',
    'pipe',
    'compose',
    'thread_first',
    'thread_last',
    'Chainable',
    'Nest',
    'NestList',
    'FixedPoint',
    'FixedPointList',
]
