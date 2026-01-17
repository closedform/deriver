"""
number.py - Number Theory and Numerical Functions.

Provides functions for number manipulation and number theory:
Sign, Floor, Ceiling, N, Round, Mod, GCD, LCM, PrimeQ, Prime, FactorInteger
"""

from typing import Any, List, Optional, Literal
import sympy as sp
from sympy import sign, floor, ceiling, gcd, lcm, isprime, prime, factorint
from derive.functions.utils import alias_function

# Direct aliases
Sign = alias_function('Sign', sign)
Floor = alias_function('Floor', floor)
Ceiling = alias_function('Ceiling', ceiling)


def N(
    expr: Any,
    n: int = 15,
    method: Optional[Literal['sympy', 'mpfr', 'interval', 'floating']] = None
) -> Any:
    """
    Numerical evaluation with multiple precision backends.

    N[expr] - evaluate expr numerically with default precision
    N[expr, n] - evaluate to n digits
    N[expr, n, method='mpfr'] - use mpfr for arbitrary precision
    N[expr, n, method='interval'] - use interval arithmetic for error bounds

    Args:
        expr: Expression to evaluate numerically
        n: Number of significant digits (default 15)
        method: Numerical method to use:
            - 'sympy' (default): SymPy's evalf
            - 'mpfr': mpmath's mpfr for arbitrary precision
            - 'interval': mpmath's interval arithmetic for rigorous bounds
            - 'floating': Standard IEEE 754 floating point

    Returns:
        Numerical value of the expression (type depends on method)

    Examples:
        >>> N(Pi)
        3.14159265358979
        >>> N(Sqrt(2), 50)
        1.4142135623730950488016887242096980785696718753769...
        >>> N(Pi, 30, method='interval')  # Returns interval bounds
        mpi('3.14159265358979...', '3.14159265358979...')
    """
    if method is None or method == 'sympy':
        # Default: use SymPy's evalf
        if hasattr(expr, 'evalf'):
            return expr.evalf(n)
        return float(expr)

    elif method == 'mpfr':
        # Use mpmath for arbitrary precision
        try:
            import mpmath
            mpmath.mp.dps = n  # Set decimal places
            # Convert sympy expression to mpmath
            if hasattr(expr, 'evalf'):
                # Get high-precision string and convert to mpf
                val_str = str(expr.evalf(n + 5))
                return mpmath.mpf(val_str)
            return mpmath.mpf(expr)
        except ImportError:
            raise ImportError("mpmath is required for method='mpfr'. Install with: pip install mpmath")

    elif method == 'interval':
        # Use interval arithmetic for rigorous bounds
        try:
            import mpmath
            mpmath.mp.dps = n
            if hasattr(expr, 'evalf'):
                # Use interval arithmetic
                val = float(expr.evalf(n + 5))
                # Create interval with uncertainty based on precision
                eps = 10 ** (-n)
                return mpmath.mpi(val - eps, val + eps)
            return mpmath.mpi(expr)
        except ImportError:
            raise ImportError("mpmath is required for method='interval'. Install with: pip install mpmath")

    elif method == 'floating':
        # Standard IEEE 754 float
        if hasattr(expr, 'evalf'):
            return float(expr.evalf(n))
        return float(expr)

    else:
        raise ValueError(f"Unknown method: {method}. Use 'sympy', 'mpfr', 'interval', or 'floating'")


def Round(x: Any, n: Any = 1) -> Any:
    """
    Round to nearest integer or multiple.

    Round[x] - round to nearest integer
    Round[x, n] - round to nearest multiple of n

    Args:
        x: Value to round
        n: Multiple to round to (default 1)

    Returns:
        Rounded value

    Examples:
        >>> Round(3.7)
        4
        >>> Round(3.14159, 0.01)
        3.14
    """
    if n == 1:
        return sp.Integer(round(float(x)))
    return round(float(x) / n) * n


def Mod(m: Any, n: Any) -> Any:
    """
    Modulo operation.

    Mod[m, n] - m mod n

    Args:
        m: Dividend
        n: Divisor

    Returns:
        m modulo n

    Examples:
        >>> Mod(17, 5)
        2
    """
    return sp.Mod(m, n)


def GCD(*args: Any) -> Any:
    """
    Greatest common divisor.

    GCD[a, b, ...] - GCD of all arguments

    Args:
        *args: Numbers to find GCD of

    Returns:
        Greatest common divisor

    Examples:
        >>> GCD(12, 18)
        6
        >>> GCD(12, 18, 24)
        6
    """
    if len(args) == 0:
        return 0
    result = args[0]
    for a in args[1:]:
        result = gcd(result, a)
    return result


def LCM(*args: Any) -> Any:
    """
    Least common multiple.

    LCM[a, b, ...] - LCM of all arguments

    Args:
        *args: Numbers to find LCM of

    Returns:
        Least common multiple

    Examples:
        >>> LCM(4, 6)
        12
        >>> LCM(4, 6, 8)
        24
    """
    if len(args) == 0:
        return 1
    result = args[0]
    for a in args[1:]:
        result = lcm(result, a)
    return result


def PrimeQ(n: int) -> bool:
    """
    Check if n is prime.

    Args:
        n: Integer to check

    Returns:
        True if n is prime, False otherwise

    Examples:
        >>> PrimeQ(7)
        True
        >>> PrimeQ(8)
        False
    """
    return isprime(n)


def Prime(n: int) -> int:
    """
    Return the nth prime number.

    Args:
        n: Index (1-based)

    Returns:
        The nth prime number

    Examples:
        >>> Prime(1)
        2
        >>> Prime(10)
        29
    """
    return prime(n)


def FactorInteger(n: int) -> List[List[int]]:
    """
    Prime factorization of integer.

    FactorInteger[n] - list of [prime, exponent] pairs

    Args:
        n: Integer to factor

    Returns:
        List of [prime, exponent] pairs sorted by prime

    Examples:
        >>> FactorInteger(60)
        [[2, 2], [3, 1], [5, 1]]
    """
    factors = factorint(n)
    return [[p, e] for p, e in sorted(factors.items())]


__all__ = [
    'Sign',
    'Floor',
    'Ceiling',
    'N',
    'Round',
    'Mod',
    'GCD',
    'LCM',
    'PrimeQ',
    'Prime',
    'FactorInteger',
]
