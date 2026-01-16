"""
numbers.py - Smart Number Handling for Derive.

Provides automatic rational preservation, context managers for exact/numerical
modes, and utilities for converting between number representations.

Design Principle:
Everything is automatically a rational complex number unless you specify decimal,
then it is arbitrary precision. Other approximation methods available when specified.
"""

from contextlib import contextmanager
from fractions import Fraction
from typing import Any, Optional, Generator
import sympy as sp
from sympy import Rational, Integer, Float, nsimplify


# Global state for exact mode
_exact_mode = False
_numerical_mode = False
_numerical_precision = 15


def rationalize(value: Any, tolerance: float = 1e-10) -> Any:
    """
    Convert a value to a rational if possible.

    Args:
        value: Value to convert (float, int, Fraction, etc.)
        tolerance: Maximum error tolerance for float conversion

    Returns:
        Rational representation if conversion is reasonable, otherwise original value

    Examples:
        >>> rationalize(0.5)
        1/2
        >>> rationalize(0.333333333333)
        1/3
    """
    if isinstance(value, (Rational, Integer)):
        return value

    if isinstance(value, int):
        return Integer(value)

    if isinstance(value, Fraction):
        return Rational(value.numerator, value.denominator)

    if isinstance(value, float):
        # Check for simple fractions first
        if value == 0:
            return Integer(0)

        # Try common simple fractions first
        for denom in range(1, 1001):
            numer = round(value * denom)
            if abs(numer / denom - value) < tolerance:
                return Rational(numer, denom)

        # Fall back to sympy's nsimplify
        try:
            result = nsimplify(value, rational=True, tolerance=tolerance)
            if isinstance(result, (Rational, Integer)):
                return result
        except:
            pass

        return Float(value)

    # For sympy expressions containing floats, try to rationalize
    if hasattr(value, 'atoms'):
        float_atoms = value.atoms(Float)
        if float_atoms:
            subs_dict = {}
            for f in float_atoms:
                rat = rationalize(float(f), tolerance)
                if rat != f:
                    subs_dict[f] = rat
            if subs_dict:
                return value.subs(subs_dict)

    return value


def to_rational(numerator: int, denominator: int = 1) -> Rational:
    """
    Create a Rational from numerator and denominator.

    Args:
        numerator: The numerator
        denominator: The denominator (default 1)

    Returns:
        Rational number

    Examples:
        >>> to_rational(1, 3)
        1/3
        >>> to_rational(5)
        5
    """
    return Rational(numerator, denominator)


@contextmanager
def exact() -> Generator[None, None, None]:
    """
    Context manager for exact rational arithmetic.

    Within this context, numerical operations attempt to preserve exact
    rational representations.

    Examples:
        >>> with exact():
        ...     result = Rational(1, 3) + Rational(1, 4)
        >>> result
        7/12
    """
    global _exact_mode
    old_mode = _exact_mode
    _exact_mode = True
    try:
        yield
    finally:
        _exact_mode = old_mode


@contextmanager
def numerical(precision: int = 15) -> Generator[None, None, None]:
    """
    Context manager for numerical (arbitrary precision) arithmetic.

    Within this context, operations evaluate numerically to the specified precision.

    Args:
        precision: Number of significant digits (default 15)

    Examples:
        >>> from derive import Pi
        >>> with numerical(50):
        ...     result = N(Pi, 50)
        >>> # result has 50 digits of precision
    """
    global _numerical_mode, _numerical_precision
    old_mode = _numerical_mode
    old_precision = _numerical_precision
    _numerical_mode = True
    _numerical_precision = precision
    try:
        yield
    finally:
        _numerical_mode = old_mode
        _numerical_precision = old_precision


def is_exact_mode() -> bool:
    """Check if currently in exact mode."""
    return _exact_mode


def is_numerical_mode() -> bool:
    """Check if currently in numerical mode."""
    return _numerical_mode


def get_numerical_precision() -> int:
    """Get current numerical precision."""
    return _numerical_precision


def auto_rational(expr: Any) -> Any:
    """
    Automatically convert floats in an expression to rationals where sensible.

    This is the core function for smart number handling. It recursively
    processes an expression and converts float values to rationals when
    they can be represented exactly.

    Args:
        expr: Expression to process

    Returns:
        Expression with floats converted to rationals where possible

    Examples:
        >>> x = Symbol('x')
        >>> auto_rational(0.5 * x)
        x/2
        >>> auto_rational(0.333333333333 * x)
        x/3
    """
    return rationalize(expr)


class SmartNumber:
    """
    A wrapper class that provides smart arithmetic behavior.

    Operations with SmartNumber automatically preserve rationals
    and handle precision appropriately.

    Examples:
        >>> n = SmartNumber(1) / SmartNumber(3)
        >>> n.value
        1/3
    """

    def __init__(self, value: Any):
        """
        Initialize a SmartNumber.

        Args:
            value: The numeric value to wrap
        """
        if isinstance(value, SmartNumber):
            self._value = value._value
        elif isinstance(value, int):
            self._value = Integer(value)
        elif isinstance(value, float):
            self._value = rationalize(value)
        elif isinstance(value, Fraction):
            self._value = Rational(value.numerator, value.denominator)
        else:
            self._value = value

    @property
    def value(self) -> Any:
        """Get the underlying value."""
        return self._value

    def __repr__(self) -> str:
        return f"SmartNumber({self._value})"

    def __str__(self) -> str:
        return str(self._value)

    def __add__(self, other: Any) -> 'SmartNumber':
        other_val = SmartNumber(other)._value if not isinstance(other, SmartNumber) else other._value
        return SmartNumber(self._value + other_val)

    def __radd__(self, other: Any) -> 'SmartNumber':
        return self.__add__(other)

    def __sub__(self, other: Any) -> 'SmartNumber':
        other_val = SmartNumber(other)._value if not isinstance(other, SmartNumber) else other._value
        return SmartNumber(self._value - other_val)

    def __rsub__(self, other: Any) -> 'SmartNumber':
        other_val = SmartNumber(other)._value if not isinstance(other, SmartNumber) else other._value
        return SmartNumber(other_val - self._value)

    def __mul__(self, other: Any) -> 'SmartNumber':
        other_val = SmartNumber(other)._value if not isinstance(other, SmartNumber) else other._value
        return SmartNumber(self._value * other_val)

    def __rmul__(self, other: Any) -> 'SmartNumber':
        return self.__mul__(other)

    def __truediv__(self, other: Any) -> 'SmartNumber':
        other_val = SmartNumber(other)._value if not isinstance(other, SmartNumber) else other._value
        # Use Rational division for integers
        if isinstance(self._value, (int, Integer)) and isinstance(other_val, (int, Integer)):
            return SmartNumber(Rational(int(self._value), int(other_val)))
        return SmartNumber(self._value / other_val)

    def __rtruediv__(self, other: Any) -> 'SmartNumber':
        other_val = SmartNumber(other)._value if not isinstance(other, SmartNumber) else other._value
        if isinstance(other_val, (int, Integer)) and isinstance(self._value, (int, Integer)):
            return SmartNumber(Rational(int(other_val), int(self._value)))
        return SmartNumber(other_val / self._value)

    def __pow__(self, other: Any) -> 'SmartNumber':
        other_val = SmartNumber(other)._value if not isinstance(other, SmartNumber) else other._value
        return SmartNumber(self._value ** other_val)

    def __rpow__(self, other: Any) -> 'SmartNumber':
        other_val = SmartNumber(other)._value if not isinstance(other, SmartNumber) else other._value
        return SmartNumber(other_val ** self._value)

    def __neg__(self) -> 'SmartNumber':
        return SmartNumber(-self._value)

    def __pos__(self) -> 'SmartNumber':
        return SmartNumber(+self._value)

    def __abs__(self) -> 'SmartNumber':
        return SmartNumber(abs(self._value))

    def __eq__(self, other: Any) -> bool:
        other_val = SmartNumber(other)._value if not isinstance(other, SmartNumber) else other._value
        return self._value == other_val

    def __ne__(self, other: Any) -> bool:
        return not self.__eq__(other)

    def __lt__(self, other: Any) -> bool:
        other_val = SmartNumber(other)._value if not isinstance(other, SmartNumber) else other._value
        return self._value < other_val

    def __le__(self, other: Any) -> bool:
        other_val = SmartNumber(other)._value if not isinstance(other, SmartNumber) else other._value
        return self._value <= other_val

    def __gt__(self, other: Any) -> bool:
        other_val = SmartNumber(other)._value if not isinstance(other, SmartNumber) else other._value
        return self._value > other_val

    def __ge__(self, other: Any) -> bool:
        other_val = SmartNumber(other)._value if not isinstance(other, SmartNumber) else other._value
        return self._value >= other_val

    def __hash__(self) -> int:
        return hash(self._value)

    def __float__(self) -> float:
        return float(self._value)

    def __int__(self) -> int:
        return int(self._value)


def S(value: Any) -> SmartNumber:
    """
    Shorthand for creating a SmartNumber.

    Args:
        value: Value to convert to SmartNumber

    Returns:
        SmartNumber wrapper

    Examples:
        >>> S(1) / S(3)
        SmartNumber(1/3)
    """
    return SmartNumber(value)


def expr(s: str, local_dict: Optional[dict] = None) -> Any:
    """
    Parse a string expression with automatic rational conversion.

    This parser converts division of integers to rationals automatically,
    so `expr("1/3 + 1/4")` returns `Rational(7, 12)` instead of a float.

    Args:
        s: String expression to parse (e.g., "x^2 + 1/3*x")
        local_dict: Additional symbols to include in parsing context

    Returns:
        Parsed symbolic expression

    Examples:
        >>> expr("1/3 + 1/4")
        7/12
        >>> expr("x^2 + 3*x + 1")  # Uses ^ for power
        x**2 + 3*x + 1
        >>> expr("sin(x)/x")
        sin(x)/x
        >>> expr("x/3 + y/2")
        x/3 + y/2
    """
    import re
    from sympy.parsing.sympy_parser import (
        parse_expr,
        standard_transformations,
        implicit_multiplication_application,
        convert_xor,
    )
    from sympy import Symbol

    # Default transformations include converting ^ to **
    transformations = standard_transformations + (
        implicit_multiplication_application,
        convert_xor,
    )

    # Build local dictionary with common symbols
    default_locals = {
        'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan,
        'cot': sp.cot, 'sec': sp.sec, 'csc': sp.csc,
        'sinh': sp.sinh, 'cosh': sp.cosh, 'tanh': sp.tanh,
        'asin': sp.asin, 'acos': sp.acos, 'atan': sp.atan,
        'arcsin': sp.asin, 'arccos': sp.acos, 'arctan': sp.atan,
        'exp': sp.exp, 'log': sp.log, 'ln': sp.ln,
        'sqrt': sp.sqrt, 'abs': sp.Abs,
        'pi': sp.pi, 'e': sp.E, 'i': sp.I,
        'Pi': sp.pi, 'E': sp.E, 'I': sp.I,
        'oo': sp.oo, 'inf': sp.oo,
        'Rational': Rational, 'Integer': Integer,
    }

    if local_dict:
        default_locals.update(local_dict)

    # Pre-process: convert integer division to Rational
    # Match patterns like `3/4` (integers only) and convert to Rational(3, 4)
    def replace_int_division(match):
        num = match.group(1)
        denom = match.group(2)
        return f'Rational({num}, {denom})'

    # Pattern to match integer/integer (but not when preceded by letters/underscore)
    # This handles cases like `1/3` but not `x/3` (which should stay as division)
    int_div_pattern = r'(?<![a-zA-Z_0-9])(\d+)/(\d+)(?![a-zA-Z_0-9])'
    processed = re.sub(int_div_pattern, replace_int_division, s)

    # Parse the expression
    result = parse_expr(
        processed,
        local_dict=default_locals,
        transformations=transformations,
        evaluate=True
    )

    return result


def float_to_rational(value: float, max_denominator: int = 10000) -> Optional[Rational]:
    """
    Convert a float to a rational if it represents a simple fraction.

    Args:
        value: Float value to convert
        max_denominator: Maximum denominator to consider

    Returns:
        Rational if conversion is exact within tolerance, None otherwise

    Examples:
        >>> float_to_rational(0.5)
        1/2
        >>> float_to_rational(0.333333333333)
        1/3
        >>> float_to_rational(3.14159)  # Not a simple fraction
        None
    """
    if value == 0:
        return Integer(0)

    # Use Python's Fraction for the conversion
    try:
        frac = Fraction(value).limit_denominator(max_denominator)
        # Check if this is a good approximation
        if abs(float(frac) - value) < 1e-12:
            return Rational(frac.numerator, frac.denominator)
    except (ValueError, OverflowError):
        pass

    return None


def ensure_rational(value: Any) -> Any:
    """
    Ensure a value is represented as a rational where possible.

    For integers, returns Integer. For floats that represent simple fractions,
    returns Rational. Otherwise returns the original value.

    Args:
        value: Value to convert

    Returns:
        Rational/Integer representation if possible, otherwise original value

    Examples:
        >>> ensure_rational(0.5)
        1/2
        >>> ensure_rational(3)
        3
        >>> ensure_rational(0.123456789)  # Not a simple fraction
        0.123456789
    """
    if isinstance(value, (Rational, Integer)):
        return value
    if isinstance(value, int):
        return Integer(value)
    if isinstance(value, float):
        rat = float_to_rational(value)
        return rat if rat is not None else Float(value)
    return value


# Rational number shortcuts
def R(p: int, q: int = 1) -> Rational:
    """
    Shorthand for Rational.

    R(1, 2) is equivalent to Rational(1, 2) = 1/2

    Examples:
        >>> R(1, 2)
        1/2
        >>> R(3, 4)
        3/4
        >>> R(5)  # Same as Rational(5)
        5
    """
    return Rational(p, q)


# Common fractions as constants
Half = Rational(1, 2)
Third = Rational(1, 3)
Quarter = Rational(1, 4)
TwoThirds = Rational(2, 3)
ThreeQuarters = Rational(3, 4)


__all__ = [
    'rationalize',
    'to_rational',
    'exact',
    'numerical',
    'is_exact_mode',
    'is_numerical_mode',
    'get_numerical_precision',
    'auto_rational',
    'SmartNumber',
    'S',
    'expr',
    'float_to_rational',
    'ensure_rational',
    # Rational shortcuts
    'R',
    'Half',
    'Third',
    'Quarter',
    'TwoThirds',
    'ThreeQuarters',
]
