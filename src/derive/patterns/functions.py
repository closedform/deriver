"""
functions.py - Pattern-Based Function Definitions

Allows defining functions with pattern matching on arguments.
Supports multiple definitions with specificity ordering.

Example:
    >>> f = DefineFunction('f')
    >>> f.define(x_, x_**2)  # f(anything) = anything^2
    >>> f.define(Integer(0), Integer(1))  # f(0) = 1 (specific case)
    >>> f(3)  # Returns 9
    >>> f(0)  # Returns 1 (specific case takes precedence)
"""

from typing import Any, Callable, List, Optional, Tuple, Union
from sympy import Wild, Symbol, Expr, Basic, sympify
from sympy.core.function import Function, UndefinedFunction
import sympy as sp


class PatternFunction:
    """
    A function defined by pattern-matching rules.

    PatternFunction allows defining functions where different patterns
    can trigger different behaviors. More specific patterns take precedence
    over general ones.

    Examples:
        >>> f = PatternFunction('f')
        >>> x_ = Pattern_('x')
        >>> f.define(x_, x_**2)
        >>> f(3)
        9
        >>> f.define(0, 1)  # More specific: f(0) = 1
        >>> f(0)
        1
    """

    def __init__(self, name: str):
        """
        Create a new pattern-based function.

        Args:
            name: Name of the function
        """
        self.name = name
        # List of (pattern, condition, replacement, specificity) tuples
        # Ordered by specificity (most specific first)
        self._rules: List[Tuple[Any, Optional[Callable], Any, int]] = []
        # Create the underlying sympy Function
        self._func = Function(name)

    def __repr__(self) -> str:
        return f"PatternFunction('{self.name}', {len(self._rules)} rules)"

    def _calculate_specificity(self, pattern: Any) -> int:
        """
        Calculate the specificity of a pattern.

        More specific patterns have higher scores:
        - Exact values (numbers, symbols) have highest specificity
        - Patterns with more constraints are more specific
        - Patterns with Wild symbols are less specific

        Args:
            pattern: The pattern to evaluate

        Returns:
            Specificity score (higher = more specific)
        """
        if not hasattr(pattern, 'atoms'):
            # Atomic non-sympy value - very specific
            return 1000

        wilds = pattern.atoms(Wild)
        if not wilds:
            # No Wild symbols - exact pattern, most specific
            return 1000

        # Count constrained wilds (those with properties)
        constrained_count = sum(1 for w in wilds if w.assumptions0)

        # Base specificity reduced by number of wilds
        # Boosted by number of constrained wilds
        return 100 - len(wilds) * 10 + constrained_count * 5

    def define(self, pattern: Any, replacement: Any, condition: Optional[Callable] = None) -> 'PatternFunction':
        """
        Add a definition rule for this function.

        Args:
            pattern: Pattern to match (can include Wild symbols)
            replacement: Expression to return when pattern matches
            condition: Optional function (match_dict) -> bool

        Returns:
            self (for chaining)

        Examples:
            >>> f = PatternFunction('f')
            >>> x_ = Pattern_('x')
            >>> f.define(x_, x_**2)  # General case
            >>> f.define(0, 1)       # Specific case
        """
        pattern = sympify(pattern)
        replacement = sympify(replacement)

        specificity = self._calculate_specificity(pattern)

        # Insert rule in specificity order (most specific first)
        inserted = False
        for i, (_, _, _, spec) in enumerate(self._rules):
            if specificity > spec:
                self._rules.insert(i, (pattern, condition, replacement, specificity))
                inserted = True
                break

        if not inserted:
            self._rules.append((pattern, condition, replacement, specificity))

        return self

    def clear(self) -> 'PatternFunction':
        """Clear all rules."""
        self._rules = []
        return self

    def _match_and_replace(self, arg: Any) -> Tuple[Any, bool]:
        """
        Try to match argument against rules and return replacement.

        Args:
            arg: Argument to match

        Returns:
            Tuple of (result, was_matched)
        """
        arg = sympify(arg)

        for pattern, condition, replacement, _ in self._rules:
            # Check for exact match first (no patterns)
            if not hasattr(pattern, 'atoms') or not pattern.atoms(Wild):
                if arg == pattern:
                    if condition is None or condition({}):
                        return replacement, True
                continue

            # Pattern match
            if hasattr(arg, 'match'):
                match = arg.match(pattern)
                if match is not None:
                    # Check condition if present
                    if condition is not None:
                        try:
                            if not condition(match):
                                continue
                        except:
                            continue

                    # Apply substitution
                    try:
                        result = replacement.xreplace(match)
                        return result, True
                    except:
                        continue

        return arg, False

    def __call__(self, *args) -> Any:
        """
        Call the function with arguments.

        For single argument, tries pattern matching directly.
        For multiple arguments, creates a tuple-like expression.

        Args:
            *args: Arguments to the function

        Returns:
            Result of applying matching rule, or unevaluated expression
        """
        if len(args) == 1:
            result, matched = self._match_and_replace(args[0])
            if matched:
                return result
            # Return unevaluated function call
            return self._func(args[0])

        # Multiple arguments - try to match as a whole
        # For now, return unevaluated
        return self._func(*args)

    def rules(self) -> List[Tuple[Any, Any]]:
        """
        Get all defined rules.

        Returns:
            List of (pattern, replacement) tuples
        """
        return [(p, r) for p, _, r, _ in self._rules]


def DefineFunction(name: str) -> PatternFunction:
    """
    Create a new pattern-based function.

    This is the main entry point for defining functions with
    pattern matching capabilities.

    Args:
        name: Name of the function

    Returns:
        PatternFunction instance

    Examples:
        >>> f = DefineFunction('f')
        >>> x_ = Pattern_('x')
        >>> f.define(x_, x_**2)
        >>> f(3)
        9
    """
    return PatternFunction(name)


class FunctionRegistry:
    """
    Global registry of pattern-based functions.

    Allows looking up functions by name and ensures
    consistency across the session.
    """
    _functions: dict = {}

    @classmethod
    def register(cls, func: PatternFunction) -> None:
        """Register a function."""
        cls._functions[func.name] = func

    @classmethod
    def get(cls, name: str) -> Optional[PatternFunction]:
        """Get a function by name."""
        return cls._functions.get(name)

    @classmethod
    def clear(cls) -> None:
        """Clear all registered functions."""
        cls._functions = {}

    @classmethod
    def all_functions(cls) -> List[str]:
        """Get all registered function names."""
        return list(cls._functions.keys())


__all__ = [
    'PatternFunction',
    'DefineFunction',
    'FunctionRegistry',
]
