"""
matching.py - Pattern Matching and Transformation

Provides pattern matching and expression transformation capabilities
using term rewriting techniques.
"""

from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import sympy as sp
from sympy import Wild, Symbol, Expr, Integer, Float, Rational
from sympy.core.basic import Basic


def _create_wild(name: str) -> Wild:
    """Create a Wild pattern variable."""
    return Wild(name)


def Pattern_(name: str, constraint: Optional[Callable] = None, exclude: Optional[tuple] = None) -> Wild:
    """
    Create a pattern variable that matches any expression.

    Pattern_('x') creates a pattern that matches any expression
    and binds it to 'x'. Optionally takes a constraint function
    and types to exclude from matching.

    Args:
        name: Name of the pattern variable
        constraint: Optional predicate function (expr) -> bool
        exclude: Tuple of types/expressions to exclude from matching

    Returns:
        Wild symbol that matches any expression

    Examples:
        >>> x_ = Pattern_('x')
        >>> Replace(a + b, x_ + y_ >> x_ * y_)  # Would need implementation
    """
    kwargs = {}
    if constraint:
        kwargs['properties'] = [constraint]
    if exclude:
        kwargs['exclude'] = exclude
    return Wild(name, **kwargs)


def Integer_(name: str) -> Wild:
    """
    Create a pattern that matches only integers.

    Args:
        name: Name of the pattern variable

    Returns:
        Wild symbol constrained to integers

    Examples:
        >>> n_ = Integer_('n')
        >>> # n_ matches 1, 2, 3 but not x or 1/2
    """
    return Wild(name, properties=[lambda x: x.is_integer])


def Real_(name: str) -> Wild:
    """
    Create a pattern that matches real numbers.

    Args:
        name: Name of the pattern variable

    Returns:
        Wild symbol constrained to reals

    Examples:
        >>> r_ = Real_('r')
        >>> # r_ matches 1, 1.5, pi but not I
    """
    return Wild(name, properties=[lambda x: x.is_real])


def Positive_(name: str) -> Wild:
    """
    Create a pattern that matches positive values.

    Args:
        name: Name of the pattern variable

    Returns:
        Wild symbol constrained to positive values
    """
    return Wild(name, properties=[lambda x: x.is_positive])


def Negative_(name: str) -> Wild:
    """
    Create a pattern that matches negative values.

    Args:
        name: Name of the pattern variable

    Returns:
        Wild symbol constrained to negative values
    """
    return Wild(name, properties=[lambda x: x.is_negative])


def NonNegative_(name: str) -> Wild:
    """
    Create a pattern that matches non-negative values.

    Args:
        name: Name of the pattern variable

    Returns:
        Wild symbol constrained to non-negative values
    """
    return Wild(name, properties=[lambda x: x.is_nonnegative])


def Symbol_(name: str) -> Wild:
    """
    Create a pattern that matches only symbols.

    Args:
        name: Name of the pattern variable

    Returns:
        Wild symbol constrained to symbols
    """
    return Wild(name, properties=[lambda x: isinstance(x, Symbol) and not isinstance(x, Wild)])


class Rule:
    """
    Represents a transformation rule: pattern -> replacement.

    A Rule consists of a pattern (with Wild variables) and a replacement
    expression. When matched, Wild variables are substituted into the
    replacement.

    Examples:
        >>> x_ = Pattern_('x')
        >>> y_ = Pattern_('y')
        >>> rule = Rule(x_ + y_, x_ * y_)
        >>> # Transforms a + b into a * b
    """

    def __init__(self, pattern: Any, replacement: Any, condition: Optional[Callable] = None):
        """
        Create a transformation rule.

        Args:
            pattern: Pattern to match (can contain Wild variables)
            replacement: Expression to substitute (uses same Wild variables)
            condition: Optional condition function that takes match dict, returns bool
        """
        self.pattern = pattern
        self.replacement = replacement
        self.condition = condition
        # Get the Wild symbols in the pattern
        self._wilds = set()
        if hasattr(pattern, 'atoms'):
            self._wilds = pattern.atoms(Wild)

    def __repr__(self) -> str:
        if self.condition:
            return f"Rule({self.pattern} -> {self.replacement} /; condition)"
        return f"Rule({self.pattern} -> {self.replacement})"

    def _is_trivial_match(self, match: dict) -> bool:
        """
        Check if a match is trivial (e.g., 0 + x = x type match).

        We want to avoid matches where a Wild variable matched 0 or 1
        in a way that's mathematically valid but not meaningful.
        """
        if match is None:
            return True

        # Count how many wilds matched 0 or 1 (trivial identity matches)
        trivial_count = sum(1 for v in match.values()
                          if v == 0 or v == 1 or v == sp.Integer(0) or v == sp.Integer(1))

        # If more than half of the wilds matched trivially, consider it trivial
        if self._wilds and trivial_count > 0 and trivial_count >= len(self._wilds) / 2:
            return True

        return False

    def apply(self, expr: Any) -> Tuple[Any, bool]:
        """
        Try to apply this rule to an expression.

        Args:
            expr: Expression to transform

        Returns:
            Tuple of (transformed_expr, was_applied)
        """
        if not hasattr(expr, 'match'):
            return expr, False

        match = expr.match(self.pattern)
        if match is not None:
            # Reject trivial matches like 0 + x = x
            if self._is_trivial_match(match):
                return expr, False

            # Check condition if present
            if self.condition is not None:
                try:
                    if not self.condition(match):
                        return expr, False
                except:
                    return expr, False

            # Apply substitution
            try:
                result = self.replacement.xreplace(match)
                return result, True
            except:
                return expr, False

        return expr, False


def rule(pattern: Any, replacement: Any, condition: Optional[Callable] = None) -> Rule:
    """
    Create a transformation rule (shorthand for Rule).

    Args:
        pattern: Pattern to match
        replacement: Replacement expression
        condition: Optional condition function

    Returns:
        Rule object

    Examples:
        >>> x_ = Pattern_('x')
        >>> r = rule(x_**2, x_)  # x^2 -> x
    """
    return Rule(pattern, replacement, condition)


def Replace(expr: Any, *rules: Union[Rule, Tuple[Any, Any], Dict[Any, Any]]) -> Any:
    """
    Apply transformation rules once at the top level.

    Replace tries each rule in order and applies the first one that matches.
    Only the outermost expression is transformed - subexpressions are not
    recursively processed.

    Args:
        expr: Expression to transform
        *rules: Rules to apply (Rule objects, (pattern, replacement) tuples, or dict)

    Returns:
        Transformed expression, or original if no rule matched

    Examples:
        >>> x_ = Pattern_('x')
        >>> y_ = Pattern_('y')
        >>> Replace(a + b, rule(x_ + y_, x_ * y_))
        a*b
        >>> Replace(a**2, (x_**2, x_))  # Tuple syntax
        a
        >>> Replace(a + b, {a: 1})  # Dict syntax (direct substitution)
        1 + b
    """
    for r in rules:
        if isinstance(r, dict):
            # Direct substitution using dict
            if hasattr(expr, 'subs'):
                return expr.subs(r)
            continue
        elif isinstance(r, tuple):
            r = Rule(r[0], r[1])

        result, applied = r.apply(expr)
        if applied:
            return result

    return expr


def ReplaceAll(expr: Any, *rules: Union[Rule, Tuple[Any, Any], Dict[Any, Any]]) -> Any:
    """
    Apply transformation rules at all levels of an expression.

    ReplaceAll recursively processes the expression tree, applying rules
    to each subexpression. Rules are tried in order. Unlike Replace, this
    function processes subexpressions before the whole expression.

    Args:
        expr: Expression to transform
        *rules: Rules to apply (Rule objects, tuples, or dict of rules)

    Returns:
        Transformed expression

    Examples:
        >>> x_ = Pattern_('x')
        >>> ReplaceAll(a + b + c, rule(x_ + y_, x_ * y_))
        # Transforms at all levels
    """
    # Normalize rules
    rule_list: List[Rule] = []
    for r in rules:
        if isinstance(r, dict):
            for pattern, replacement in r.items():
                rule_list.append(Rule(pattern, replacement))
        elif isinstance(r, tuple):
            rule_list.append(Rule(r[0], r[1]))
        else:
            rule_list.append(r)

    def try_rules_on_expr(e: Any) -> Any:
        """Try to apply rules to a single expression (not recursively)."""
        for r in rule_list:
            result, applied = r.apply(e)
            if applied:
                return result
        return e

    def apply_rules(e: Any, depth: int = 0) -> Any:
        """Apply rules to expression and its subexpressions."""
        # Prevent infinite recursion
        if depth > 50:
            return e

        if not hasattr(e, 'args') or not e.args:
            # Atomic expression - try to apply rules directly
            return try_rules_on_expr(e)

        # First, recursively process arguments (bottom-up)
        new_args = []
        changed = False
        for arg in e.args:
            new_arg = apply_rules(arg, depth + 1)
            new_args.append(new_arg)
            if new_arg != arg:
                changed = True

        # Reconstruct expression if arguments changed
        if changed:
            try:
                e = e.func(*new_args)
            except:
                pass

        # Then try to apply rules to the expression itself
        return try_rules_on_expr(e)

    return apply_rules(expr)


def ReplaceRepeated(expr: Any, *rules: Union[Rule, Tuple[Any, Any], Dict[Any, Any]],
                    max_iterations: int = 100) -> Any:
    """
    Apply transformation rules repeatedly until no more changes occur.

    ReplaceRepeated applies ReplaceAll repeatedly until the expression
    reaches a fixed point (no more rules apply) or the iteration limit
    is reached.

    Args:
        expr: Expression to transform
        *rules: Rules to apply
        max_iterations: Maximum number of iterations (default 100)

    Returns:
        Transformed expression after reaching fixed point

    Examples:
        >>> x_ = Pattern_('x')
        >>> y_ = Pattern_('y')
        >>> # Repeatedly apply until stable
        >>> ReplaceRepeated((a + b)**2, rule((x_ + y_)**2, x_**2 + 2*x_*y_ + y_**2))
    """
    current = expr
    for _ in range(max_iterations):
        new_expr = ReplaceAll(current, *rules)
        if new_expr == current:
            # Fixed point reached
            return current
        current = new_expr

    return current


def MatchQ(expr: Any, pattern: Any) -> bool:
    """
    Test if an expression matches a pattern.

    Args:
        expr: Expression to test
        pattern: Pattern to match against

    Returns:
        True if expr matches pattern

    Examples:
        >>> x_ = Pattern_('x')
        >>> MatchQ(a + b, x_ + y_)
        True
        >>> MatchQ(a * b, x_ + y_)
        False
    """
    if not hasattr(expr, 'match'):
        return False

    match = expr.match(pattern)
    return match is not None


def Cases(expr: Any, pattern: Any, level: int = -1) -> List[Any]:
    """
    Find all subexpressions matching a pattern.

    Args:
        expr: Expression to search
        pattern: Pattern to match
        level: Depth level (-1 for all levels)

    Returns:
        List of matching subexpressions

    Examples:
        >>> x_ = Integer_('x')
        >>> Cases(a + 1 + b + 2, x_)
        [1, 2]
    """
    results = []

    def search(e: Any, depth: int = 0) -> None:
        if level != -1 and depth > level:
            return

        if hasattr(e, 'match'):
            match = e.match(pattern)
            if match is not None:
                results.append(e)

        if hasattr(e, 'args'):
            for arg in e.args:
                search(arg, depth + 1)

    search(expr)
    return results


def Count(expr: Any, pattern: Any, level: int = -1) -> int:
    """
    Count occurrences of a pattern in an expression.

    Args:
        expr: Expression to search
        pattern: Pattern to count
        level: Depth level (-1 for all levels)

    Returns:
        Number of matches

    Examples:
        >>> x_ = Symbol_('x')
        >>> Count(a + b + c, x_)
        3
    """
    return len(Cases(expr, pattern, level))


def FreeQ(expr: Any, pattern: Any) -> bool:
    """
    Test if an expression is free of a pattern.

    Args:
        expr: Expression to test
        pattern: Pattern to look for

    Returns:
        True if pattern does not appear in expr

    Examples:
        >>> FreeQ(a + b, c)
        True
        >>> FreeQ(a + b, a)
        False
    """
    if hasattr(expr, 'has'):
        # For simple patterns (symbols), use sympy's has
        if isinstance(pattern, Symbol) and not isinstance(pattern, Wild):
            return not expr.has(pattern)

    # For Wild patterns, check with Cases
    return len(Cases(expr, pattern)) == 0


def Position(expr: Any, pattern: Any) -> List[List[int]]:
    """
    Find positions of subexpressions matching a pattern.

    Args:
        expr: Expression to search
        pattern: Pattern to match

    Returns:
        List of positions (each position is a list of indices)

    Examples:
        >>> Position(f(a, g(b, c)), b)
        [[1, 0]]  # Position of b in f(a, g(b, c))
    """
    positions = []

    def search(e: Any, path: List[int]) -> None:
        if hasattr(e, 'match'):
            match = e.match(pattern)
            if match is not None:
                positions.append(path.copy())

        if hasattr(e, 'args'):
            for i, arg in enumerate(e.args):
                path.append(i)
                search(arg, path)
                path.pop()

    search(expr, [])
    return positions


__all__ = [
    # Pattern creators
    'Pattern_', 'Integer_', 'Real_', 'Positive_', 'Negative_',
    'NonNegative_', 'Symbol_',
    # Rule class and function
    'Rule', 'rule',
    # Transformation functions
    'Replace', 'ReplaceAll', 'ReplaceRepeated',
    # Pattern matching functions
    'MatchQ', 'Cases', 'Count', 'FreeQ', 'Position',
]
