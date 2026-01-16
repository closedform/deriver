"""
patterns - Pattern Matching and Transformation

Provides pattern matching and expression transformation capabilities
using term rewriting techniques.

Functions:
- Replace: Single-pass pattern replacement
- ReplaceAll: Replace all occurrences at all levels
- ReplaceRepeated: Apply rules until expression stops changing
- Pattern helpers: Pattern_, Integer_, Real_, etc.
"""

from derive.patterns.matching import (
    Pattern_, Integer_, Real_, Positive_, Negative_,
    NonNegative_, Symbol_,
    Rule, rule,
    Replace, ReplaceAll, ReplaceRepeated,
    MatchQ, Cases, Count, FreeQ, Position,
)
from derive.patterns.functions import (
    PatternFunction, DefineFunction, FunctionRegistry,
)

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
    # Pattern-based functions
    'PatternFunction', 'DefineFunction', 'FunctionRegistry',
]
