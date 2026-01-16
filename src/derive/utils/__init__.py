"""
Utils module - Display, Logic, String operations, and Performance utilities.

Provides utility functions for output, control flow, lazy evaluation,
caching, and composable operations.
"""

from derive.utils.display import (
    TeXForm, PrettyForm, Print, RichPrint, RichLatex, TableForm, show,
)
from derive.utils.logic import (
    Equal, Unequal, Less, LessEqual, Greater, GreaterEqual,
    And, Or, Not, Xor, Nand, Nor,
    If, Which, Switch, Piecewise, Max, Min,
)
from derive.utils.strings import (
    StringJoin, StringLength, StringTake, StringDrop,
    StringReplace, ToString,
)
from derive.utils.lazy import Lazy, LazyExpr, lazy, force
from derive.utils.cache import (
    ExpressionCache, memoize, memoize_method, cached_simplify,
    get_christoffel_cache, get_riemann_cache, clear_all_caches,
)
from derive.utils.compose import (
    Pipe, pipe, compose, thread_first, thread_last, Chainable,
    Nest, NestList, FixedPoint, FixedPointList,
)
from derive.utils.assumptions import (
    Assuming, Refine, Ask, SimplifyWithAssumptions,
    get_assumptions, assume_positive, assume_real, assume_integer,
    AssumedSymbol,
    Positive, Negative, Real, Integer as IntegerPred, Rational as RationalPred,
    Complex as ComplexPred, Even, Odd, Prime as PrimePred,
    Nonzero, Nonnegative, Nonpositive, Finite, Infinite, Zero, Bounded,
)

__all__ = [
    # Display
    'TeXForm', 'PrettyForm', 'Print', 'RichPrint', 'RichLatex', 'TableForm', 'show',
    # Logic
    'Equal', 'Unequal', 'Less', 'LessEqual', 'Greater', 'GreaterEqual',
    'And', 'Or', 'Not', 'Xor', 'Nand', 'Nor',
    'If', 'Which', 'Switch', 'Piecewise', 'Max', 'Min',
    # Strings
    'StringJoin', 'StringLength', 'StringTake', 'StringDrop',
    'StringReplace', 'ToString',
    # Lazy evaluation
    'Lazy', 'LazyExpr', 'lazy', 'force',
    # Caching
    'ExpressionCache', 'memoize', 'memoize_method', 'cached_simplify',
    'get_christoffel_cache', 'get_riemann_cache', 'clear_all_caches',
    # Composable API
    'Pipe', 'pipe', 'compose', 'thread_first', 'thread_last', 'Chainable',
    'Nest', 'NestList', 'FixedPoint', 'FixedPointList',
    # Assumptions
    'Assuming', 'Refine', 'Ask', 'SimplifyWithAssumptions',
    'get_assumptions', 'assume_positive', 'assume_real', 'assume_integer',
    'AssumedSymbol',
    'Positive', 'Negative', 'Real', 'IntegerPred', 'RationalPred',
    'ComplexPred', 'Even', 'Odd', 'PrimePred',
    'Nonzero', 'Nonnegative', 'Nonpositive', 'Finite', 'Infinite', 'Zero', 'Bounded',
]
