"""
Derive: A Powerful Symbolic Mathematics Library for Python

This package provides intuitive symbolic math capabilities including calculus,
linear algebra, differential equations, tensor calculus, and more.

Usage:
    from derive import *

    x = Symbol('x')
    Integrate(Sin(x), x)  # Returns -Cos(x)
    D(x**3, x)            # Returns 3*x**2
"""

__version__ = "0.1.0"

# Core: Symbols, Constants, and Smart Numbers
from derive.core import (
    Symbol, symbols, Function,
    Rational, Integer, Float,
    # Common types (so users don't need to import sympy)
    Array, ImmutableDenseNDimArray, MutableDenseNDimArray,
    Heaviside, DiracDelta,
    Pi, E, I, Infinity, oo, ComplexInfinity, Indeterminate,
    # Smart number handling
    rationalize, to_rational, exact, numerical,
    is_exact_mode, is_numerical_mode, get_numerical_precision,
    auto_rational, SmartNumber, S, expr,
    float_to_rational, ensure_rational,
    # Rational shortcuts: R(1,2) instead of Rational(1,2)
    R, Half, Third, Quarter, TwoThirds, ThreeQuarters,
)

# Functions: Trig, Exponential, Complex, Special, Number
from derive.functions import (
    # Trigonometric
    Sin, Cos, Tan, Cot, Sec, Csc,
    Sinh, Cosh, Tanh, Coth, Sech, Csch,
    ArcSin, ArcCos, ArcTan, ArcCot, ArcSec, ArcCsc,
    ArcSinh, ArcCosh, ArcTanh, ArcCoth, ArcSech, ArcCsch,
    # Exponential
    Exp, Log, Ln, Sqrt, Power,
    # Complex
    Re, Im, Conjugate, Arg, Abs,
    # Special functions
    Factorial, Binomial, Gamma, Beta,
    BesselJ, BesselY, BesselI, BesselK,
    HankelH1, HankelH2,
    SphericalBesselJ, SphericalBesselY,
    LegendreP, AssociatedLegendreP,
    ChebyshevT, ChebyshevU,
    HermiteH, LaguerreL, AssociatedLaguerreL,
    GegenbauerC, JacobiP,
    SphericalHarmonicY,
    EllipticK, EllipticE, EllipticF, EllipticPi,
    Erf, Erfc, Erfi,
    ExpIntegralEi, SinIntegral, CosIntegral, LogIntegral,
    FresnelS, FresnelC,
    AiryAi, AiryBi,
    Zeta, PolyLog,
    Hypergeometric2F1, HypergeometricPFQ, MeijerG,
    # Number functions
    Sign, Floor, Ceiling, N, Round, Mod,
    GCD, LCM, PrimeQ, Prime, FactorInteger,
)

# Calculus
from derive.calculus import (
    D, Integrate, NIntegrate, Limit, Series, Sum, Product,
    # Change of variables
    ChangeVariables, IntegrateWithSubstitution,
    # Integral transforms
    FourierTransform, InverseFourierTransform,
    LaplaceTransform, InverseLaplaceTransform,
    Convolve,
)

# Algebra: Solving, Simplification, Linear Algebra
from derive.algebra import (
    Solve, NSolve, FindRoot,
    Simplify, Expand, Factor, Collect,
    Cancel, Apart, Together, TrigSimplify, TrigExpand, TrigReduce,
    PowerSimplify, LogSimplify,
    Matrix, Dot, Transpose, Inverse, Det,
    Eigenvalues, Eigenvectors,
    IdentityMatrix, DiagonalMatrix, ZeroMatrix,
    Tr, MatrixRank, NullSpace, RowReduce,
    ConjugateTranspose, MatrixExp, CharacteristicPolynomial,
)

# ODE: Differential Equations
from derive.ode import DSolve, NDSolve

# Probability and Statistics
from derive.probability import (
    NormalDistribution, UniformDistribution, ExponentialDistribution,
    PoissonDistribution, BinomialDistribution,
    PDF, CDF, Mean, Variance, StandardDeviation,
    RandomVariate, Probability,
)

# Data: I/O and List Operations
from derive.data import (
    Import, Export,
    Table, Range, Map, Select, Sort, Total, Length,
    First, Last, Take, Drop, Append, Prepend, Join,
    Flatten, Partition,
)

# Utils: Display, Logic, Strings, Performance
from derive.utils import (
    TeXForm, PrettyForm, Print, RichPrint, RichLatex, TableForm, show,
    Equal, Unequal, Less, LessEqual, Greater, GreaterEqual,
    And, Or, Not, Xor, Nand, Nor,
    If, Which, Switch, Piecewise, Max, Min,
    StringJoin, StringLength, StringTake, StringDrop,
    StringReplace, ToString,
    # Lazy evaluation
    Lazy, LazyExpr, lazy, force,
    # Caching
    ExpressionCache, memoize, memoize_method, cached_simplify,
    get_christoffel_cache, get_riemann_cache, clear_all_caches,
    # Composable API
    Pipe, pipe, compose, thread_first, thread_last, Chainable,
    Nest, NestList, FixedPoint, FixedPointList,
    # Assumptions
    Assuming, Refine, Ask, SimplifyWithAssumptions,
    get_assumptions, assume_positive, assume_real, assume_integer,
    AssumedSymbol,
    Positive, Negative, Real, IntegerPred, RationalPred,
    ComplexPred, Even, Odd, PrimePred,
    Nonzero, Nonnegative, Nonpositive, Finite, Infinite, Zero, Bounded,
)

# Pattern matching, transformation, and pattern-based functions
from derive.patterns import (
    Pattern_, Integer_, Real_, Positive_, Negative_,
    NonNegative_, Symbol_,
    Rule, rule,
    Replace, ReplaceAll, ReplaceRepeated,
    MatchQ, Cases, Count, FreeQ, Position,
    PatternFunction, DefineFunction, FunctionRegistry,
)

# Custom types
from derive.types import (
    CustomType, DefineType,
    ComplexNumber, Quaternion, Vector3D,
)

# Additional SymPy exports for advanced use
from sympy import Eq

# Differential geometry module
from derive.diffgeo import *

# Variational calculus and Green's functions
from derive.calculus import (
    VariationalDerivative, EulerLagrangeEquation,
    GreenFunction, GreenFunctionPoisson1D, GreenFunctionHelmholtz1D,
    GreenFunctionLaplacian3D, GreenFunctionWave1D,
)

# Plotting module
from derive.plotting import *

# Notebook integration module
from derive.utils.notebook import *

# Optimization module (optional - requires cvxpy)
try:
    from derive.optimize import (
        OptVar, OptimizationProblem, Minimize, Maximize,
        Norm as OptNorm, Sum as OptSum, Quad, PositiveSemidefinite,
    )
    _HAS_CVXPY = True
except ImportError:
    _HAS_CVXPY = False

# Comprehensive __all__ for `from derive import *`
__all__ = [
    # Symbols
    'Symbol', 'symbols', 'Function',
    'Rational', 'Integer', 'Float',
    # Common types
    'Array', 'ImmutableDenseNDimArray', 'MutableDenseNDimArray',
    'Heaviside', 'DiracDelta',
    # Constants
    'Pi', 'E', 'I', 'Infinity', 'ComplexInfinity', 'Indeterminate',
    # Smart number handling
    'rationalize', 'to_rational', 'exact', 'numerical',
    'is_exact_mode', 'is_numerical_mode', 'get_numerical_precision',
    'auto_rational', 'SmartNumber', 'S', 'expr',
    'float_to_rational', 'ensure_rational',
    # Trigonometric
    'Sin', 'Cos', 'Tan', 'Cot', 'Sec', 'Csc',
    'Sinh', 'Cosh', 'Tanh', 'Coth', 'Sech', 'Csch',
    'ArcSin', 'ArcCos', 'ArcTan', 'ArcCot', 'ArcSec', 'ArcCsc',
    'ArcSinh', 'ArcCosh', 'ArcTanh', 'ArcCoth', 'ArcSech', 'ArcCsch',
    # Exponential/Log
    'Exp', 'Log', 'Ln', 'Sqrt', 'Power',
    # Complex
    'Re', 'Im', 'Conjugate', 'Arg', 'Abs',
    # Special functions (basic)
    'Factorial', 'Binomial', 'Gamma', 'Beta',
    # Special functions (Bessel)
    'BesselJ', 'BesselY', 'BesselI', 'BesselK',
    'HankelH1', 'HankelH2',
    'SphericalBesselJ', 'SphericalBesselY',
    # Special functions (Orthogonal polynomials)
    'LegendreP', 'AssociatedLegendreP',
    'ChebyshevT', 'ChebyshevU',
    'HermiteH', 'LaguerreL', 'AssociatedLaguerreL',
    'GegenbauerC', 'JacobiP',
    # Special functions (Spherical harmonics)
    'SphericalHarmonicY',
    # Special functions (Elliptic)
    'EllipticK', 'EllipticE', 'EllipticF', 'EllipticPi',
    # Special functions (Error functions)
    'Erf', 'Erfc', 'Erfi',
    # Special functions (Integrals)
    'ExpIntegralEi', 'SinIntegral', 'CosIntegral', 'LogIntegral',
    'FresnelS', 'FresnelC',
    # Special functions (Airy)
    'AiryAi', 'AiryBi',
    # Special functions (Number theory & QFT)
    'Zeta', 'PolyLog',
    # Special functions (Hypergeometric)
    'Hypergeometric2F1', 'HypergeometricPFQ', 'MeijerG',
    # Number functions
    'Sign', 'Floor', 'Ceiling', 'N', 'Round', 'Mod',
    'GCD', 'LCM', 'PrimeQ', 'Prime', 'FactorInteger',
    # Calculus
    'D', 'Integrate', 'NIntegrate', 'Limit', 'Series', 'Sum', 'Product',
    # Change of variables
    'ChangeVariables', 'IntegrateWithSubstitution',
    # Integral transforms
    'FourierTransform', 'InverseFourierTransform',
    'LaplaceTransform', 'InverseLaplaceTransform',
    'Convolve',
    # Differential equations
    'DSolve', 'NDSolve',
    # Solving
    'Solve', 'NSolve', 'FindRoot',
    # Simplification
    'Simplify', 'Expand', 'Factor', 'Collect',
    'Cancel', 'Apart', 'Together', 'TrigSimplify', 'TrigExpand', 'TrigReduce',
    'PowerSimplify', 'LogSimplify',
    # Linear algebra
    'Matrix', 'Dot', 'Transpose', 'Inverse', 'Det',
    'Eigenvalues', 'Eigenvectors',
    'IdentityMatrix', 'DiagonalMatrix', 'ZeroMatrix',
    'Tr', 'MatrixRank', 'NullSpace', 'RowReduce',
    'ConjugateTranspose', 'MatrixExp', 'CharacteristicPolynomial',
    # Lists
    'Table', 'Range', 'Map', 'Select', 'Sort', 'Total', 'Length',
    'First', 'Last', 'Take', 'Drop', 'Append', 'Prepend', 'Join',
    'Flatten', 'Partition',
    # Logic
    'Equal', 'Unequal', 'Less', 'LessEqual', 'Greater', 'GreaterEqual',
    'And', 'Or', 'Not', 'Xor', 'Nand', 'Nor',
    'If', 'Which', 'Switch', 'Piecewise', 'Max', 'Min',
    # Strings
    'StringJoin', 'StringLength', 'StringTake', 'StringDrop',
    'StringReplace', 'ToString',
    # Output & Display
    'TeXForm', 'PrettyForm', 'Print', 'RichPrint', 'RichLatex', 'TableForm', 'show',
    # Data I/O
    'Import', 'Export',
    # Probability and Statistics
    'NormalDistribution', 'UniformDistribution', 'ExponentialDistribution',
    'PoissonDistribution', 'BinomialDistribution',
    'PDF', 'CDF', 'Mean', 'Variance', 'StandardDeviation',
    'RandomVariate', 'Probability',
    # Lazy evaluation
    'Lazy', 'LazyExpr', 'lazy', 'force',
    # Caching
    'ExpressionCache', 'memoize', 'memoize_method', 'cached_simplify',
    'get_christoffel_cache', 'get_riemann_cache', 'clear_all_caches',
    # Composable API
    'Pipe', 'pipe', 'compose', 'thread_first', 'thread_last', 'Chainable',
    # Assumptions
    'Assuming', 'Refine', 'Ask', 'SimplifyWithAssumptions',
    'get_assumptions', 'assume_positive', 'assume_real', 'assume_integer',
    'AssumedSymbol',
    'Positive', 'Negative', 'Real', 'IntegerPred', 'RationalPred',
    'ComplexPred', 'Even', 'Odd', 'PrimePred',
    'Nonzero', 'Nonnegative', 'Nonpositive', 'Finite', 'Infinite', 'Zero', 'Bounded',
    # Pattern matching
    'Pattern_', 'Integer_', 'Real_', 'Positive_', 'Negative_',
    'NonNegative_', 'Symbol_',
    'Rule', 'rule',
    'Replace', 'ReplaceAll', 'ReplaceRepeated',
    'MatchQ', 'Cases', 'Count', 'FreeQ', 'Position',
    # Pattern-based functions
    'PatternFunction', 'DefineFunction', 'FunctionRegistry',
    # Custom types
    'CustomType', 'DefineType',
    'ComplexNumber', 'Quaternion', 'Vector3D',
    # SymPy passthrough for advanced use
    'Eq',
]

# Add optimization exports if cvxpy is available
if _HAS_CVXPY:
    __all__.extend([
        'OptVar', 'OptimizationProblem', 'Minimize', 'Maximize',
        'OptNorm', 'OptSum', 'Quad', 'PositiveSemidefinite',
    ])
