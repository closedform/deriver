"""
Test suite for the derive API module.

These tests verify core symbolic computation functions.
"""

import pytest
from derive import *


class TestBasicFunctions:
    """Test basic mathematical functions."""

    def test_sin_cos(self):
        x = Symbol('x')
        assert Sin(0) == 0
        assert Cos(0) == 1
        assert Sin(Pi/2) == 1
        assert Cos(Pi) == -1

    def test_exp_log(self):
        x = Symbol('x')
        assert Exp(0) == 1
        assert Log(1) == 0
        assert Log(E) == 1

    def test_sqrt(self):
        assert Sqrt(4) == 2
        assert Sqrt(9) == 3


class TestCalculus:
    """Test calculus operations."""

    def test_differentiation(self):
        x = Symbol('x')
        # D[x^3, x] = 3x^2
        assert D(x**3, x) == 3*x**2
        # D[Sin[x], x] = Cos[x]
        assert D(Sin(x), x) == Cos(x)
        # D[x^5, {x, 2}] = 20x^3
        assert D(x**5, (x, 2)) == 20*x**3

    def test_integration_indefinite(self):
        x = Symbol('x')
        # Integrate[x^2, x] = x^3/3
        result = Integrate(x**2, x)
        assert result == x**3/3

    def test_integration_definite(self):
        x = Symbol('x')
        # Integrate[x, {x, 0, 1}] = 1/2
        result = Integrate(x, (x, 0, 1))
        from sympy import Rational
        assert result == Rational(1, 2)

    def test_integration_trig(self):
        x = Symbol('x')
        # Integrate[Sin[x], {x, 0, Pi}] = 2
        result = Integrate(Sin(x), (x, 0, Pi))
        assert result == 2

    def test_limit(self):
        x = Symbol('x')
        # Limit[Sin[x]/x, x -> 0] = 1
        assert Limit(Sin(x)/x, x, 0) == 1

    def test_series(self):
        x = Symbol('x')
        # Series[Exp[x], {x, 0, 3}]
        result = Series(Exp(x), (x, 0, 3))
        # Should contain 1 + x + x^2/2 + x^3/6
        assert result.removeO() == 1 + x + x**2/2 + x**3/6

    def test_sum(self):
        i = Symbol('i')
        # Sum[i, {i, 1, 10}] = 55
        result = Sum(i, (i, 1, 10))
        assert result == 55

    def test_sum_symbolic(self):
        i, n = symbols('i n')
        # Sum[i, {i, 1, n}] = n(n+1)/2
        result = Sum(i, (i, 1, n))
        # Result is equivalent to n*(n+1)/2 but may be expanded
        from sympy import simplify
        assert simplify(result - n*(n + 1)/2) == 0

    def test_product_factorial(self):
        i = Symbol('i')
        # Product[i, {i, 1, 5}] = 5! = 120
        result = Product(i, (i, 1, 5))
        assert result == 120

    def test_product_symbolic(self):
        i, n = symbols('i n')
        # Product[i, {i, 1, n}] = n!
        result = Product(i, (i, 1, n))
        from sympy import factorial
        assert result == factorial(n)


class TestNumericalIntegration:
    """Test numerical integration."""

    def test_nintegrate_basic(self):
        x = Symbol('x')
        # NIntegrate[x^2, {x, 0, 1}] ≈ 1/3
        result = NIntegrate(x**2, (x, 0, 1))
        assert abs(result - 1/3) < 1e-10

    def test_nintegrate_gaussian(self):
        x = Symbol('x')
        import numpy as np
        # NIntegrate[Exp[-x^2], {x, 0, Infinity}] = sqrt(pi)/2
        result = NIntegrate(Exp(-x**2), (x, 0, Infinity))
        expected = np.sqrt(np.pi) / 2
        assert abs(result - expected) < 1e-6


class TestDifferentialEquations:
    """Test differential equation solvers."""

    def test_dsolve_exponential(self):
        x = Symbol('x')
        y = Function('y')
        from sympy import Eq, exp
        # y' = y has solution y = C1*exp(x)
        eq = Eq(y(x).diff(x), y(x))
        result = DSolve(eq, y(x), x)
        # Result is an Eq object: y(x) = C1*exp(x)
        assert result.has(exp(x))

    def test_ndsolve_exponential_decay(self):
        """Test NDSolve with exponential decay: dy/dt = -y"""
        import numpy as np
        # dy/dt = -y with y(0) = 1
        sol = NDSolve(lambda t, y: -y, 1.0, ('t', 0, 5))
        # At t=5, y should be approximately exp(-5)
        assert abs(sol.y[0][-1] - np.exp(-5)) < 0.01

    def test_ndsolve_simple_growth(self):
        """Test NDSolve with linear growth: dy/dt = 1"""
        # dy/dt = 1 with y(0) = 0 -> y = t
        sol = NDSolve(lambda t, y: 1, 0.0, ('t', 0, 5))
        # At t=5, y should be approximately 5
        assert abs(sol.y[0][-1] - 5.0) < 0.1


class TestSolving:
    """Test equation solving."""

    def test_solve_quadratic(self):
        x = Symbol('x')
        # x^2 - 4 = 0 has solutions x = -2, 2
        result = Solve(x**2 - 4, x)
        solutions = [sol[x] for sol in result]
        assert -2 in solutions
        assert 2 in solutions

    def test_solve_cubic(self):
        x = Symbol('x')
        # x^3 - 6x^2 + 11x - 6 = 0 has solutions 1, 2, 3
        result = Solve(x**3 - 6*x**2 + 11*x - 6, x)
        solutions = [sol[x] for sol in result]
        assert 1 in solutions
        assert 2 in solutions
        assert 3 in solutions

    def test_findroot(self):
        x = Symbol('x')
        import math
        # Find root of x^2 - 2 starting from 1 -> sqrt(2)
        result = FindRoot(x**2 - 2, (x, 1))
        assert abs(result[x] - math.sqrt(2)) < 1e-6


class TestSimplification:
    """Test simplification functions."""

    def test_expand(self):
        x, y = symbols('x y')
        # Expand[(x+y)^2] = x^2 + 2xy + y^2
        result = Expand((x + y)**2)
        assert result == x**2 + 2*x*y + y**2

    def test_factor(self):
        x = Symbol('x')
        # Factor[x^2 - 4] = (x-2)(x+2)
        result = Factor(x**2 - 4)
        assert result == (x - 2)*(x + 2)

    def test_simplify_trig(self):
        x = Symbol('x')
        # Sin^2 + Cos^2 = 1
        expr = Sin(x)**2 + Cos(x)**2
        result = Simplify(expr)
        assert result == 1


class TestLinearAlgebra:
    """Test linear algebra functions."""

    def test_matrix_creation(self):
        m = Matrix([[1, 2], [3, 4]])
        assert m.shape == (2, 2)

    def test_identity_matrix(self):
        I3 = IdentityMatrix(3)
        assert I3.shape == (3, 3)
        assert I3[0, 0] == 1
        assert I3[0, 1] == 0

    def test_dot_product(self):
        m1 = Matrix([[1, 2], [3, 4]])
        m2 = Matrix([[5], [6]])
        result = Dot(m1, m2)
        assert result == Matrix([[17], [39]])

    def test_transpose(self):
        m = Matrix([[1, 2], [3, 4]])
        result = Transpose(m)
        assert result == Matrix([[1, 3], [2, 4]])

    def test_inverse(self):
        m = Matrix([[1, 2], [3, 4]])
        inv = Inverse(m)
        # m * inv = I
        product = m * inv
        assert product == IdentityMatrix(2)

    def test_determinant(self):
        m = Matrix([[1, 2], [3, 4]])
        assert Det(m) == -2

    def test_eigenvalues(self):
        m = Matrix([[1, 2], [2, 1]])
        eigs = Eigenvalues(m)
        # Eigenvalues are -1 and 3
        assert -1 in eigs
        assert 3 in eigs


class TestListOperations:
    """Test list manipulation functions."""

    def test_range(self):
        assert Range(5) == [1, 2, 3, 4, 5]
        assert Range(2, 5) == [2, 3, 4, 5]

    def test_table(self):
        i = Symbol('i')
        result = Table(i**2, (i, 1, 5))
        assert result == [1, 4, 9, 16, 25]

    def test_map(self):
        result = Map(lambda x: x**2, [1, 2, 3])
        assert result == [1, 4, 9]

    def test_select(self):
        result = Select([1, 2, 3, 4, 5], lambda x: x > 2)
        assert result == [3, 4, 5]

    def test_total(self):
        assert Total([1, 2, 3, 4, 5]) == 15

    def test_length(self):
        assert Length([1, 2, 3]) == 3

    def test_first_last(self):
        lst = [1, 2, 3, 4, 5]
        assert First(lst) == 1
        assert Last(lst) == 5

    def test_take_drop(self):
        lst = [1, 2, 3, 4, 5]
        assert Take(lst, 3) == [1, 2, 3]
        assert Drop(lst, 2) == [3, 4, 5]

    def test_join(self):
        assert Join([1, 2], [3, 4]) == [1, 2, 3, 4]

    def test_flatten(self):
        nested = [[1, 2], [3, [4, 5]]]
        assert Flatten(nested) == [1, 2, 3, 4, 5]

    def test_partition(self):
        result = Partition([1, 2, 3, 4, 5, 6], 2)
        assert result == [[1, 2], [3, 4], [5, 6]]


class TestLogic:
    """Test logic and conditionals."""

    def test_if(self):
        assert If(True, 1, 2) == 1
        assert If(False, 1, 2) == 2

    def test_which(self):
        x = 5
        result = Which(x < 0, "negative", x == 0, "zero", x > 0, "positive")
        assert result == "positive"

    def test_switch(self):
        result = Switch(2, 1, "one", 2, "two", 3, "three")
        assert result == "two"


class TestStringOperations:
    """Test string functions."""

    def test_string_join(self):
        assert StringJoin("Hello", " ", "World") == "Hello World"

    def test_string_length(self):
        assert StringLength("Hello") == 5

    def test_string_take_drop(self):
        s = "Hello World"
        assert StringTake(s, 5) == "Hello"
        assert StringDrop(s, 6) == "World"


class TestOutput:
    """Test output functions."""

    def test_texform(self):
        x = Symbol('x')
        result = TeXForm(x**2 + 1)
        assert 'x' in result
        assert '2' in result


class TestConstants:
    """Test mathematical constants."""

    def test_pi(self):
        import math
        # Float comparison for Pi
        assert abs(float(Pi) - math.pi) < 1e-10

    def test_e(self):
        import math
        assert abs(float(E) - math.e) < 1e-10

    def test_infinity(self):
        from sympy import oo
        assert Infinity == oo


class TestProbability:
    """Test probability and statistics functions."""

    def test_normal_distribution(self):
        dist = NormalDistribution(0, 1)
        assert dist is not None
        # PDF at mean should be highest
        assert PDF(dist, 0) > PDF(dist, 2)

    def test_pdf_normal(self):
        import math
        dist = NormalDistribution(0, 1)
        # PDF at 0 for standard normal is 1/sqrt(2*pi)
        expected = 1 / math.sqrt(2 * math.pi)
        assert abs(PDF(dist, 0) - expected) < 1e-10

    def test_cdf_normal(self):
        dist = NormalDistribution(0, 1)
        # CDF at 0 for standard normal is 0.5
        assert abs(CDF(dist, 0) - 0.5) < 1e-10

    def test_mean_distribution(self):
        dist = NormalDistribution(5, 2)
        assert Mean(dist) == 5.0

    def test_variance_distribution(self):
        dist = NormalDistribution(0, 2)
        assert Variance(dist) == 4.0

    def test_mean_data(self):
        data = [1, 2, 3, 4, 5]
        assert Mean(data) == 3.0

    def test_uniform_distribution(self):
        dist = UniformDistribution(0, 1)
        assert Mean(dist) == 0.5

    def test_random_variate(self):
        dist = NormalDistribution(0, 1)
        samples = RandomVariate(dist, 100)
        assert len(samples) == 100

    def test_poisson_distribution(self):
        dist = PoissonDistribution(5)
        assert Mean(dist) == 5.0

    def test_binomial_distribution(self):
        dist = BinomialDistribution(10, 0.5)
        assert Mean(dist) == 5.0
