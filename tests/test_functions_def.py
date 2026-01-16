"""
Tests for Pattern-Based Function Definitions (Phase 15).

Tests DefineFunction, PatternFunction, and pattern-based dispatch.
"""

import pytest
from sympy import Symbol, Integer, sin, cos

from derive import (
    DefineFunction, PatternFunction, FunctionRegistry,
    Pattern_, Integer_,
)


# Create symbols for testing
a, b, c = Symbol('a'), Symbol('b'), Symbol('c')
x, y, z = Symbol('x'), Symbol('y'), Symbol('z')


class TestDefineFunction:
    """Tests for DefineFunction."""

    def test_create_function(self):
        """DefineFunction creates PatternFunction."""
        f = DefineFunction('f')
        assert isinstance(f, PatternFunction)
        assert f.name == 'f'

    def test_define_simple_rule(self):
        """Define a simple transformation rule."""
        f = DefineFunction('square')
        x_ = Pattern_('x')
        f.define(x_, x_**2)

        result = f(Integer(3))
        assert result == 9

    def test_define_symbol_rule(self):
        """Define rule that works with symbols."""
        f = DefineFunction('square')
        x_ = Pattern_('x')
        f.define(x_, x_**2)

        result = f(a)
        assert result == a**2

    def test_specific_case_precedence(self):
        """More specific cases take precedence."""
        f = DefineFunction('factorial_like')
        x_ = Pattern_('x')

        # General case: f(x) = x * f(x-1)
        f.define(x_, x_ * f._func(x_ - 1))
        # Specific case: f(0) = 1
        f.define(Integer(0), Integer(1))

        # f(0) should return 1 (specific case)
        assert f(Integer(0)) == 1

    def test_multiple_rules(self):
        """Function can have multiple rules."""
        f = DefineFunction('abs_like')
        x_ = Pattern_('x')

        f.define(Integer(0), Integer(0))
        f.define(x_, x_)  # General case

        assert f(Integer(0)) == 0
        assert f(Integer(5)) == 5
        assert f(a) == a

    def test_clear_rules(self):
        """Clear removes all rules."""
        f = DefineFunction('test')
        x_ = Pattern_('x')
        f.define(x_, x_**2)

        f.clear()
        assert len(f.rules()) == 0


class TestPatternFunction:
    """Tests for PatternFunction class."""

    def test_repr(self):
        """PatternFunction has readable repr."""
        f = PatternFunction('test')
        assert 'test' in repr(f)
        assert '0 rules' in repr(f)

    def test_chaining(self):
        """Define returns self for chaining."""
        f = PatternFunction('test')
        x_ = Pattern_('x')
        result = f.define(x_, x_**2)
        assert result is f

    def test_rules_list(self):
        """rules() returns defined rules."""
        f = PatternFunction('test')
        x_ = Pattern_('x')
        f.define(x_, x_**2)
        f.define(Integer(0), Integer(1))

        rules = f.rules()
        assert len(rules) == 2


class TestConditionalDefinitions:
    """Tests for conditional function definitions."""

    def test_conditional_rule(self):
        """Rule with condition."""
        f = DefineFunction('pos_square')
        x_ = Pattern_('x')

        # Condition that always passes (for testing mechanism)
        def always_true(match):
            return True

        f.define(x_, x_**2, condition=always_true)

        # Works for general symbols
        assert f(a) == a**2

    def test_conditional_rule_with_numeric(self):
        """Rule with numeric condition."""
        f = DefineFunction('abs_val')
        x_ = Pattern_('x')

        # Define for positive integers
        def is_positive_int(match):
            val = match.get(x_)
            return val is not None and val.is_integer and val.is_positive

        f.define(x_, x_, condition=is_positive_int)

        # Test with positive integer
        result = f(Integer(5))
        assert result == 5


class TestFunctionRegistry:
    """Tests for FunctionRegistry."""

    def test_register_and_get(self):
        """Register and retrieve functions."""
        FunctionRegistry.clear()

        f = DefineFunction('registered_f')
        FunctionRegistry.register(f)

        retrieved = FunctionRegistry.get('registered_f')
        assert retrieved is f

    def test_get_nonexistent(self):
        """Get returns None for unknown functions."""
        FunctionRegistry.clear()
        assert FunctionRegistry.get('unknown') is None

    def test_all_functions(self):
        """List all registered functions."""
        FunctionRegistry.clear()

        f = DefineFunction('func1')
        g = DefineFunction('func2')
        FunctionRegistry.register(f)
        FunctionRegistry.register(g)

        names = FunctionRegistry.all_functions()
        assert 'func1' in names
        assert 'func2' in names

    def test_clear(self):
        """Clear removes all functions."""
        f = DefineFunction('test')
        FunctionRegistry.register(f)
        FunctionRegistry.clear()

        assert len(FunctionRegistry.all_functions()) == 0


class TestIntegrationScenarios:
    """Integration tests for realistic use cases."""

    def test_piecewise_function(self):
        """Define a piecewise-like function."""
        f = DefineFunction('sign_like')
        x_ = Pattern_('x')

        # Define specific cases first (higher specificity)
        f.define(Integer(0), Integer(0))
        # General case
        f.define(x_, x_)

        assert f(Integer(0)) == 0
        assert f(Integer(5)) == 5
        assert f(Integer(-3)) == -3

    def test_recursive_definition_base_case(self):
        """Define base case of recursive function."""
        f = DefineFunction('fib_base')

        # Base cases
        f.define(Integer(0), Integer(0))
        f.define(Integer(1), Integer(1))

        assert f(Integer(0)) == 0
        assert f(Integer(1)) == 1

    def test_derivative_like_rule(self):
        """Define derivative-like transformation."""
        d = DefineFunction('deriv')
        x_ = Pattern_('x')
        n_ = Pattern_('n')

        # d/dx(x^n) = n*x^(n-1)
        d.define(x_**n_, n_ * x_**(n_ - 1))

        # Test with concrete values
        result = d(a**3)
        assert result == 3 * a**2

    def test_trig_simplification_rule(self):
        """Define trig simplification rules."""
        simp = DefineFunction('trig_simp')
        x_ = Pattern_('x')

        # sin^2 + cos^2 = 1
        simp.define(sin(x_)**2 + cos(x_)**2, Integer(1))

        result = simp(sin(a)**2 + cos(a)**2)
        assert result == 1


class TestEdgeCases:
    """Edge case tests."""

    def test_no_rules_defined(self):
        """Function with no rules returns unevaluated."""
        f = DefineFunction('empty')
        result = f(Integer(5))
        # Should return unevaluated function call
        assert 'empty' in str(result)

    def test_no_matching_rule(self):
        """When no rule matches, return unevaluated."""
        f = DefineFunction('partial')
        f.define(Integer(0), Integer(1))

        # This doesn't match the rule
        result = f(Integer(5))
        assert 'partial' in str(result)

    def test_function_with_expressions(self):
        """Function can be called with complex expressions."""
        f = DefineFunction('apply')
        x_ = Pattern_('x')
        f.define(x_, x_**2)

        result = f(a + b)
        assert result == (a + b)**2
