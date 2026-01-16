"""
Tests for derive.utils.display module.

Tests display formatting functions including TeXForm, PrettyForm,
rich terminal output, and table formatting.
"""

import pytest
from sympy import Symbol, sin, cos, sqrt, Rational, pi

from derive.utils.display import (
    TeXForm, PrettyForm, Print, RichPrint, RichLatex, TableForm, show
)


# =============================================================================
# TeXForm Tests
# =============================================================================

class TestTeXForm:
    """Tests for LaTeX conversion."""

    def test_simple_expression(self):
        """Test LaTeX conversion of simple expression."""
        x = Symbol('x')
        result = TeXForm(x**2 + 1)
        assert 'x^{2}' in result or 'x^2' in result

    def test_fraction(self):
        """Test LaTeX conversion of fraction."""
        x = Symbol('x')
        result = TeXForm(x / 2)
        assert 'frac' in result or '/' in result

    def test_sqrt(self):
        """Test LaTeX conversion of square root."""
        x = Symbol('x')
        result = TeXForm(sqrt(x))
        assert 'sqrt' in result

    def test_trig_functions(self):
        """Test LaTeX conversion of trig functions."""
        x = Symbol('x')
        result = TeXForm(sin(x) + cos(x))
        assert 'sin' in result
        assert 'cos' in result

    def test_greek_symbols(self):
        """Test LaTeX conversion preserves Greek symbols."""
        alpha = Symbol('alpha')
        result = TeXForm(alpha**2)
        assert 'alpha' in result

    def test_pi_constant(self):
        """Test LaTeX conversion of pi."""
        result = TeXForm(pi)
        assert 'pi' in result


# =============================================================================
# PrettyForm Tests
# =============================================================================

class TestPrettyForm:
    """Tests for pretty printing."""

    def test_simple_expression(self):
        """Test pretty printing of simple expression."""
        x = Symbol('x')
        result = PrettyForm(x**2 + 1)
        assert '2' in result  # Exponent should be shown
        assert 'x' in result

    def test_unicode_mode(self):
        """Test unicode pretty printing."""
        x = Symbol('x')
        result = PrettyForm(sqrt(x), use_unicode=True)
        assert 'x' in result

    def test_ascii_mode(self):
        """Test ASCII pretty printing."""
        x = Symbol('x')
        result = PrettyForm(sqrt(x), use_unicode=False)
        assert 'x' in result

    def test_fraction_pretty(self):
        """Test pretty printing of fractions."""
        x = Symbol('x')
        result = PrettyForm(Rational(1, 2) * x)
        assert 'x' in result

    def test_multiline_output(self):
        """Test that complex expressions can produce multiline output."""
        x = Symbol('x')
        result = PrettyForm(x**2 / (x + 1))
        assert 'x' in result


# =============================================================================
# Print Tests
# =============================================================================

class TestPrint:
    """Tests for Print function."""

    def test_print_basic(self, capsys):
        """Test basic print functionality."""
        x = Symbol('x')
        Print(x**2)
        captured = capsys.readouterr()
        assert 'x' in captured.out
        assert '2' in captured.out

    def test_print_multiple_args(self, capsys):
        """Test printing multiple arguments."""
        x, y = Symbol('x'), Symbol('y')
        Print(x, y)
        captured = capsys.readouterr()
        assert 'x' in captured.out
        assert 'y' in captured.out

    def test_print_latex_mode(self, capsys):
        """Test print with latex mode."""
        x = Symbol('x')
        Print(x**2, latex_mode=True)
        captured = capsys.readouterr()
        # Should contain LaTeX formatting
        assert 'x' in captured.out

    def test_print_pretty_mode(self, capsys):
        """Test print with pretty mode."""
        x = Symbol('x')
        Print(x**2, pretty=True)
        captured = capsys.readouterr()
        assert 'x' in captured.out


# =============================================================================
# RichPrint Tests
# =============================================================================

class TestRichPrint:
    """Tests for RichPrint function."""

    def test_rich_print_basic(self, capsys):
        """Test basic rich printing."""
        x = Symbol('x')
        RichPrint(x**2)
        captured = capsys.readouterr()
        # Should output something (rich or fallback)
        assert 'x' in captured.out

    def test_rich_print_with_style(self, capsys):
        """Test rich printing with style."""
        x = Symbol('x')
        RichPrint(x**2, style="bold")
        captured = capsys.readouterr()
        assert 'x' in captured.out

    def test_rich_print_multiple(self, capsys):
        """Test rich printing multiple args."""
        x, y = Symbol('x'), Symbol('y')
        RichPrint(x, y)
        captured = capsys.readouterr()
        assert 'x' in captured.out
        assert 'y' in captured.out


# =============================================================================
# RichLatex Tests
# =============================================================================

class TestRichLatex:
    """Tests for RichLatex function."""

    def test_rich_latex_basic(self, capsys):
        """Test basic rich LaTeX output."""
        x = Symbol('x')
        RichLatex(x**2)
        captured = capsys.readouterr()
        # Should contain LaTeX or expression
        assert 'x' in captured.out


# =============================================================================
# TableForm Tests
# =============================================================================

class TestTableForm:
    """Tests for TableForm function."""

    def test_table_2d_list(self, capsys):
        """Test table from 2D list."""
        data = [[1, 2], [3, 4]]
        TableForm(data)
        captured = capsys.readouterr()
        # Should contain all values
        assert '1' in captured.out
        assert '2' in captured.out
        assert '3' in captured.out
        assert '4' in captured.out

    def test_table_with_headers(self, capsys):
        """Test table with headers."""
        data = [[1, 2], [3, 4]]
        headers = ['A', 'B']
        TableForm(data, headers=headers)
        captured = capsys.readouterr()
        assert 'A' in captured.out
        assert 'B' in captured.out

    def test_table_single_column(self, capsys):
        """Test table with single column data."""
        data = [[1], [2], [3]]
        TableForm(data)
        captured = capsys.readouterr()
        assert '1' in captured.out
        assert '2' in captured.out
        assert '3' in captured.out

    def test_table_symbolic_data(self, capsys):
        """Test table with symbolic expressions."""
        x = Symbol('x')
        data = [[x, x**2], [2*x, x + 1]]
        TableForm(data)
        captured = capsys.readouterr()
        assert 'x' in captured.out


# =============================================================================
# show() Tests
# =============================================================================

class TestShow:
    """Tests for show function."""

    def test_show_plain_mode(self, capsys):
        """Test show with plain mode."""
        x = Symbol('x')
        show(x**2, mode="plain")
        captured = capsys.readouterr()
        assert 'x' in captured.out

    def test_show_latex_mode(self, capsys):
        """Test show with latex mode."""
        x = Symbol('x')
        show(x**2, mode="latex")
        captured = capsys.readouterr()
        assert 'x' in captured.out

    def test_show_pretty_mode(self, capsys):
        """Test show with pretty mode."""
        x = Symbol('x')
        show(x**2, mode="pretty")
        captured = capsys.readouterr()
        assert 'x' in captured.out

    def test_show_auto_mode(self, capsys):
        """Test show with auto mode (default)."""
        x = Symbol('x')
        show(x**2)  # Auto mode
        captured = capsys.readouterr()
        # Should produce output
        assert 'x' in captured.out


# =============================================================================
# Integration Tests
# =============================================================================

class TestDisplayIntegration:
    """Integration tests for display module."""

    def test_complex_expression_all_modes(self, capsys):
        """Test a complex expression through all display modes."""
        x = Symbol('x')
        expr = sin(x)**2 + cos(x)**2

        # All modes should work without error
        tex = TeXForm(expr)
        pretty = PrettyForm(expr)
        Print(expr)
        RichPrint(expr)
        show(expr)

        assert 'sin' in tex or 'cos' in tex
        assert 'sin' in pretty.lower() or 'cos' in pretty.lower()

    def test_matrix_like_data(self, capsys):
        """Test displaying matrix-like data."""
        data = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        TableForm(data, headers=['x', 'y', 'z'])
        captured = capsys.readouterr()
        assert '1' in captured.out
        assert '0' in captured.out
