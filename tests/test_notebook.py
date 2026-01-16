"""
Test suite for notebook.py module.

Tests for Jupyter integration and LaTeX rendering.
"""

import pytest
from sympy import Symbol, sin, cos, latex
from derive.utils.notebook import (
    MathForm, DisplayForm, TableForm,
    create_notebook_template, save_notebook_template,
)
import tempfile
import json


class TestMathForm:
    """Tests for MathForm function."""

    def test_mathform_basic(self):
        """Test MathForm returns LaTeX string."""
        x = Symbol('x')
        result = MathForm(x**2 + 1)
        assert 'x' in result
        assert '2' in result
        # Should be valid LaTeX
        assert result == latex(x**2 + 1)

    def test_mathform_trig(self):
        """Test MathForm with trig functions."""
        x = Symbol('x')
        result = MathForm(sin(x))
        assert 'sin' in result or '\\sin' in result


class TestDisplayForm:
    """Tests for DisplayForm function."""

    def test_displayform_latex(self):
        """Test DisplayForm with latex option."""
        x = Symbol('x')
        result = DisplayForm(x**2, 'latex')
        assert 'x' in result

    def test_displayform_unicode(self):
        """Test DisplayForm with unicode option."""
        x = Symbol('x')
        result = DisplayForm(x**2, 'unicode')
        assert result is not None

    def test_displayform_ascii(self):
        """Test DisplayForm with ascii option."""
        x = Symbol('x')
        result = DisplayForm(x**2, 'ascii')
        assert result is not None


class TestTableForm:
    """Tests for TableForm function."""

    def test_tableform_basic(self):
        """Test TableForm with simple data."""
        data = [[1, 2], [3, 4]]
        result = TableForm(data)
        assert result is not None
        # In terminal mode, should contain pipe separators
        assert '|' in result or '<table' in result

    def test_tableform_with_headings(self):
        """Test TableForm with headings."""
        data = [[1, 2], [3, 4]]
        result = TableForm(data, headings=['a', 'b'])
        assert 'a' in result
        assert 'b' in result


class TestNotebookTemplate:
    """Tests for notebook template generation."""

    def test_create_notebook_template(self):
        """Test creating notebook template."""
        template = create_notebook_template()
        assert template is not None
        # Should be valid JSON
        notebook = json.loads(template)
        assert 'cells' in notebook
        assert 'metadata' in notebook
        assert notebook['nbformat'] == 4

    def test_save_notebook_template(self):
        """Test saving notebook template to file."""
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test_notebook.ipynb')
            result = save_notebook_template(path)
            assert os.path.exists(path)

            # Verify content
            with open(path) as f:
                notebook = json.load(f)
            assert 'cells' in notebook


class TestExportPDF:
    """Tests for PDF export functionality."""

    def test_export_plot_pdf(self):
        """Test exporting a plot to PDF."""
        import tempfile
        import os
        from derive import Export
        from derive.plotting import Plot
        from sympy import Symbol, sin, pi

        x = Symbol('x')
        fig = Plot(sin(x), (x, 0, 2*pi))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test_plot.pdf')
            Export(path, fig)
            assert os.path.exists(path)
            # PDF files start with %PDF
            with open(path, 'rb') as f:
                header = f.read(4)
            assert header == b'%PDF'


# =============================================================================
# Marimo Integration Tests
# =============================================================================

class TestMarimoTemplate:
    """Tests for Marimo notebook template generation."""

    def test_create_marimo_template(self):
        """Test creating Marimo template."""
        from derive.utils.notebook import create_marimo_template
        template = create_marimo_template()
        assert template is not None
        assert isinstance(template, str)
        # Template is valid marimo format (contains required elements)
        assert 'import marimo' in template
        assert 'marimo.App()' in template

    def test_marimo_template_has_imports(self):
        """Test Marimo template has correct imports."""
        from derive.utils.notebook import create_marimo_template
        template = create_marimo_template()
        assert 'import marimo' in template
        assert 'from derive import' in template

    def test_marimo_template_has_app(self):
        """Test Marimo template creates app."""
        from derive.utils.notebook import create_marimo_template
        template = create_marimo_template()
        assert 'marimo.App()' in template
        assert '@app.cell' in template

    def test_marimo_template_has_examples(self):
        """Test Marimo template includes example code."""
        from derive.utils.notebook import create_marimo_template
        template = create_marimo_template()
        # Should have calculus examples
        assert 'Sin' in template or 'Cos' in template
        assert 'Integrate' in template or 'D(' in template

    def test_save_marimo_template(self):
        """Test saving Marimo template to file."""
        import os
        from derive.utils.notebook import save_marimo_template

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test_notebook.py')
            result = save_marimo_template(path)
            assert os.path.exists(path)

            # Verify content has required marimo elements
            with open(path) as f:
                content = f.read()
            assert 'import marimo' in content
            assert '@app.cell' in content


class TestMarimoFunctions:
    """Tests for Marimo display functions."""

    def test_marimo_latex_returns_result(self):
        """Test marimo_latex returns some result."""
        from derive.utils.notebook import marimo_latex
        x = Symbol('x')
        result = marimo_latex(x**2 + 1)
        # Should return something (marimo object or string)
        assert result is not None

    def test_marimo_table_returns_result(self):
        """Test marimo_table returns some result."""
        from derive.utils.notebook import marimo_table
        data = [[1, 2], [3, 4]]
        result = marimo_table(data)
        # Should return something (marimo table or TableForm result)
        assert result is not None

    def test_marimo_table_with_headings(self):
        """Test marimo_table with column headings."""
        from derive.utils.notebook import marimo_table
        data = [[1, 2], [3, 4]]
        result = marimo_table(data, headings=['A', 'B'])
        assert result is not None


class TestDisplayFormExtended:
    """Extended tests for DisplayForm."""

    def test_displayform_tree(self):
        """Test DisplayForm with tree (srepr) option."""
        x = Symbol('x')
        result = DisplayForm(x**2, 'tree')
        # Tree form shows internal structure
        assert 'Symbol' in result or 'Pow' in result

    def test_displayform_unknown_form(self):
        """Test DisplayForm with unknown form falls back to str."""
        x = Symbol('x')
        result = DisplayForm(x**2, 'unknown_form')
        assert str(x**2) == result or 'x' in result
