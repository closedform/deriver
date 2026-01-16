"""
Test suite for plotting.py module.

Tests for symbolic plotting functions.
"""

import pytest
from sympy import Symbol, pi, sin, cos
from derive.plotting import Plot, ListPlot, ListLinePlot, ParametricPlot
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for testing


class TestPlot:
    """Tests for Plot function."""

    def test_plot_symbolic(self):
        """Test plotting a symbolic expression."""
        x = Symbol('x')
        fig = Plot(sin(x), (x, 0, 2*pi))
        assert fig is not None
        assert len(fig.axes) == 1

    def test_plot_lambda(self):
        """Test plotting a lambda function."""
        import numpy as np
        fig = Plot(lambda x: np.sin(x), ('x', 0, 6.28))
        assert fig is not None

    def test_plot_multiple(self):
        """Test plotting multiple expressions."""
        x = Symbol('x')
        fig = Plot([sin(x), cos(x)], (x, 0, 2*pi))
        assert fig is not None
        # Should have two lines
        ax = fig.axes[0]
        assert len(ax.get_lines()) == 2

    def test_plot_with_options(self):
        """Test plotting with named options."""
        x = Symbol('x')
        fig = Plot(x**2, (x, -2, 2),
                  PlotLabel="Parabola",
                  PlotStyle='Red')
        assert fig is not None
        ax = fig.axes[0]
        assert ax.get_title() == "Parabola"

    def test_plot_with_plot_range(self):
        """Test PlotRange option."""
        x = Symbol('x')
        fig = Plot(x**2, (x, -2, 2), PlotRange=[(0, 4)])
        assert fig is not None


class TestListPlot:
    """Tests for ListPlot function."""

    def test_listplot_y_values(self):
        """Test ListPlot with y-values only."""
        fig = ListPlot([1, 4, 9, 16, 25])
        assert fig is not None
        ax = fig.axes[0]
        # Should have scatter points
        assert len(ax.collections) > 0

    def test_listplot_xy_pairs(self):
        """Test ListPlot with (x,y) pairs."""
        data = [[0, 0], [1, 1], [2, 4], [3, 9]]
        fig = ListPlot(data)
        assert fig is not None

    def test_listplot_with_options(self):
        """Test ListPlot with options."""
        fig = ListPlot([1, 4, 9, 16], PlotLabel="Squares", GridLines=True)
        assert fig is not None
        ax = fig.axes[0]
        assert ax.get_title() == "Squares"


class TestListLinePlot:
    """Tests for ListLinePlot function."""

    def test_listlineplot_basic(self):
        """Test ListLinePlot with y-values."""
        fig = ListLinePlot([1, 2, 4, 8, 16])
        assert fig is not None
        ax = fig.axes[0]
        # Should have lines
        assert len(ax.get_lines()) >= 1


class TestParametricPlot:
    """Tests for ParametricPlot function."""

    def test_parametricplot_circle(self):
        """Test parametric plot of a circle."""
        t = Symbol('t')
        fig = ParametricPlot([cos(t), sin(t)], (t, 0, 2*pi))
        assert fig is not None
        ax = fig.axes[0]
        assert len(ax.get_lines()) == 1


class TestPlotOptions:
    """Test plot options."""

    def test_plot_style_colors(self):
        """Test PlotStyle color options."""
        x = Symbol('x')
        # Test various color names
        for color in ['Red', 'Blue', 'Green']:
            fig = Plot(x, (x, 0, 1), PlotStyle=color)
            assert fig is not None

    def test_axes_label(self):
        """Test AxesLabel option."""
        x = Symbol('x')
        fig = Plot(x**2, (x, 0, 1), AxesLabel=['x', 'y'])
        ax = fig.axes[0]
        assert ax.get_xlabel() == 'x'
        assert ax.get_ylabel() == 'y'

    def test_image_size(self):
        """Test ImageSize option."""
        x = Symbol('x')
        fig = Plot(x, (x, 0, 1), ImageSize=400)
        assert fig is not None
