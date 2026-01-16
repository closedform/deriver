"""
plotting.py - Plotting Functions

This module provides Plot and ListPlot functions with syntax.
Uses matplotlib under the hood.
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import sympy as sp
from typing import Union, List, Tuple, Dict, Optional, Callable, Any

# Use non-interactive backend by default for better notebook compatibility
matplotlib.use('Agg')


def _parse_plot_options(options: Dict) -> Dict:
    """
    Convert plot options to matplotlib options.

    Derive uses: PlotRange, PlotStyle, PlotLabel
    matplotlib uses: xlim/ylim, color, title, etc.
    """
    mpl_opts = {}

    if 'PlotRange' in options:
        pr = options['PlotRange']
        if pr == 'All' or pr == 'Automatic':
            pass  # matplotlib handles this automatically
        elif isinstance(pr, (list, tuple)):
            if len(pr) == 2:
                # Could be y-range only or both
                if isinstance(pr[0], (list, tuple)):
                    mpl_opts['xlim'] = pr[0]
                    mpl_opts['ylim'] = pr[1]
                else:
                    mpl_opts['ylim'] = pr

    if 'PlotStyle' in options:
        style = options['PlotStyle']
        # Handle color names
        color_map = {
            'Red': 'red', 'Blue': 'blue', 'Green': 'green',
            'Orange': 'orange', 'Purple': 'purple', 'Black': 'black',
            'Gray': 'gray', 'White': 'white', 'Yellow': 'yellow',
            'Cyan': 'cyan', 'Magenta': 'magenta',
        }
        if isinstance(style, str):
            mpl_opts['color'] = color_map.get(style, style.lower())
        elif isinstance(style, dict):
            mpl_opts.update(style)

    if 'PlotLabel' in options:
        mpl_opts['title'] = options['PlotLabel']

    if 'AxesLabel' in options:
        labels = options['AxesLabel']
        if isinstance(labels, (list, tuple)) and len(labels) == 2:
            mpl_opts['xlabel'] = labels[0]
            mpl_opts['ylabel'] = labels[1]

    if 'PlotLegends' in options:
        mpl_opts['legend'] = options['PlotLegends']

    if 'GridLines' in options:
        mpl_opts['grid'] = options['GridLines'] not in (None, False, 'None')

    if 'AspectRatio' in options:
        ar = options['AspectRatio']
        if ar == 'Automatic':
            mpl_opts['aspect'] = 'auto'
        elif isinstance(ar, (int, float)):
            mpl_opts['aspect'] = ar

    if 'ImageSize' in options:
        size = options['ImageSize']
        if isinstance(size, (int, float)):
            mpl_opts['figsize'] = (size/100, size/100)
        elif isinstance(size, (list, tuple)):
            mpl_opts['figsize'] = (size[0]/100, size[1]/100)

    return mpl_opts


def Plot(expr_or_func, var_range, **options):
    """
    Plot function.

    Plot[f, {x, xmin, xmax}] - plot function f from xmin to xmax
    Plot[{f1, f2}, {x, xmin, xmax}] - plot multiple functions

    Args:
        expr_or_func: SymPy expression, Python function, or list of these
        var_range: Tuple (var, xmin, xmax) specifying variable and range
        **options: options like PlotRange, PlotStyle, PlotLabel

    Returns:
        matplotlib Figure object

    Examples:
        >>> x = Symbol('x')
        >>> fig = Plot(Sin(x), (x, 0, 2*Pi))
        >>> fig = Plot([Sin(x), Cos(x)], (x, 0, 2*Pi), PlotLabel="Trig Functions")
        >>> fig = Plot(x**2, (x, -2, 2), PlotStyle='Red', PlotRange=[0, 4])
    """
    # Parse options
    mpl_opts = _parse_plot_options(options)

    # Create figure
    figsize = mpl_opts.pop('figsize', (8, 6))
    fig, ax = plt.subplots(figsize=figsize)

    # Parse variable range
    if isinstance(var_range, (list, tuple)) and len(var_range) == 3:
        var, xmin, xmax = var_range
        # Convert to float if symbolic
        xmin_val = float(xmin) if hasattr(xmin, 'evalf') else float(xmin)
        xmax_val = float(xmax) if hasattr(xmax, 'evalf') else float(xmax)
    else:
        raise ValueError("var_range must be (var, xmin, xmax)")

    # Generate x values
    x_vals = np.linspace(xmin_val, xmax_val, 1000)

    # Handle single expression or list of expressions
    exprs = expr_or_func if isinstance(expr_or_func, list) else [expr_or_func]

    # Get color for single plot or cycle for multiple
    color = mpl_opts.pop('color', None)
    colors = [color] if color else plt.rcParams['axes.prop_cycle'].by_key()['color']

    legend_labels = mpl_opts.pop('legend', None)

    for i, expr in enumerate(exprs):
        # Convert expression to numerical function
        if hasattr(expr, 'free_symbols'):
            # SymPy expression
            f = sp.lambdify(var, expr, modules=['numpy'])
        elif callable(expr):
            # Already a function
            f = expr
        else:
            raise ValueError(f"Cannot plot: {expr}")

        # Compute y values
        y_vals = f(x_vals)

        # Plot
        plot_color = colors[i % len(colors)]
        label = legend_labels[i] if legend_labels and i < len(legend_labels) else None
        ax.plot(x_vals, y_vals, color=plot_color, label=label)

    # Apply remaining options
    if 'xlim' in mpl_opts:
        ax.set_xlim(mpl_opts['xlim'])
    if 'ylim' in mpl_opts:
        ax.set_ylim(mpl_opts['ylim'])
    if 'title' in mpl_opts:
        ax.set_title(mpl_opts['title'])
    if 'xlabel' in mpl_opts:
        ax.set_xlabel(mpl_opts['xlabel'])
    if 'ylabel' in mpl_opts:
        ax.set_ylabel(mpl_opts['ylabel'])
    if mpl_opts.get('grid'):
        ax.grid(True)
    if legend_labels:
        ax.legend()

    plt.tight_layout()
    return fig


def ListPlot(data, **options):
    """
    ListPlot function.

    ListPlot[{y1, y2, ...}] - plot points at (1,y1), (2,y2), ...
    ListPlot[{{x1,y1}, {x2,y2}, ...}] - plot points at given coordinates
    ListPlot[{list1, list2, ...}] - plot multiple datasets

    Args:
        data: List of y-values, list of (x,y) pairs, or list of lists
        **options: options

    Returns:
        matplotlib Figure object

    Examples:
        >>> fig = ListPlot([1, 4, 9, 16, 25])
        >>> fig = ListPlot([[0,0], [1,1], [2,4], [3,9]])
        >>> fig = ListPlot([[1,2,3], [1,4,9]], PlotStyle=['Blue', 'Red'])
    """
    mpl_opts = _parse_plot_options(options)

    figsize = mpl_opts.pop('figsize', (8, 6))
    fig, ax = plt.subplots(figsize=figsize)

    # Determine data format
    if not data:
        return fig

    # Check if it's multiple datasets
    if isinstance(data[0], list) and len(data[0]) > 0:
        if isinstance(data[0][0], (list, tuple)):
            # Single dataset of (x,y) pairs
            datasets = [data]
        elif isinstance(data[0][0], (int, float, np.number)):
            # Could be multiple y-value lists or single list of (x,y)
            if len(data[0]) == 2 and not isinstance(data[1][0], (list, tuple)):
                # Likely list of (x,y) pairs
                datasets = [data]
            else:
                # Multiple y-value lists
                datasets = data
        else:
            datasets = [data]
    else:
        # Single list of y-values
        datasets = [data]

    color = mpl_opts.pop('color', None)
    colors = [color] if color else plt.rcParams['axes.prop_cycle'].by_key()['color']

    if 'PlotStyle' in options and isinstance(options['PlotStyle'], list):
        colors = options['PlotStyle']

    legend_labels = mpl_opts.pop('legend', None)

    for i, dataset in enumerate(datasets):
        # Determine if (x,y) pairs or just y values
        if isinstance(dataset[0], (list, tuple)) and len(dataset[0]) == 2:
            # (x, y) pairs
            x_vals = [p[0] for p in dataset]
            y_vals = [p[1] for p in dataset]
        else:
            # Just y values
            x_vals = list(range(1, len(dataset) + 1))
            y_vals = list(dataset)

        plot_color = colors[i % len(colors)]
        if isinstance(plot_color, str) and plot_color[0].isupper():
            plot_color = plot_color.lower()

        label = legend_labels[i] if legend_labels and i < len(legend_labels) else None
        ax.scatter(x_vals, y_vals, color=plot_color, label=label)

    # Apply options
    if 'xlim' in mpl_opts:
        ax.set_xlim(mpl_opts['xlim'])
    if 'ylim' in mpl_opts:
        ax.set_ylim(mpl_opts['ylim'])
    if 'title' in mpl_opts:
        ax.set_title(mpl_opts['title'])
    if 'xlabel' in mpl_opts:
        ax.set_xlabel(mpl_opts['xlabel'])
    if 'ylabel' in mpl_opts:
        ax.set_ylabel(mpl_opts['ylabel'])
    if mpl_opts.get('grid'):
        ax.grid(True)
    if legend_labels:
        ax.legend()

    plt.tight_layout()
    return fig


def ListLinePlot(data, **options):
    """
    ListLinePlot - like ListPlot but with connected lines.

    Same arguments as ListPlot.
    """
    mpl_opts = _parse_plot_options(options)

    figsize = mpl_opts.pop('figsize', (8, 6))
    fig, ax = plt.subplots(figsize=figsize)

    if not data:
        return fig

    # Similar logic to ListPlot
    if isinstance(data[0], list) and len(data[0]) > 0:
        if isinstance(data[0][0], (list, tuple)):
            datasets = [data]
        elif isinstance(data[0][0], (int, float, np.number)):
            if len(data[0]) == 2 and not isinstance(data[1][0], (list, tuple)):
                datasets = [data]
            else:
                datasets = data
        else:
            datasets = [data]
    else:
        datasets = [data]

    color = mpl_opts.pop('color', None)
    colors = [color] if color else plt.rcParams['axes.prop_cycle'].by_key()['color']
    legend_labels = mpl_opts.pop('legend', None)

    for i, dataset in enumerate(datasets):
        if isinstance(dataset[0], (list, tuple)) and len(dataset[0]) == 2:
            x_vals = [p[0] for p in dataset]
            y_vals = [p[1] for p in dataset]
        else:
            x_vals = list(range(1, len(dataset) + 1))
            y_vals = list(dataset)

        plot_color = colors[i % len(colors)]
        label = legend_labels[i] if legend_labels and i < len(legend_labels) else None
        ax.plot(x_vals, y_vals, color=plot_color, marker='o', label=label)

    if 'title' in mpl_opts:
        ax.set_title(mpl_opts['title'])
    if legend_labels:
        ax.legend()

    plt.tight_layout()
    return fig


def ParametricPlot(funcs, t_range, **options):
    """
    ParametricPlot.

    ParametricPlot[{fx, fy}, {t, tmin, tmax}] - parametric curve

    Examples:
        >>> t = Symbol('t')
        >>> fig = ParametricPlot([Cos(t), Sin(t)], (t, 0, 2*Pi))  # Circle
    """
    mpl_opts = _parse_plot_options(options)

    figsize = mpl_opts.pop('figsize', (8, 6))
    fig, ax = plt.subplots(figsize=figsize)

    if isinstance(t_range, (list, tuple)) and len(t_range) == 3:
        t_var, tmin, tmax = t_range
        tmin_val = float(tmin) if hasattr(tmin, 'evalf') else float(tmin)
        tmax_val = float(tmax) if hasattr(tmax, 'evalf') else float(tmax)
    else:
        raise ValueError("t_range must be (t, tmin, tmax)")

    t_vals = np.linspace(tmin_val, tmax_val, 1000)

    # Parse functions
    if len(funcs) == 2:
        fx, fy = funcs
        if hasattr(fx, 'free_symbols'):
            fx_func = sp.lambdify(t_var, fx, modules=['numpy'])
        else:
            fx_func = fx
        if hasattr(fy, 'free_symbols'):
            fy_func = sp.lambdify(t_var, fy, modules=['numpy'])
        else:
            fy_func = fy

        x_vals = fx_func(t_vals)
        y_vals = fy_func(t_vals)

        color = mpl_opts.pop('color', 'blue')
        ax.plot(x_vals, y_vals, color=color)
    else:
        raise ValueError("funcs must be [fx, fy]")

    if 'title' in mpl_opts:
        ax.set_title(mpl_opts['title'])

    ax.set_aspect('equal')
    plt.tight_layout()
    return fig


def Show(*figures, **options):
    """
    Combine multiple plots into one figure.

    Show[fig1, fig2, ...] - overlay multiple plots
    """
    # Create new figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Copy lines from each figure
    for i, src_fig in enumerate(figures):
        src_axes = src_fig.axes
        if src_axes:
            src_ax = src_axes[0]
            for line in src_ax.get_lines():
                ax.plot(line.get_xdata(), line.get_ydata(),
                       color=line.get_color(),
                       linestyle=line.get_linestyle())

    plt.tight_layout()
    return fig


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'Plot', 'ListPlot', 'ListLinePlot', 'ParametricPlot', 'Show',
]
