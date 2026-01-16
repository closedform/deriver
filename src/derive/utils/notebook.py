"""
notebook.py - Jupyter Notebook Integration

This module provides Jupyter/IPython integration for derive,
including LaTeX rendering and rich display of expressions.
"""

import sympy as sp
from sympy import latex
from typing import Any, Optional
import io


# Check if we're in a Jupyter/IPython environment
def _in_notebook() -> bool:
    """Check if running in a Jupyter notebook."""
    try:
        from IPython import get_ipython
        shell = get_ipython()
        if shell is None:
            return False
        if shell.__class__.__name__ == 'ZMQInteractiveShell':
            return True  # Jupyter notebook or qtconsole
        elif shell.__class__.__name__ == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False
    except (ImportError, NameError):
        return False


def _display_latex(expr):
    """Display expression as LaTeX in Jupyter."""
    try:
        from IPython.display import display, Math, Latex
        latex_str = latex(expr)
        display(Math(latex_str))
    except ImportError:
        print(expr)


def _display_html(html_str: str):
    """Display HTML in Jupyter."""
    try:
        from IPython.display import display, HTML
        display(HTML(html_str))
    except ImportError:
        print(html_str)


class DeriveFormatter:
    """
    Custom formatter for derive objects in Jupyter.

    Automatically renders SymPy expressions as LaTeX.
    """

    @staticmethod
    def _repr_latex_(expr):
        """Return LaTeX representation for Jupyter."""
        if hasattr(expr, '_repr_latex_'):
            return expr._repr_latex_()
        return f"${latex(expr)}$"


def enable_latex_printing():
    """
    Enable automatic LaTeX rendering in Jupyter notebooks.

    Call this at the start of a notebook to have all
    symbolic expressions render as LaTeX.
    """
    try:
        from IPython import get_ipython
        from IPython.display import display, Math

        ip = get_ipython()
        if ip is None:
            return

        # Enable SymPy's LaTeX printing
        sp.init_printing(use_latex='mathjax')

        print("LaTeX printing enabled. Symbolic expressions will render as math.")

    except ImportError:
        pass


def MathForm(expr) -> str:
    """
    Convert expression to LaTeX and display it.

    In Jupyter: renders as formatted math
    In terminal: returns LaTeX string

    Examples:
        >>> x = Symbol('x')
        >>> MathForm(x**2 + 1)  # Displays as LaTeX in notebook
    """
    latex_str = latex(expr)

    if _in_notebook():
        try:
            from IPython.display import display, Math
            display(Math(latex_str))
        except ImportError:
            pass

    return latex_str


def DisplayForm(expr, form: str = "latex"):
    """
    Display expression in specified form.

    Args:
        expr: Expression to display
        form: One of 'latex', 'unicode', 'ascii', 'tree'

    Examples:
        >>> x = Symbol('x')
        >>> DisplayForm(x**2 + 1, 'latex')
        >>> DisplayForm(x**2 + 1, 'unicode')
    """
    if form == 'latex':
        return MathForm(expr)
    elif form == 'unicode':
        return sp.pretty(expr, use_unicode=True)
    elif form == 'ascii':
        return sp.pretty(expr, use_unicode=False)
    elif form == 'tree':
        return sp.srepr(expr)
    else:
        return str(expr)


def TableForm(data, headings=None):
    """
    Display data as a formatted table.

    In Jupyter: renders as HTML table
    In terminal: prints ASCII table

    Args:
        data: 2D list of data
        headings: Optional list of column headings

    Examples:
        >>> TableForm([[1, 2], [3, 4]], headings=['a', 'b'])
    """
    if _in_notebook():
        # Create HTML table
        html = '<table border="1" style="border-collapse: collapse;">'

        if headings:
            html += '<tr>'
            for h in headings:
                html += f'<th style="padding: 8px;">{h}</th>'
            html += '</tr>'

        for row in data:
            html += '<tr>'
            for cell in row:
                # Render as LaTeX if it's a SymPy expression
                if hasattr(cell, 'free_symbols'):
                    cell_str = f'${latex(cell)}$'
                else:
                    cell_str = str(cell)
                html += f'<td style="padding: 8px;">{cell_str}</td>'
            html += '</tr>'

        html += '</table>'
        _display_html(html)
        return html
    else:
        # ASCII table
        lines = []
        if headings:
            lines.append(' | '.join(str(h) for h in headings))
            lines.append('-' * len(lines[0]))

        for row in data:
            lines.append(' | '.join(str(cell) for cell in row))

        result = '\n'.join(lines)
        print(result)
        return result


def setup_notebook():
    """
    Configure the notebook environment for derive.

    This function:
    1. Enables LaTeX printing
    2. Sets up matplotlib inline
    3. Imports common symbols

    Call at the start of a notebook:
        from derive.notebook import setup_notebook
        setup_notebook()
    """
    enable_latex_printing()

    try:
        from IPython import get_ipython
        ip = get_ipython()
        if ip:
            # Enable matplotlib inline
            ip.run_line_magic('matplotlib', 'inline')

            # Import common symbols into user namespace
            ip.user_ns.update({
                'x': sp.Symbol('x'),
                'y': sp.Symbol('y'),
                'z': sp.Symbol('z'),
                't': sp.Symbol('t'),
            })
            print("Common symbols (x, y, z, t) imported.")
            print("Ready for derive!")
    except Exception:
        pass


def create_notebook_template() -> str:
    """
    Create a template Jupyter notebook for derive.

    Returns:
        JSON string of notebook content
    """
    import json

    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["# Derive Notebook\n",
                          "\n",
                          "A symbolic mathematics environment for Python."]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": ["from derive import *\n",
                          "from derive.notebook import setup_notebook\n",
                          "setup_notebook()"],
                "outputs": []
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## Examples"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": ["# Symbolic differentiation\n",
                          "D(Sin(x) * Exp(x), x)"],
                "outputs": []
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": ["# Integration\n",
                          "Integrate(x**2 * Exp(-x), (x, 0, Infinity))"],
                "outputs": []
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": ["# Plotting\n",
                          "Plot([Sin(x), Cos(x)], (x, 0, 2*Pi), PlotLabel='Trig Functions')"],
                "outputs": []
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    return json.dumps(notebook, indent=2)


def save_notebook_template(path: str = "derive_template.ipynb"):
    """Save a template notebook to file."""
    content = create_notebook_template()
    with open(path, 'w') as f:
        f.write(content)
    print(f"Template notebook saved to {path}")
    return path


# =============================================================================
# Marimo Integration
# =============================================================================

def _in_marimo() -> bool:
    """Check if running in a Marimo notebook."""
    try:
        import marimo as mo
        # Check if we're in a marimo runtime
        return mo.running_in_notebook()
    except (ImportError, AttributeError):
        return False


def marimo_latex(expr) -> Any:
    """
    Display expression as LaTeX in Marimo.

    Args:
        expr: Expression to display

    Returns:
        Marimo markdown element with LaTeX

    Examples:
        >>> marimo_latex(x**2 + 1)
    """
    try:
        import marimo as mo
        latex_str = latex(expr)
        return mo.md(f"$${latex_str}$$")
    except ImportError:
        return f"$${latex(expr)}$$"


def marimo_table(data, headings=None) -> Any:
    """
    Display data as a table in Marimo.

    Args:
        data: 2D list of data
        headings: Optional column headings

    Returns:
        Marimo table element
    """
    try:
        import marimo as mo

        # Convert data to list of dicts - marimo requires this format
        if headings:
            rows = [dict(zip(headings, row)) for row in data]
        else:
            # Generate column names if not provided
            if data and hasattr(data[0], '__iter__'):
                col_names = [f'col_{i}' for i in range(len(data[0]))]
                rows = [dict(zip(col_names, row)) for row in data]
            else:
                rows = [{'value': item} for item in data]
        return mo.ui.table(rows)
    except (ImportError, Exception):
        # Fallback to standard TableForm
        return TableForm(data, headings)


def marimo_plot(fig) -> Any:
    """
    Display a matplotlib figure in Marimo.

    Args:
        fig: Matplotlib figure

    Returns:
        Marimo element displaying the plot
    """
    try:
        import marimo as mo
        return mo.as_html(fig)
    except ImportError:
        return fig


def setup_marimo():
    """
    Configure the Marimo environment for derive.

    Call at the start of a Marimo notebook.
    """
    try:
        import marimo as mo
        # Marimo doesn't need explicit setup like Jupyter
        # but we can provide helpful output
        return mo.md("""
**Derive** symbolic mathematics library loaded.

Common symbols: Use `Symbol('x')` to create symbols.

Example:
```python
x = Symbol('x')
Integrate(Sin(x), x)
```
        """)
    except ImportError:
        print("Marimo not available. Use Jupyter setup instead.")


def create_marimo_template() -> str:
    """
    Create a template Marimo notebook for derive.

    Returns:
        Python code string for a Marimo notebook
    """
    template = '''"""Derive Symbolic Mathematics Notebook"""
import marimo

__generated_with = "0.8.0"
app = marimo.App()


@app.cell
def __():
    import marimo as mo
    from derive import *
    return mo, Symbol, Integrate, D, Sin, Cos, Exp, Pi


@app.cell
def __(mo):
    mo.md("""
    # Derive Notebook

    A symbolic mathematics environment for Python.
    """)
    return


@app.cell
def __(Symbol, D, Sin, Exp):
    x = Symbol('x')
    # Symbolic differentiation
    D(Sin(x) * Exp(x), x)
    return x,


@app.cell
def __(Symbol, Integrate, Exp):
    x = Symbol('x')
    # Integration
    Integrate(x**2 * Exp(-x), x)
    return


@app.cell
def __(Symbol, Sin, Cos, Pi):
    from derive import Plot
    x = Symbol('x')
    # Plotting
    Plot([Sin(x), Cos(x)], (x, 0, 2*Pi))
    return


if __name__ == "__main__":
    app.run()
'''
    return template


def save_marimo_template(path: str = "derive_notebook.py"):
    """Save a template Marimo notebook to file."""
    content = create_marimo_template()
    with open(path, 'w') as f:
        f.write(content)
    print(f"Marimo template notebook saved to {path}")
    print(f"Run with: marimo run {path}")
    return path


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Jupyter
    'enable_latex_printing', 'MathForm', 'DisplayForm', 'TableForm',
    'setup_notebook', 'create_notebook_template', 'save_notebook_template',
    # Marimo
    'marimo_latex', 'marimo_table', 'marimo_plot', 'setup_marimo',
    'create_marimo_template', 'save_marimo_template',
]
