"""
display.py - Output and Display Functions.

Provides functions for output formatting including rich terminal output,
LaTeX conversion, and pretty printing.
"""

from typing import Any, Optional
from sympy import latex, pretty, Symbol
from sympy.printing.pretty.pretty import PrettyPrinter


def TeXForm(expr: Any) -> str:
    """
    Convert expression to LaTeX string.

    Args:
        expr: Expression to convert

    Returns:
        LaTeX representation

    Examples:
        >>> x = Symbol('x')
        >>> TeXForm(x**2 + 1)
        'x^{2} + 1'
    """
    return latex(expr)


def PrettyForm(expr: Any, use_unicode: bool = True) -> str:
    """
    Convert expression to pretty-printed string.

    Uses SymPy's pretty printer for ASCII/Unicode art rendering
    of mathematical expressions.

    Args:
        expr: Expression to convert
        use_unicode: Use Unicode symbols (default True)

    Returns:
        Pretty-printed string representation

    Examples:
        >>> x = Symbol('x')
        >>> print(PrettyForm(x**2 + 1))
         2
        x  + 1
    """
    return pretty(expr, use_unicode=use_unicode)


def Print(*args: Any, pretty: bool = False, latex_mode: bool = False) -> None:
    """
    Print expressions with optional formatting.

    Args:
        *args: Expressions to print
        pretty: Use pretty printing (default False)
        latex_mode: Print LaTeX (default False)
    """
    for arg in args:
        if latex_mode:
            print(TeXForm(arg))
        elif pretty:
            print(PrettyForm(arg))
        else:
            print(arg)


def RichPrint(*args: Any, style: Optional[str] = None) -> None:
    """
    Print expressions using rich library for enhanced terminal output.

    Args:
        *args: Expressions to print
        style: Optional rich style (e.g., "bold", "green", "bold blue")

    Examples:
        >>> RichPrint(expr, style="bold green")
    """
    try:
        from rich.console import Console
        from rich.text import Text

        console = Console()
        for arg in args:
            text = str(arg)
            if style:
                console.print(text, style=style)
            else:
                console.print(text)
    except ImportError:
        # Fallback to regular print
        for arg in args:
            print(arg)


def RichLatex(expr: Any) -> None:
    """
    Print LaTeX form with rich formatting.

    Args:
        expr: Expression to print
    """
    try:
        from rich.console import Console
        from rich.panel import Panel

        console = Console()
        latex_str = TeXForm(expr)
        console.print(Panel(latex_str, title="LaTeX", border_style="blue"))
    except ImportError:
        print(f"LaTeX: {TeXForm(expr)}")


def TableForm(data: Any, headers: Optional[list] = None) -> None:
    """
    Print data in a formatted table using rich.

    Args:
        data: 2D list or matrix to display
        headers: Optional column headers
    """
    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table()

        # Add headers
        if headers:
            for h in headers:
                table.add_column(str(h))
        else:
            # Generate column numbers
            if hasattr(data, '__iter__') and len(data) > 0:
                first_row = data[0] if hasattr(data[0], '__iter__') else [data[0]]
                for i in range(len(first_row)):
                    table.add_column(f"Col {i+1}")

        # Add rows
        for row in data:
            if hasattr(row, '__iter__'):
                table.add_row(*[str(x) for x in row])
            else:
                table.add_row(str(row))

        console.print(table)
    except ImportError:
        # Fallback
        if headers:
            print('\t'.join(str(h) for h in headers))
        for row in data:
            if hasattr(row, '__iter__'):
                print('\t'.join(str(x) for x in row))
            else:
                print(row)


def show(expr: Any, mode: str = "auto") -> None:
    """
    Display an expression in the best available format.

    Automatically detects environment (terminal, Jupyter, etc.)
    and chooses appropriate display method.

    Args:
        expr: Expression to display
        mode: Display mode - "auto", "pretty", "latex", "plain"

    Examples:
        >>> show(x**2 + 1)  # Auto-detects best format
    """
    if mode == "latex":
        print(TeXForm(expr))
    elif mode == "pretty":
        print(PrettyForm(expr))
    elif mode == "plain":
        print(expr)
    else:  # auto
        # Try to detect environment
        try:
            # Check if in IPython/Jupyter
            from IPython import get_ipython
            ipython = get_ipython()
            if ipython is not None:
                from IPython.display import display, Math
                display(Math(TeXForm(expr)))
                return
        except (ImportError, NameError):
            pass

        # Terminal output with pretty printing
        print(PrettyForm(expr))


__all__ = [
    'TeXForm',
    'PrettyForm',
    'Print',
    'RichPrint',
    'RichLatex',
    'TableForm',
    'show',
]
