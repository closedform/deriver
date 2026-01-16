"""Derive Symbolic Mathematics Notebook"""
import marimo

__generated_with = "0.8.0"
app = marimo.App()


@app.cell(hide_code=True)
def __():
    import marimo as mo
    from derive import Symbol, Integrate, D, Sin, Cos, Exp, Pi, Plot
    return mo, Symbol, Integrate, D, Sin, Cos, Exp, Pi, Plot


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""
    # Derive Notebook

    A symbolic mathematics environment for Python.
    """)
    return


@app.cell(hide_code=True)
def __(Symbol, D, Sin, Exp):
    x = Symbol('x')
    # Symbolic differentiation
    D(Sin(x) * Exp(x), x)
    return x,


@app.cell(hide_code=True)
def __(Symbol, Integrate, Exp):
    x = Symbol('x')
    # Integration
    Integrate(x**2 * Exp(-x), x)
    return


@app.cell(hide_code=True)
def __(Symbol, Sin, Cos, Pi, Plot):
    x = Symbol('x')
    # Plotting
    Plot([Sin(x), Cos(x)], (x, 0, 2*Pi))
    return


if __name__ == "__main__":
    app.run()
