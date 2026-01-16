# Derive Documentation

**Derive** is a Python library that provides intuitive symbolic mathematics capabilities for calculus, linear algebra, differential equations, tensor calculus, and more.

## Installation

```bash
uv add derive
# or
pip install derive
```

## Quick Start

```python
from derive import *

# Create symbols
x, y = symbols('x y')

# Calculus
D(Sin(x), x)           # cos(x)
Integrate(x**2, x)     # x**3/3
Limit(Sin(x)/x, x, 0)  # 1

# Plotting
Plot(Sin(x), (x, 0, 2*Pi))
```

## Modules

- [Calculus](calculus.md) - Differentiation, integration, limits
- [Special Functions](special.md) - Bessel, Legendre, etc.
- [Differential Geometry](diffgeo.md) - Metrics, tensors, abstract indices
- [Plotting](plotting.md) - Visualization

## API Reference

### Symbols
- `Symbol(name)` - Create a symbolic variable
- `symbols('x y z')` - Create multiple symbols

### Trigonometric Functions
- `Sin(x)`, `Cos(x)`, `Tan(x)`, `Cot(x)`, `Sec(x)`, `Csc(x)`
- `ArcSin(x)`, `ArcCos(x)`, `ArcTan(x)`, etc.
- `Sinh(x)`, `Cosh(x)`, `Tanh(x)`, etc.

### Exponential/Logarithmic
- `Exp(x)` - Exponential e^x
- `Log(x)` - Natural logarithm
- `Sqrt(x)` - Square root
- `Power(x, n)` - x^n

### Constants
- `Pi` - 3.14159...
- `E` - Euler's number
- `I` - Imaginary unit
- `Infinity` - Positive infinity

### Calculus
- `D(expr, x)` - Differentiate
- `Integrate(expr, x)` - Integrate
- `Limit(expr, x, a)` - Compute limit
- `Series(expr, (x, x0, n))` - Taylor series
- `Sum(expr, (i, a, b))` - Summation
- `Product(expr, (i, a, b))` - Product

### Differential Equations
- `DSolve(eq, y(x), x)` - Symbolic ODE solver
- `NDSolve(f, y0, (t, t0, tf))` - Numerical ODE solver

### Linear Algebra
- `Matrix(...)` - Create matrix
- `Dot(A, B)` - Matrix multiplication
- `Det(M)` - Determinant
- `Inverse(M)` - Matrix inverse
- `Eigenvalues(M)` - Eigenvalues
- `Eigenvectors(M)` - Eigenvectors

### Special Functions
- `BesselJ(n, x)`, `BesselY(n, x)` - Bessel functions
- `LegendreP(n, x)` - Legendre polynomial
- `SphericalHarmonicY(l, m, theta, phi)` - Spherical harmonics
- `Gamma(x)`, `Beta(a, b)` - Gamma and Beta functions
- `Erf(x)`, `Erfc(x)` - Error functions
- `Zeta(s)` - Riemann zeta function

### Plotting
- `Plot(f, (x, a, b))` - 2D function plot
- `ListPlot(data)` - Plot data points
- `ParametricPlot(...)` - Parametric curves

### Probability
- `NormalDistribution(mu, sigma)` - Normal distribution
- `PDF(dist, x)` - Probability density
- `CDF(dist, x)` - Cumulative distribution
- `Mean(dist)`, `Variance(dist)` - Statistics

### Pattern Matching
- `Replace(expr, rule)` - Single replacement
- `ReplaceAll(expr, rules)` - Replace all occurrences
- `ReplaceRepeated(expr, rules)` - Repeatedly apply rules
- `Rule(pattern, replacement)` - Create a replacement rule
- `DefineFunction(name, func)` - Define custom function

### Custom Types
- `DefineType(name, fields)` - Create custom type
- `Quaternion(w, x, y, z)` - Quaternion numbers
- `Vector3D(x, y, z)` - 3D vectors

### Differential Geometry
- `Metric(coords, components)` - Define metric tensor
- `IndexType(name, dim)` - Define index type
- `AbstractTensor(name, rank, index_type)` - Abstract tensor
- `T[a, -b]` - Index notation (positive=up, negative=down)

## Examples

See the [examples notebook](../examples/showcase.ipynb) for interactive demonstrations.

## License

MIT License
