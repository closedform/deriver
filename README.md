# Derive

A powerful symbolic mathematics library for Python.

## Features

- **Symbolic Computation**: Work with mathematical expressions symbolically
- **Calculus**: Differentiation, integration, limits, series, variational derivatives
- **Linear Algebra**: Matrix operations, eigenvalues, determinants
- **Differential Equations**: Symbolic and numerical ODE solving
- **Differential Geometry**: Metrics, Christoffel symbols, curvature tensors, abstract indices
- **Special Functions**: Bessel, Gamma, error functions, and more
- **Pattern Matching**: Replace, ReplaceAll, custom function definitions
- **Custom Types**: Define your own types (Quaternion, Vector3D included)
- **Probability**: Distribution functions and statistics
- **Optimization**: Convex optimization via cvxpy (optional)
- **Plotting**: Publication-quality mathematical plots
- **Composable API**: Pipe operations and functional composition

## Installation

To install, clone the repository and sync the dependencies:

```bash
git clone git@github.com:closedform/deriver.git
cd deriver
uv sync
```

## Quick Start

```python
from derive import *

# Define symbols
x, y = symbols('x y')

# Calculus
D(Sin(x), x)                    # Differentiation: Cos(x)
Integrate(x**2, x)              # Integration: x**3/3
Limit(Sin(x)/x, x, 0)           # Limits: 1
Series(Exp(x), (x, 0, 5))        # Taylor series

# Algebra
Solve(Eq(x**2 - 4, 0), x)       # Solve equations: [-2, 2]
Factor(x**2 - 5*x + 6)          # Factor: (x-2)(x-3)
Simplify(Sin(x)**2 + Cos(x)**2) # Simplify: 1

# Linear Algebra
A = Matrix([[1, 2], [3, 4]])
Det(A)                          # Determinant: -2
Eigenvalues(A)                  # Eigenvalues
Inverse(A)                      # Matrix inverse

# Differential Equations
t = Symbol('t')
y = Function('y')
DSolve(Eq(y(t).diff(t), y(t)), y(t), t)  # Solve dy/dt = y

# Plotting
Plot(Sin(x), (x, 0, 2*Pi), PlotLabel="Sine Wave")
```

## Differential Geometry

```python
from derive.diffgeo import *

# Create a metric (2-sphere)
theta, phi = symbols('theta phi', real=True)
sphere = Metric(
    coords=[theta, phi],
    components=[
        [1, 0],
        [0, Sin(theta)**2]
    ]
)

# Compute geometric quantities
christoffel = sphere.christoffel_second_kind()
riemann = sphere.riemann_tensor()
ricci = sphere.ricci_tensor()
R = sphere.ricci_scalar()

# Abstract index notation (xAct-style)
spacetime = IndexType('spacetime', 4, metric=minkowski_metric())
a, b, c = spacetime.indices('a b c')

T = AbstractTensor('T', rank=2, index_type=spacetime)
g = AbstractTensor('g', rank=2, index_type=spacetime, symmetric=[(0, 1)])
F = AbstractTensor('F', rank=2, index_type=spacetime, antisymmetric=[(0, 1)])

# Use sign convention: a = upper, -a = lower
T[a, -b]   # T^a_b
T[-a, -b]  # T_ab
```

## Pattern Matching

```python
from derive import *

# Pattern-based replacement
Replace(x**2 + y**2, Rule(x**2, a))  # a + y**2

# Replace all occurrences
ReplaceAll(x + x**2 + x**3, Rule(x, 2))  # 2 + 4 + 8

# Define custom functions
x_ = Pattern_('x')
f = DefineFunction('f')
f.define(x_, x_**2 + 1)
f(3)  # 10
```

## Custom Types

```python
from derive.types import Quaternion, Vector3D

# Quaternions
q1 = Quaternion(1, 2, 3, 4)
q2 = Quaternion(5, 6, 7, 8)
q1 * q2  # Quaternion multiplication

# 3D Vectors
v1 = Vector3D(1, 2, 3)
v2 = Vector3D(4, 5, 6)
v1.cross(v2)  # Cross product
v1.dot(v2)    # Dot product
```

## Composable API

```python
from derive import *

# Chain operations with Pipe
result = (
    Pipe((x + 1)**3)
    .then(Expand)
    .then(Simplify)
    .value
)

# Or use the pipe function
result = pipe(Sin(x)**2 + Cos(x)**2, Simplify)  # Returns 1

# Repeated function application
Nest(lambda x: x**2, 2, 3)           # 256 (2 -> 4 -> 16 -> 256)
NestList(lambda x: 2*x, 1, 4)        # [1, 2, 4, 8, 16]

# Find fixed points
FixedPoint(lambda x: (x + 2/x)/2, 1.0)  # sqrt(2)
FixedPointList(lambda x: Cos(x), 1.0)   # Convergence path
```

## Variational Calculus

```python
from derive import *
from derive.calculus import VariationalDerivative

# Klein-Gordon Lagrangian
x, t, m = symbols('x t m')
phi = Function('phi')(x, t)

L = Rational(1,2)*D(phi,t)**2 - Rational(1,2)*D(phi,x)**2 - Rational(1,2)*m**2*phi**2

# Compute Euler-Lagrange equation
eq = VariationalDerivative(L, phi, [x, t])
# Result: -m²φ - ∂²φ/∂t² + ∂²φ/∂x² = 0
```

## Documentation

See the `docs/` directory for detailed documentation on:
- [Calculus](docs/calculus.md) - Differentiation, integration, limits, series, transforms
- [Special Functions](docs/special.md) - Bessel, Legendre, Hermite, elliptic integrals
- [Differential Geometry](docs/diffgeo.md) - Metrics, Christoffel, curvature, tensors
- [Plotting](docs/plotting.md) - Function plots, data visualization

## Example Notebooks

The `examples/` directory contains interactive Marimo notebooks demonstrating complex use cases:

| Notebook | Description | Rendered | GitHub |
|----------|-------------|----------|--------|
| `derive_marimo.py` | Intro notebook for core calculus/plotting | [HTML](examples/rendered/derive_marimo.html) | [IPYNB](examples/rendered/derive_marimo.ipynb) |
| `linearized_gravity.py` | Metric perturbations and gravitational waves | [HTML](examples/rendered/linearized_gravity.html) | [IPYNB](examples/rendered/linearized_gravity.ipynb) |
| `quantum_mechanics.py` | Harmonic oscillator, hydrogen atom, perturbation theory | [HTML](examples/rendered/quantum_mechanics.html) | [IPYNB](examples/rendered/quantum_mechanics.ipynb) |
| `classical_mechanics.py` | Lagrangian/Hamiltonian mechanics, Noether's theorem | [HTML](examples/rendered/classical_mechanics.html) | [IPYNB](examples/rendered/classical_mechanics.ipynb) |
| `electromagnetism.py` | Maxwell equations, gauge theory, EM waves | [HTML](examples/rendered/electromagnetism.html) | [IPYNB](examples/rendered/electromagnetism.ipynb) |
| `differential_geometry.py` | Manifolds, curvature tensors, connections | [HTML](examples/rendered/differential_geometry.html) | [IPYNB](examples/rendered/differential_geometry.ipynb) |
| `renormalization_group.py` | CLT as RG fixed point, universality, beta functions | [HTML](examples/rendered/renormalization_group.html) | [IPYNB](examples/rendered/renormalization_group.ipynb) |

The RG notebook demonstrates how the Central Limit Theorem emerges as a renormalization group fixed point, inspired by [The Simplest Renormalization Group](https://dinunno.substack.com/p/the-simplest-renormalization-group).

### Interactive Notebooks

The example notebooks are written for [Marimo](https://marimo.io/), a reactive Python notebook. Marimo is included as a dependency.

To launch an interactive session for any of the examples:

```bash
uv run marimo edit examples/quantum_mechanics.py
```

Replace `examples/quantum_mechanics.py` with the path to any other notebook in the `examples/` directory.

Marimo notebooks are pure Python files that can also be imported as modules or run as scripts.

## Running Tests

```bash
uv run pytest tests/
```

## License

MIT License

## Acknowledgements

Thanks to the open-source libraries Derive is built on:

- [SymPy](https://www.sympy.org/)
- [NumPy](https://numpy.org/)
- [SciPy](https://scipy.org/)
- [Matplotlib](https://matplotlib.org/)
- [Polars](https://www.pola.rs/)
- [mpmath](https://mpmath.org/)
- [Rich](https://rich.readthedocs.io/)
- [Marimo](https://marimo.io/)
- [CVXPY](https://www.cvxpy.org/) (optional, for optimization features)

**Note**: This project is the result of a collaboration between Brandon DiNunno, Tom Mainiero, and Claude Code. Any likeness to proprietary APIs is strictly coincidental.
