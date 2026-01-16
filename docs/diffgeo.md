# Differential Geometry

Derive provides tools for differential geometry, tensor calculus, and abstract index notation.

## Metrics

### Creating a Metric

```python
from derive import *
from derive.diffgeo import Metric, minkowski_metric, schwarzschild_metric

# Define a 2-sphere metric
theta, phi = symbols('theta phi', real=True)
sphere = Metric(
    coords=[theta, phi],
    components=[
        [1, 0],
        [0, Sin(theta)**2]
    ]
)

# Or use predefined metrics
minkowski = minkowski_metric()      # Flat spacetime
schwarzschild = schwarzschild_metric()  # Black hole
```

## Christoffel Symbols

Christoffel symbols are computed with symmetry optimization (only n×n(n+1)/2 unique components).

```python
# First kind: Γ_ijk
gamma_1st = sphere.christoffel_first_kind()

# Second kind: Γ^i_jk
gamma_2nd = sphere.christoffel_second_kind()

# Access components
gamma_2nd[0, 1, 1]  # Γ^θ_φφ
```

## Curvature Tensors

### Riemann Tensor

```python
riemann = sphere.riemann_tensor()
# R^ρ_σμν
```

### Ricci Tensor and Scalar

```python
ricci = sphere.ricci_tensor()
R = sphere.ricci_scalar()
```

### Einstein Tensor

```python
G = sphere.einstein_tensor()
# G_μν = R_μν - (1/2)g_μν R
```

## Abstract Index Notation

xAct-style abstract indices with sign convention for up/down.

```python
from derive.diffgeo import IndexType, AbstractTensor, minkowski_metric

# Create an index type with associated metric
g = minkowski_metric(4)
spacetime = IndexType('spacetime', 4, metric=g)

# Create abstract indices
a, b, c = spacetime.indices('a b c')

# Define tensors with symmetry properties
T = AbstractTensor('T', rank=2, index_type=spacetime)
metric = AbstractTensor('g', rank=2, index_type=spacetime, symmetric=[(0, 1)])
field = AbstractTensor('F', rank=2, index_type=spacetime, antisymmetric=[(0, 1)])

# Use sign convention: positive = up, negative = down
T[a, -b]    # T^a_b (mixed tensor)
T[-a, -b]   # T_ab (covariant)
T[a, b]     # T^ab (contravariant)
```

### Raising and Lowering Indices

```python
from derive.diffgeo import raise_index, lower_index

expr = T[-a, -b]       # T_ab
raised = raise_index(expr, 0)  # T^a_b (raise first index)
```

## Coordinate Transformations

```python
from derive import *
from derive.diffgeo import CoordinateTransformation, cartesian_to_spherical_3d

# Use predefined transformation
trans = cartesian_to_spherical_3d()

# Or define custom transformation
x, y, z = symbols('x y z')
r, theta, phi = symbols('r theta phi')

transform = CoordinateTransformation(
    old_coords=[x, y, z],
    new_coords=[r, theta, phi],
    transformations={
        x: r*Sin(theta)*Cos(phi),
        y: r*Sin(theta)*Sin(phi),
        z: r*Cos(theta)
    }
)

# Compute Jacobian
J = transform.jacobian
det_J = transform.jacobian_determinant  # r² sin(θ)
```

## Tensors with Components

```python
from derive import *
from derive.diffgeo import Tensor

# Create a contravariant vector (upper index)
v = Tensor('v', Array([1, 2, 3]), [+1])
v.is_upper(0)  # True

# Create a covariant vector (lower index)
w = Tensor('w', Array([1, 2, 3]), [-1])
w.is_lower(0)  # True

# Mixed tensor T^μ_ν
components = MutableDenseNDimArray.zeros(3, 3)
T = Tensor('T', components, [+1, -1])
```

## Covariant Derivatives

```python
from derive import *
from derive.diffgeo import CovariantDerivative, Tensor, spherical_metric_3d

g = spherical_metric_3d()
v = Tensor('v', Array([1, 0, 0]), [+1])

# Covariant derivative ∇_μ v^ν
nabla_v = CovariantDerivative(v, 0, g)
```

## Variational Derivatives

Compute Euler-Lagrange equations from Lagrangians.

```python
from derive import *
from derive.calculus import VariationalDerivative, EulerLagrangeEquation

# Klein-Gordon field
x, t = symbols('x t')
m = symbols('m', positive=True)
phi = Function('phi')(x, t)

# Lagrangian: L = (1/2)(∂φ/∂t)² - (1/2)(∂φ/∂x)² - (1/2)m²φ²
L = Rational(1,2)*D(phi,t)**2 - Rational(1,2)*D(phi,x)**2 - Rational(1,2)*m**2*phi**2

# Get equation of motion
eq = VariationalDerivative(L, phi, [x, t])
# Result: ∂²φ/∂x² - ∂²φ/∂t² - m²φ = 0 (Klein-Gordon equation)
```

## Predefined Metrics

| Function | Description |
|----------|-------------|
| `minkowski_metric(dim)` | Flat spacetime (-,+,+,+) |
| `schwarzschild_metric()` | Schwarzschild black hole |
| `flrw_metric()` | FLRW cosmology |
| `spherical_metric_3d()` | 3D Euclidean in spherical coords |

## Predefined Transformations

| Function | Description |
|----------|-------------|
| `cartesian_to_spherical_3d()` | (x,y,z) → (r,θ,φ) |
| `cartesian_to_cylindrical()` | (x,y,z) → (ρ,φ,z) |

## Symmetry Utilities

```python
from derive.diffgeo import (
    symmetric_index_pairs,
    symmetric_christoffel_indices,
    SymmetricMatrix,
    SymmetricChristoffel
)

# Generate unique index pairs for symmetric tensor
pairs = symmetric_index_pairs(3)  # [(0,0), (0,1), (0,2), (1,1), (1,2), (2,2)]

# Efficient symmetric matrix storage
m = SymmetricMatrix(3)
m[0, 1] = 5
m[1, 0]  # Returns 5 (symmetric access)

# Christoffel symbol storage with symmetry
sc = SymmetricChristoffel(4)
sc.num_unique   # 40 (vs 64 total)
sc.savings_ratio  # 0.375 (37.5% reduction)
```

### Levi-Civita Symbol

```python
from derive.diffgeo import LeviCivita, levi_civita_tensor

# Get Levi-Civita symbol value
LeviCivita(0, 1, 2)     # +1 (even permutation)
LeviCivita(1, 0, 2)     # -1 (odd permutation)
LeviCivita(0, 0, 1)     # 0 (repeated index)

# Create full tensor
eps = levi_civita_tensor(3)  # 3D antisymmetric tensor
```

### Tensor Filling with Symmetries

```python
from derive.diffgeo import (
    fill_symmetric_tensor,
    fill_antisymmetric_tensor,
    fill_with_symmetries
)

# Fill symmetric tensor from unique components
T = fill_symmetric_tensor({(0,0): 1, (0,1): 2, (1,1): 3}, 2)

# Fill antisymmetric tensor
A = fill_antisymmetric_tensor({(0,1): 5}, 2)  # A[0,1]=5, A[1,0]=-5

# General symmetry filling
T = fill_with_symmetries(components, dim, symmetric=[(0,1)], antisymmetric=[(2,3)])
```

## Einstein Summation

### Einsum - Einstein Notation

Evaluate expressions using Einstein summation convention.

```python
from derive import *
from derive.diffgeo import Einsum

A = Array([[1, 2], [3, 4]])
B = Array([[5, 6], [7, 8]])

# Matrix multiplication: C_ik = A_ij B_jk
C = Einsum("ij,jk->ik", A, B)

# Trace: tr(A) = A_ii
trace = Einsum("ii->", A)

# Transpose: B_ji = A_ij
B = Einsum("ij->ji", A)

# Outer product
outer = Einsum("i,j->ij", Array([1,2]), Array([3,4]))
```

### Contract - Tensor Contraction

```python
from derive import *
from derive.diffgeo import Contract

T = Array([[[1,2],[3,4]], [[5,6],[7,8]]])

# Contract first two indices
result = Contract(T, (0, 1))
```

### Tensor Operations

```python
from derive import *
from derive.diffgeo import TensorProd, OuterProduct, InnerProduct, Trace

a = Array([1, 2])
b = Array([3, 4])

# Outer product
outer = OuterProduct(a, b)  # [[3,4], [6,8]]

# Inner product (full contraction)
inner = InnerProduct(a, b)  # 1*3 + 2*4 = 11

# Tensor product of multiple arrays
T = TensorProd(a, b)

# Trace of matrix
A = Array([[1, 2], [3, 4]])
tr = Trace(A)  # 1 + 4 = 5
```
