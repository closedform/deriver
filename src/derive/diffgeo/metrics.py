"""
metrics.py - Metric Tensors and Related Operations

Provides Metric and Tensor classes along with predefined metrics
like Minkowski, Schwarzschild, FLRW, and spherical coordinates.

Exploits tensor symmetries for optimized computation:
- Christoffel symbols: Γ^ρ_μν = Γ^ρ_νμ (symmetric in lower indices)
- Metric: g_μν = g_νμ (symmetric)
"""

import sympy as sp
from sympy import (
    Symbol, symbols, Function, Matrix, Array,
    sin, cos, sqrt, Rational,
    tensorproduct, tensorcontraction,
    MutableDenseNDimArray, ImmutableDenseNDimArray,
    derive_by_array,
)
from typing import List, Tuple, Optional
from itertools import product as iterprod, combinations

# Use derive's own APIs for self-consistency
from derive.calculus import D
from derive.algebra import Simplify

# Symmetry utilities
from derive.diffgeo.symmetry import (
    symmetric_christoffel_indices,
    fill_antisymmetric_tensor,
)


class Metric:
    """
    Represents a metric tensor for a Riemannian/pseudo-Riemannian manifold.

    Similar to xAct's DefMetric functionality.

    Examples:
        >>> # Schwarzschild metric
        >>> r, theta, phi, t, M = symbols('r theta phi t M')
        >>> coords = [t, r, theta, phi]
        >>> g = Metric(coords, [
        ...     [-(1-2*M/r), 0, 0, 0],
        ...     [0, 1/(1-2*M/r), 0, 0],
        ...     [0, 0, r**2, 0],
        ...     [0, 0, 0, r**2*sin(theta)**2]
        ... ])
    """

    def __init__(self, coords: List[Symbol], components: List[List]):
        """
        Initialize a metric tensor.

        Args:
            coords: List of coordinate symbols
            components: Matrix of metric components g_{μν}
        """
        self.coords = list(coords)
        self.dim = len(coords)
        self.g = Matrix(components)

        if self.g.shape != (self.dim, self.dim):
            raise ValueError(f"Metric must be {self.dim}x{self.dim} for {self.dim} coordinates")

        # Compute inverse metric
        self._g_inv = None
        self._christoffel_1st = None
        self._christoffel_2nd = None
        self._riemann = None
        self._ricci = None
        self._ricci_scalar = None

    @property
    def inverse(self) -> Matrix:
        """Get the inverse metric g^{μν}."""
        if self._g_inv is None:
            self._g_inv = self.g.inv()
        return self._g_inv

    def __getitem__(self, indices: Tuple[int, int]):
        """Get metric component g_{μν}."""
        return self.g[indices]

    def lower(self, i: int, j: int) -> sp.Expr:
        """Get covariant (lower) metric component g_{ij}."""
        return self.g[i, j]

    def upper(self, i: int, j: int) -> sp.Expr:
        """Get contravariant (upper) metric component g^{ij}."""
        return self.inverse[i, j]

    def christoffel_first_kind(self) -> Array:
        """
        Compute Christoffel symbols of the first kind: Γ_{ijk}.

        Γ_{ijk} = (1/2)(∂_j g_{ki} + ∂_k g_{ij} - ∂_i g_{jk})

        Uses vectorized derive_by_array to compute all partial derivatives at once.
        Exploits symmetry Γ_{ijk} = Γ_{ikj} by only computing j <= k.
        """
        if self._christoffel_1st is not None:
            return self._christoffel_1st

        n = self.dim

        # Vectorized: compute all ∂g_ab/∂x^c at once
        # dg[c, a, b] = ∂g_ab/∂x^c
        dg = derive_by_array(self.g, self.coords)

        gamma = MutableDenseNDimArray.zeros(n, n, n)

        # Only compute unique components (i, j, k) with j <= k
        for i, j, k in symmetric_christoffel_indices(n):
            # Γ_{ijk} = (1/2)(∂_j g_{ki} + ∂_k g_{ij} - ∂_i g_{jk})
            # dg[j, k, i] = ∂g_ki/∂x^j, etc.
            val = Rational(1, 2) * (dg[j, k, i] + dg[k, i, j] - dg[i, j, k])
            simplified = Simplify(val)
            gamma[i, j, k] = simplified
            if j != k:
                gamma[i, k, j] = simplified  # Symmetric

        self._christoffel_1st = ImmutableDenseNDimArray(gamma)
        return self._christoffel_1st

    def christoffel_second_kind(self) -> Array:
        """
        Compute Christoffel symbols of the second kind: Γ^i_{jk}.

        Γ^i_{jk} = g^{il} Γ_{ljk}

        Note: Γ^i_{jk} is symmetric in j,k: Γ^i_{jk} = Γ^i_{kj}
        We exploit this symmetry by only computing j <= k.
        """
        if self._christoffel_2nd is not None:
            return self._christoffel_2nd

        n = self.dim
        gamma_1 = self.christoffel_first_kind()
        g_inv = self.inverse
        gamma = MutableDenseNDimArray.zeros(n, n, n)

        # Only compute unique components (i, j, k) with j <= k
        for i, j, k in symmetric_christoffel_indices(n):
            val = sum(g_inv[i, l] * gamma_1[l, j, k] for l in range(n))
            simplified = Simplify(val)
            gamma[i, j, k] = simplified
            if j != k:
                gamma[i, k, j] = simplified  # Symmetric

        self._christoffel_2nd = ImmutableDenseNDimArray(gamma)
        return self._christoffel_2nd

    def riemann_tensor(self) -> Array:
        """
        Compute Riemann curvature tensor: R^ρ_{σμν}.

        R^ρ_{σμν} = ∂_μ Γ^ρ_{νσ} - ∂_ν Γ^ρ_{μσ} + Γ^ρ_{μλ}Γ^λ_{νσ} - Γ^ρ_{νλ}Γ^λ_{μσ}

        Uses vectorized derive_by_array for partial derivatives.
        Exploits antisymmetry R^ρ_{σμν} = -R^ρ_{σνμ} to compute ~half the components.
        """
        if self._riemann is not None:
            return self._riemann

        n = self.dim
        gamma = self.christoffel_second_kind()

        # Vectorized: compute all ∂Γ^ρ_{νσ}/∂x^μ at once
        dgamma = derive_by_array(gamma, self.coords)

        R = MutableDenseNDimArray.zeros(n, n, n, n)

        def riemann_component(rho, sigma, mu, nu):
            """Compute single Riemann component."""
            val = dgamma[mu, rho, nu, sigma] - dgamma[nu, rho, mu, sigma]
            val += sum(
                gamma[rho, mu, lam] * gamma[lam, nu, sigma] -
                gamma[rho, nu, lam] * gamma[lam, mu, sigma]
                for lam in range(n)
            )
            return Simplify(val)

        # Use symmetry system: R^ρ_{σμν} is antisymmetric in (μ,ν) at positions (2,3)
        fill_antisymmetric_tensor(R, (2, 3), riemann_component)

        self._riemann = ImmutableDenseNDimArray(R)
        return self._riemann

    def ricci_tensor(self) -> Matrix:
        """
        Compute Ricci tensor: R_{μν} = R^ρ_{μρν}.
        """
        if self._ricci is not None:
            return self._ricci

        n = self.dim
        R = self.riemann_tensor()
        Ric = Matrix.zeros(n, n)

        for mu in range(n):
            for nu in range(n):
                val = sum(R[rho, mu, rho, nu] for rho in range(n))
                Ric[mu, nu] = Simplify(val)

        self._ricci = Ric
        return self._ricci

    def ricci_scalar(self) -> sp.Expr:
        """
        Compute Ricci scalar: R = g^{μν} R_{μν}.
        """
        if self._ricci_scalar is not None:
            return self._ricci_scalar

        Ric = self.ricci_tensor()
        g_inv = self.inverse
        n = self.dim

        val = sum(g_inv[mu, nu] * Ric[mu, nu]
                  for mu in range(n) for nu in range(n))

        self._ricci_scalar = Simplify(val)
        return self._ricci_scalar

    def einstein_tensor(self) -> Matrix:
        """
        Compute Einstein tensor: G_{μν} = R_{μν} - (1/2)g_{μν}R.
        """
        Ric = self.ricci_tensor()
        R = self.ricci_scalar()
        n = self.dim

        G = Matrix.zeros(n, n)
        for mu in range(n):
            for nu in range(n):
                G[mu, nu] = Simplify(Ric[mu, nu] - Rational(1, 2) * self.g[mu, nu] * R)

        return G


class Tensor:
    """
    Represents a general tensor with specified indices and symmetries.

    Index positions use signed integers:
    - +1 (or any positive) = upper/contravariant index
    - -1 (or any negative) = lower/covariant index

    Symmetries specify which index pairs are symmetric or antisymmetric:
    - symmetric=[(0, 1)] means indices 0 and 1 are symmetric
    - antisymmetric=[(0, 1)] means indices 0 and 1 are antisymmetric

    Examples:
        >>> v = Tensor('v', [1, 2, 3], [+1])        # Contravariant vector v^μ
        >>> w = Tensor('w', [1, 2, 3], [-1])        # Covariant vector w_μ
        >>> T = Tensor('T', [[1,2],[3,4]], [+1,-1]) # Mixed tensor T^μ_ν
        >>> g = Tensor('g', components, [-1,-1], symmetric=[(0,1)])  # Metric tensor
        >>> F = Tensor('F', components, [-1,-1], antisymmetric=[(0,1)])  # Field tensor
    """

    def __init__(
        self,
        name: str,
        components: Array,
        index_positions: List[int],
        metric: Optional[Metric] = None,
        symmetric: Optional[List[Tuple[int, int]]] = None,
        antisymmetric: Optional[List[Tuple[int, int]]] = None,
    ):
        """
        Initialize a tensor.

        Args:
            name: Tensor name/symbol
            components: Array of tensor components
            index_positions: List of +1/-1 for upper/lower indices
            metric: Optional metric for raising/lowering indices
            symmetric: List of index pairs that are symmetric
            antisymmetric: List of index pairs that are antisymmetric
        """
        self.name = name
        self.components = components if isinstance(components, (Array, ImmutableDenseNDimArray)) else ImmutableDenseNDimArray(components)
        self.index_positions = list(index_positions)
        self.rank = len(index_positions)
        self.metric = metric
        self.symmetric_pairs = symmetric or []
        self.antisymmetric_pairs = antisymmetric or []

    def is_upper(self, idx: int) -> bool:
        """Check if index at position idx is upper (contravariant)."""
        return self.index_positions[idx] > 0

    def is_lower(self, idx: int) -> bool:
        """Check if index at position idx is lower (covariant)."""
        return self.index_positions[idx] < 0

    def is_symmetric(self, idx1: int, idx2: int) -> bool:
        """Check if indices idx1 and idx2 are symmetric."""
        pair = (min(idx1, idx2), max(idx1, idx2))
        return pair in self.symmetric_pairs or (idx2, idx1) in self.symmetric_pairs

    def is_antisymmetric(self, idx1: int, idx2: int) -> bool:
        """Check if indices idx1 and idx2 are antisymmetric."""
        pair = (min(idx1, idx2), max(idx1, idx2))
        return pair in self.antisymmetric_pairs or (idx2, idx1) in self.antisymmetric_pairs

    @property
    def num_independent_components(self) -> int:
        """
        Estimate number of independent components based on symmetries.

        For a fully symmetric rank-2 tensor: n(n+1)/2
        For a fully antisymmetric rank-2 tensor: n(n-1)/2
        """
        if self.rank != 2:
            # For now, only handle rank-2
            return None
        if self.components.shape[0] != self.components.shape[1]:
            return None

        n = self.components.shape[0]
        if self.is_symmetric(0, 1):
            return n * (n + 1) // 2
        elif self.is_antisymmetric(0, 1):
            return n * (n - 1) // 2
        return n * n

    def __getitem__(self, indices):
        """Get tensor component."""
        return self.components[indices]

    def contract(self, idx1: int, idx2: int) -> 'Tensor':
        """
        Contract tensor over two indices.

        Args:
            idx1, idx2: Indices to contract (0-indexed)

        Returns:
            New tensor with contracted indices removed.
        """
        # Contract using SymPy's tensorcontraction
        result = tensorcontraction(self.components, (idx1, idx2))
        new_positions = self.index_positions[:idx1] + self.index_positions[idx1+1:idx2] + self.index_positions[idx2+1:]
        return Tensor(f"contracted({self.name})", result, new_positions, self.metric)


def CovariantDerivative(tensor: Tensor, coord_idx: int, metric: Metric) -> Tensor:
    """
    Compute covariant derivative of a tensor.

    ∇_μ T^ν = ∂_μ T^ν + Γ^ν_{μλ} T^λ  (for contravariant vector, index > 0)
    ∇_μ T_ν = ∂_μ T_ν - Γ^λ_{μν} T_λ  (for covariant vector, index < 0)
    """
    gamma = metric.christoffel_second_kind()
    n = metric.dim
    coord = metric.coords[coord_idx]

    if tensor.rank == 1:
        result = MutableDenseNDimArray.zeros(n)

        if tensor.is_upper(0):
            # Contravariant vector (positive index)
            for nu in range(n):
                val = D(tensor[nu], coord)
                for lam in range(n):
                    val += gamma[nu, coord_idx, lam] * tensor[lam]
                result[nu] = Simplify(val)
        else:
            # Covariant vector (negative index)
            for nu in range(n):
                val = D(tensor[nu], coord)
                for lam in range(n):
                    val -= gamma[lam, coord_idx, nu] * tensor[lam]
                result[nu] = Simplify(val)

        return Tensor(f"∇_{coord_idx}({tensor.name})", ImmutableDenseNDimArray(result),
                     tensor.index_positions, metric)

    raise NotImplementedError("Covariant derivative for rank > 1 tensors not yet implemented")


# =============================================================================
# Predefined Metrics
# =============================================================================

def minkowski_metric(dim: int = 4) -> Metric:
    """
    Create flat Minkowski metric η_{μν} = diag(-1, 1, 1, ...).

    Uses mostly-plus convention: (-,+,+,+)
    """
    if dim == 4:
        t, x, y, z = symbols('t x y z')
        coords = [t, x, y, z]
        g = [[-1, 0, 0, 0],
             [0, 1, 0, 0],
             [0, 0, 1, 0],
             [0, 0, 0, 1]]
    elif dim == 2:
        t, x = symbols('t x')
        coords = [t, x]
        g = [[-1, 0], [0, 1]]
    else:
        coords = symbols(f't x1:{dim}')
        g = [[-1 if i == j == 0 else (1 if i == j else 0)
              for j in range(dim)] for i in range(dim)]

    return Metric(coords, g)


def schwarzschild_metric() -> Metric:
    """
    Create Schwarzschild metric for a black hole of mass M.

    ds^2 = -(1-2M/r)dt^2 + (1-2M/r)^-1 dr^2 + r^2 dΩ^2
    """
    t, r, theta, phi = symbols('t r theta phi')
    M = Symbol('M', positive=True)

    f = 1 - 2*M/r

    g = [[   -f,     0,          0,                     0],
         [    0,  1/f,          0,                     0],
         [    0,    0,       r**2,                     0],
         [    0,    0,          0,   r**2*sin(theta)**2]]

    return Metric([t, r, theta, phi], g)


def flrw_metric() -> Metric:
    """
    Create FLRW (Friedmann-Lemaître-Robertson-Walker) metric for cosmology.

    ds^2 = -dt^2 + a(t)^2 [dr^2/(1-kr^2) + r^2 dΩ^2]

    where a(t) is the scale factor and k is the spatial curvature.
    """
    t, r, theta, phi = symbols('t r theta phi')
    a = Function('a')(t)
    k = Symbol('k')

    g = [[               -1,                   0,          0,                     0],
         [                0,  a**2/(1 - k*r**2),          0,                     0],
         [                0,                   0,  a**2*r**2,                     0],
         [                0,                   0,          0,  a**2*r**2*sin(theta)**2]]

    return Metric([t, r, theta, phi], g)


def spherical_metric_3d() -> Metric:
    """
    Create 3D Euclidean metric in spherical coordinates.

    ds^2 = dr^2 + r^2 dθ^2 + r^2 sin^2(θ) dφ^2
    """
    r, theta, phi = symbols('r theta phi')

    g = [[         1,                0,                     0],
         [         0,            r**2,                     0],
         [         0,                0,   r**2*sin(theta)**2]]

    return Metric([r, theta, phi], g)


__all__ = [
    'Metric', 'Tensor', 'CovariantDerivative',
    'minkowski_metric', 'schwarzschild_metric', 'flrw_metric', 'spherical_metric_3d',
]
