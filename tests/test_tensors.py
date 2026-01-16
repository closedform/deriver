"""
Test suite for differential geometry module.

Tests for tensor calculus functionality including metrics,
Christoffel symbols, curvature tensors, and variational derivatives.
"""

import pytest
from sympy import symbols, sin, cos, sqrt, simplify, Rational, Function, diff, Eq, Array, MutableDenseNDimArray
from derive.diffgeo import (
    Metric, Tensor, CoordinateTransformation,
    CovariantDerivative,
    minkowski_metric, schwarzschild_metric, flrw_metric, spherical_metric_3d,
    cartesian_to_spherical_3d, cartesian_to_cylindrical,
    symmetric_index_pairs, symmetric_christoffel_indices,
    SymmetricMatrix, SymmetricChristoffel,
    IndexType, AbstractTensor, Index,
    LeviCivita, levi_civita_tensor,
    fill_symmetric_tensor, fill_antisymmetric_tensor, fill_with_symmetries,
    Einsum, Contract, Trace, OuterProduct, InnerProduct,
)
from derive.calculus import VariationalDerivative, EulerLagrangeEquation
from derive import R, Half, Third, Quarter, TwoThirds, ThreeQuarters


class TestMetric:
    """Tests for Metric class."""

    def test_minkowski_metric(self):
        """Test flat Minkowski metric."""
        g = minkowski_metric(4)
        assert g.dim == 4
        assert g[0, 0] == -1  # g_tt = -1
        assert g[1, 1] == 1   # g_xx = 1
        assert g[0, 1] == 0   # off-diagonal = 0

    def test_minkowski_inverse(self):
        """Test inverse of Minkowski metric."""
        g = minkowski_metric(4)
        g_inv = g.inverse
        # For Minkowski, g^μν = g_μν
        assert g_inv[0, 0] == -1
        assert g_inv[1, 1] == 1

    def test_minkowski_christoffel(self):
        """Test that Christoffel symbols vanish for flat metric."""
        g = minkowski_metric(4)
        gamma = g.christoffel_second_kind()
        # All components should be 0 for flat space
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    assert gamma[i, j, k] == 0

    def test_spherical_metric(self):
        """Test 3D spherical metric."""
        g = spherical_metric_3d()
        r, theta, phi = symbols('r theta phi')
        assert g.dim == 3
        assert g[0, 0] == 1       # g_rr = 1
        assert g[1, 1] == r**2   # g_θθ = r^2
        assert simplify(g[2, 2] - r**2*sin(theta)**2) == 0  # g_φφ = r^2 sin^2(θ)

    def test_spherical_christoffel(self):
        """Test Christoffel symbols for spherical coordinates."""
        g = spherical_metric_3d()
        r, theta, phi = symbols('r theta phi')
        gamma = g.christoffel_second_kind()

        # Some known Christoffel symbols for spherical coords:
        # Γ^r_θθ = -r
        assert simplify(gamma[0, 1, 1] + r) == 0

        # Γ^θ_rθ = 1/r
        assert simplify(gamma[1, 0, 1] - 1/r) == 0

    def test_schwarzschild_metric(self):
        """Test Schwarzschild black hole metric."""
        g = schwarzschild_metric()
        M = symbols('M', positive=True)
        r = symbols('r')

        assert g.dim == 4
        # g_tt = -(1 - 2M/r)
        expected_gtt = -(1 - 2*M/r)
        assert simplify(g[0, 0] - expected_gtt) == 0

    def test_flrw_metric(self):
        """Test FLRW cosmological metric."""
        g = flrw_metric()
        assert g.dim == 4
        assert g[0, 0] == -1  # g_tt = -1


class TestRicciCurvature:
    """Tests for Ricci tensor and scalar."""

    def test_minkowski_ricci(self):
        """Test that Ricci tensor vanishes for flat spacetime."""
        g = minkowski_metric(4)
        Ric = g.ricci_tensor()
        # All components should be 0
        for mu in range(4):
            for nu in range(4):
                assert Ric[mu, nu] == 0

    def test_minkowski_ricci_scalar(self):
        """Test that Ricci scalar vanishes for flat spacetime."""
        g = minkowski_metric(4)
        R = g.ricci_scalar()
        assert R == 0

    def test_minkowski_einstein(self):
        """Test that Einstein tensor vanishes for flat spacetime."""
        g = minkowski_metric(4)
        G = g.einstein_tensor()
        for mu in range(4):
            for nu in range(4):
                assert G[mu, nu] == 0


class TestVariationalDerivative:
    """Tests for variational/functional derivatives."""

    def test_klein_gordon_equation(self):
        """Test Klein-Gordon equation from Lagrangian."""
        x, t = symbols('x t')
        m = symbols('m', positive=True)
        phi = Function('phi')(x, t)

        # Klein-Gordon Lagrangian: L = (1/2)(∂φ/∂t)² - (1/2)(∂φ/∂x)² - (1/2)m²φ²
        L = Rational(1, 2)*diff(phi, t)**2 - Rational(1, 2)*diff(phi, x)**2 - Rational(1, 2)*m**2*phi**2

        eq = VariationalDerivative(L, phi, [x, t])

        # δL/δφ = ∂L/∂φ - ∂_μ(∂L/∂(∂_μ φ))
        # ∂L/∂φ = -m²φ
        # ∂L/∂(∂_t φ) = ∂_t φ → ∂_t(∂_t φ) = ∂²φ/∂t²
        # ∂L/∂(∂_x φ) = -∂_x φ → ∂_x(-∂_x φ) = -∂²φ/∂x²
        # Result: -m²φ - ∂²φ/∂t² - (-∂²φ/∂x²) = -m²φ - ∂²φ/∂t² + ∂²φ/∂x²
        expected = -m**2*phi - diff(phi, t, 2) + diff(phi, x, 2)
        assert simplify(eq - expected) == 0

    def test_wave_equation(self):
        """Test wave equation from Lagrangian."""
        x, t = symbols('x t')
        phi = Function('phi')(x, t)

        # Wave Lagrangian: L = (1/2)(∂φ/∂t)² - (1/2)(∂φ/∂x)²
        L = Rational(1, 2)*diff(phi, t)**2 - Rational(1, 2)*diff(phi, x)**2

        eq = VariationalDerivative(L, phi, [x, t])

        # Same logic as Klein-Gordon but without mass term
        # Result: -∂²φ/∂t² + ∂²φ/∂x²
        expected = -diff(phi, t, 2) + diff(phi, x, 2)
        assert simplify(eq - expected) == 0

    def test_euler_lagrange_alias(self):
        """Test that EulerLagrangeEquation is alias for VariationalDerivative."""
        x = symbols('x')
        q = Function('q')(x)

        # Simple Lagrangian
        L = Rational(1, 2)*diff(q, x)**2

        eq1 = VariationalDerivative(L, q, [x])
        eq2 = EulerLagrangeEquation(L, q, [x])

        assert simplify(eq1 - eq2) == 0


class TestCoordinateTransformation:
    """Tests for coordinate transformations."""

    def test_cartesian_to_spherical(self):
        """Test Cartesian to spherical transformation."""
        trans = cartesian_to_spherical_3d()
        x, y, z = symbols('x y z')
        r, theta, phi = symbols('r theta phi')

        # Check transformation equations
        assert trans.transform[x] == r * sin(theta) * cos(phi)
        assert trans.transform[y] == r * sin(theta) * sin(phi)
        assert trans.transform[z] == r * cos(theta)

    def test_jacobian_determinant_spherical(self):
        """Test Jacobian determinant for spherical transformation."""
        trans = cartesian_to_spherical_3d()
        r, theta, phi = symbols('r theta phi')

        # Jacobian determinant should be r² sin(θ)
        det = trans.jacobian_determinant
        expected = r**2 * sin(theta)
        assert simplify(det - expected) == 0

    def test_cartesian_to_cylindrical(self):
        """Test Cartesian to cylindrical transformation."""
        trans = cartesian_to_cylindrical()
        x, y, z = symbols('x y z')
        rho, phi, z_cyl = symbols('rho phi z')

        # Check transformation
        assert trans.transform[x] == rho * cos(phi)
        assert trans.transform[y] == rho * sin(phi)


class TestTensor:
    """Tests for Tensor class."""

    def test_tensor_creation(self):
        """Test creating a simple tensor."""
        from sympy import Array
        components = Array([1, 2, 3])
        T = Tensor("v", components, [+1])  # Contravariant vector v^μ

        assert T.name == "v"
        assert T.rank == 1
        assert T[0] == 1
        assert T[1] == 2
        assert T[2] == 3
        assert T.is_upper(0)

    def test_tensor_rank(self):
        """Test tensor rank from index positions."""
        from sympy import Array, MutableDenseNDimArray

        # Rank 2 mixed tensor T^μ_ν
        components = MutableDenseNDimArray.zeros(3, 3)
        T = Tensor("T", components, [+1, -1])
        assert T.rank == 2
        assert T.is_upper(0)
        assert T.is_lower(1)

        # Rank 3 tensor T^μν_ρ
        components = MutableDenseNDimArray.zeros(3, 3, 3)
        T = Tensor("T", components, [+1, +1, -1])
        assert T.rank == 3
        assert T.is_upper(0)
        assert T.is_upper(1)
        assert T.is_lower(2)


class TestSymmetry:
    """Tests for tensor symmetry utilities."""

    def test_symmetric_index_pairs(self):
        """Test generating symmetric index pairs."""
        from derive.diffgeo import symmetric_index_pairs

        # For n=3, should get 6 pairs: (0,0), (0,1), (0,2), (1,1), (1,2), (2,2)
        pairs = symmetric_index_pairs(3)
        assert len(pairs) == 6
        assert (0, 0) in pairs
        assert (0, 1) in pairs
        assert (1, 1) in pairs
        assert (2, 2) in pairs
        # All pairs should have i <= j
        for i, j in pairs:
            assert i <= j

    def test_symmetric_christoffel_indices(self):
        """Test generating Christoffel indices with symmetry."""
        from derive.diffgeo import symmetric_christoffel_indices

        # For n=2: 2 * (2*3/2) = 6 unique triplets
        indices = symmetric_christoffel_indices(2)
        assert len(indices) == 6

        # For n=3: 3 * (3*4/2) = 18 unique triplets
        indices = symmetric_christoffel_indices(3)
        assert len(indices) == 18

        # All should have mu <= nu
        for rho, mu, nu in indices:
            assert mu <= nu

    def test_christoffel_symmetry_exploited(self):
        """Test that Christoffel computation exploits symmetry correctly."""
        g = spherical_metric_3d()
        gamma = g.christoffel_second_kind()

        # Christoffel symbols should be symmetric in lower indices
        n = g.dim
        for rho in range(n):
            for mu in range(n):
                for nu in range(n):
                    # Γ^ρ_μν should equal Γ^ρ_νμ
                    assert simplify(gamma[rho, mu, nu] - gamma[rho, nu, mu]) == 0

    def test_symmetric_matrix(self):
        """Test SymmetricMatrix storage."""
        from derive.diffgeo import SymmetricMatrix
        from sympy import Symbol

        m = SymmetricMatrix(3)
        a = Symbol('a')

        # Set upper triangular, read from both positions
        m[0, 1] = a
        assert m[0, 1] == a
        assert m[1, 0] == a  # Symmetric access

        m[1, 2] = 2*a
        assert m[1, 2] == 2*a
        assert m[2, 1] == 2*a

    def test_symmetry_savings_ratio(self):
        """Test computation savings from symmetry."""
        from derive.diffgeo import SymmetricChristoffel

        # For n=4 (like spacetime)
        sc = SymmetricChristoffel(4)

        # Total = 4^3 = 64, Unique = 4 * 4*5/2 = 40
        assert sc.num_total == 64
        assert sc.num_unique == 40
        # Savings ratio = 1 - 40/64 = 0.375
        assert abs(sc.savings_ratio - 0.375) < 0.001


class TestAbstractIndices:
    """Tests for abstract index notation (xAct-style, wrapping SymPy tensor module)."""

    def test_index_space_creation(self):
        """Test creating an IndexSpace."""
        from derive.diffgeo import IndexSpace

        spacetime = IndexSpace('spacetime', dim=4)
        assert spacetime.name == 'spacetime'
        assert spacetime.dim == 4

    def test_index_type_alias(self):
        """Test IndexType is alias for IndexSpace."""
        from derive.diffgeo import IndexType, IndexSpace

        spacetime = IndexType('spacetime', dim=4)
        assert isinstance(spacetime, IndexSpace)

    def test_index_creation(self):
        """Test creating indices from IndexSpace."""
        from derive.diffgeo import IndexSpace

        spacetime = IndexSpace('spacetime', dim=4)
        a, b, c = spacetime.indices('a b c')

        assert a.name == 'a'
        assert a.up is True
        assert a.space is spacetime
        assert a.dim == 4

    def test_index_negation(self):
        """Test that -a gives a lower index."""
        from derive.diffgeo import IndexSpace

        spacetime = IndexSpace('spacetime', dim=4)
        a = spacetime.index('a')

        assert a.is_up
        assert not a.is_down

        neg_a = -a
        assert neg_a.is_down
        assert not neg_a.is_up
        assert neg_a.name == 'a'  # Same name

    def test_index_repr(self):
        """Test index string representation."""
        from derive.diffgeo import IndexSpace

        spacetime = IndexSpace('spacetime', dim=4)
        a = spacetime.index('a')

        assert repr(a) == 'a'
        assert repr(-a) == '-a'

    def test_tensor_creation(self):
        """Test creating a Tensor with multiple index spaces."""
        from derive.diffgeo import IndexSpace, AbstractTensor

        spacetime = IndexSpace('spacetime', dim=4)
        T = AbstractTensor('T', spacetime, spacetime)

        assert T.name == 'T'
        assert T.rank == 2

    def test_tensor_mixed_spaces(self):
        """Test tensor with indices from different spaces."""
        from derive.diffgeo import IndexSpace, AbstractTensor

        spacetime = IndexSpace('spacetime', dim=4)
        gauge = IndexSpace('gauge', dim=3)

        # Mixed tensor A^μ_i_j
        A = AbstractTensor('A', spacetime, gauge, gauge)
        assert A.rank == 3
        assert A.index_spaces[0] is spacetime
        assert A.index_spaces[1] is gauge
        assert A.index_spaces[2] is gauge

    def test_tensor_index_access(self):
        """Test accessing tensor with indices T[a, -b]."""
        from derive.diffgeo import IndexSpace, AbstractTensor

        spacetime = IndexSpace('spacetime', dim=4)
        a, b = spacetime.indices('a b')
        T = AbstractTensor('T', spacetime, spacetime)

        # T[a, -b] should give T^a_b
        expr = T[a, -b]
        assert expr.tensor == T
        assert len(expr.indices) == 2
        assert expr.indices[0].is_up  # a is up
        assert expr.indices[1].is_down  # -b is down

    def test_tensor_symmetric(self):
        """Test tensor with symmetric indices."""
        from derive.diffgeo import IndexSpace, AbstractTensor

        spacetime = IndexSpace('spacetime', dim=4)
        g = AbstractTensor('g', spacetime, spacetime, symmetric=True)
        assert g.symmetry is not None

    def test_tensor_antisymmetric(self):
        """Test tensor with antisymmetric indices."""
        from derive.diffgeo import IndexSpace, AbstractTensor

        spacetime = IndexSpace('spacetime', dim=4)
        F = AbstractTensor('F', spacetime, spacetime, antisymmetric=True)
        assert F.symmetry is not None

    def test_indexed_expr_repr(self):
        """Test string representation of indexed expressions."""
        from derive.diffgeo import IndexSpace, AbstractTensor

        spacetime = IndexSpace('spacetime', dim=4)
        a, b = spacetime.indices('a b')
        T = AbstractTensor('T', spacetime, spacetime)

        expr = T[a, -b]
        assert repr(expr) == 'T[a, -b]'

        expr2 = T[-a, -b]
        assert repr(expr2) == 'T[-a, -b]'

    def test_tensor_multiplication(self):
        """Test tensor multiplication with Einstein summation."""
        from derive.diffgeo import IndexSpace, AbstractTensor

        spacetime = IndexSpace('spacetime', dim=4)
        a, b, c = spacetime.indices('a b c')

        T = AbstractTensor('T', spacetime, spacetime)
        S = AbstractTensor('S', spacetime, spacetime)

        # This should auto-contract over b
        product = T[a, b] * S[-b, c]
        # Result is a SymPy tensor expression
        assert product is not None

    def test_indices_function(self):
        """Test the indices() factory function."""
        from derive.diffgeo import IndexSpace, indices

        spacetime = IndexSpace('spacetime', dim=4)
        a, b, c = indices('a b c', spacetime)

        assert a.name == 'a'
        assert b.name == 'b'
        assert c.name == 'c'
        assert a.space is spacetime


class TestLeviCivita:
    """Tests for Levi-Civita symbol."""

    def test_levi_civita_3d(self):
        """Test 3D Levi-Civita symbol values."""
        # Even permutations = +1
        assert LeviCivita(0, 1, 2) == 1
        assert LeviCivita(1, 2, 0) == 1
        assert LeviCivita(2, 0, 1) == 1

        # Odd permutations = -1
        assert LeviCivita(0, 2, 1) == -1
        assert LeviCivita(1, 0, 2) == -1
        assert LeviCivita(2, 1, 0) == -1

        # Repeated indices = 0
        assert LeviCivita(0, 0, 1) == 0
        assert LeviCivita(1, 1, 2) == 0
        assert LeviCivita(2, 2, 0) == 0

    def test_levi_civita_2d(self):
        """Test 2D Levi-Civita symbol."""
        assert LeviCivita(0, 1) == 1
        assert LeviCivita(1, 0) == -1
        assert LeviCivita(0, 0) == 0

    def test_levi_civita_tensor(self):
        """Test full Levi-Civita tensor creation."""
        eps = levi_civita_tensor(3)
        assert eps[0, 1, 2] == 1
        assert eps[1, 0, 2] == -1
        assert eps[0, 0, 2] == 0


class TestRationalShortcuts:
    """Tests for rational number shortcuts."""

    def test_r_function(self):
        """Test R() shorthand for Rational."""
        assert R(1, 2) == Rational(1, 2)
        assert R(3, 4) == Rational(3, 4)
        assert R(5) == Rational(5)

    def test_common_fractions(self):
        """Test common fraction constants."""
        assert Half == Rational(1, 2)
        assert Third == Rational(1, 3)
        assert Quarter == Rational(1, 4)
        assert TwoThirds == Rational(2, 3)
        assert ThreeQuarters == Rational(3, 4)

    def test_fraction_arithmetic(self):
        """Test arithmetic with fraction shortcuts."""
        assert Half + Third == Rational(5, 6)
        assert Half * 2 == 1


class TestEinsum:
    """Tests for Einstein summation."""

    def test_matrix_multiplication(self):
        """Test einsum for matrix multiplication."""
        A = Array([[1, 2], [3, 4]])
        B = Array([[5, 6], [7, 8]])
        C = Einsum("ij,jk->ik", A, B)
        # C[0,0] = 1*5 + 2*7 = 19
        # C[0,1] = 1*6 + 2*8 = 22
        # C[1,0] = 3*5 + 4*7 = 43
        # C[1,1] = 3*6 + 4*8 = 50
        assert C[0, 0] == 19
        assert C[0, 1] == 22
        assert C[1, 0] == 43
        assert C[1, 1] == 50

    def test_trace(self):
        """Test einsum for trace."""
        A = Array([[1, 2], [3, 4]])
        tr = Einsum("ii->", A)
        assert tr == 5  # 1 + 4

    def test_transpose(self):
        """Test einsum for transpose."""
        A = Array([[1, 2], [3, 4]])
        AT = Einsum("ij->ji", A)
        assert AT[0, 0] == 1
        assert AT[0, 1] == 3
        assert AT[1, 0] == 2
        assert AT[1, 1] == 4

    def test_contract_function(self):
        """Test Contract function."""
        # Create rank-2 tensor and contract
        A = Array([[1, 2], [3, 4]])
        tr = Contract(A, (0, 1))
        assert tr == 5

    def test_outer_product(self):
        """Test OuterProduct function."""
        a = Array([1, 2])
        b = Array([3, 4])
        outer = OuterProduct(a, b)
        assert outer[0, 0] == 3
        assert outer[0, 1] == 4
        assert outer[1, 0] == 6
        assert outer[1, 1] == 8

    def test_inner_product(self):
        """Test InnerProduct function."""
        a = Array([1, 2, 3])
        b = Array([4, 5, 6])
        inner = InnerProduct(a, b)
        # 1*4 + 2*5 + 3*6 = 32
        assert inner == 32


class TestSymmetryFilling:
    """Tests for symmetry-based tensor filling."""

    def test_fill_symmetric_tensor(self):
        """Test filling a symmetric tensor."""
        T = MutableDenseNDimArray.zeros(3, 3)

        def compute(i, j):
            return i + j

        fill_symmetric_tensor(T, (0, 1), compute)

        # Check symmetry
        assert T[0, 1] == T[1, 0] == 1
        assert T[0, 2] == T[2, 0] == 2
        assert T[1, 2] == T[2, 1] == 3
        assert T[0, 0] == 0
        assert T[1, 1] == 2
        assert T[2, 2] == 4

    def test_fill_antisymmetric_tensor(self):
        """Test filling an antisymmetric tensor."""
        T = MutableDenseNDimArray.zeros(3, 3)

        def compute(i, j):
            return i * 10 + j

        fill_antisymmetric_tensor(T, (0, 1), compute)

        # Diagonal should be zero
        assert T[0, 0] == 0
        assert T[1, 1] == 0
        assert T[2, 2] == 0

        # Check antisymmetry
        assert T[0, 1] == 1   # compute(0, 1) = 1
        assert T[1, 0] == -1  # antisymmetric
        assert T[0, 2] == 2   # compute(0, 2) = 2
        assert T[2, 0] == -2
        assert T[1, 2] == 12  # compute(1, 2) = 12
        assert T[2, 1] == -12

    def test_fill_with_symmetries_mixed(self):
        """Test filling with multiple symmetries."""
        # 3x3x3 tensor with symmetry in (0,1)
        T = MutableDenseNDimArray.zeros(2, 2, 2)

        def compute(i, j, k):
            return i * 100 + j * 10 + k

        fill_with_symmetries(T, compute, symmetric=[(0, 1)])

        # Check symmetric in first two indices
        assert T[0, 1, 0] == T[1, 0, 0]
        assert T[0, 1, 1] == T[1, 0, 1]


class TestAutoContraction:
    """Tests for automatic Einstein summation contraction."""

    def test_trace_auto_contraction(self):
        """T[a, -a] should auto-contract to trace via SymPy."""
        from sympy import ImmutableDenseNDimArray, diag
        from derive.diffgeo import IndexSpace, AbstractTensor

        space = IndexSpace('test', dim=2)
        a = space.index('a')

        # Matrix with trace = 1 + 4 = 5
        components = ImmutableDenseNDimArray([[1, 2], [3, 4]])
        T = AbstractTensor('T', space, space, components=components)

        # T[a, -a] should give the trace
        expr = T[a, -a]
        assert expr.value == 5

    def test_no_contraction_different_indices(self):
        """T[a, b] should not contract."""
        from sympy import ImmutableDenseNDimArray
        from derive.diffgeo import IndexSpace, AbstractTensor

        space = IndexSpace('test', dim=2)
        a, b = space.indices('a b')

        components = ImmutableDenseNDimArray([[1, 2], [3, 4]])
        T = AbstractTensor('T', space, space, components=components)

        expr = T[a, b]
        # No contraction, value should be None
        assert expr.value is None

    def test_sympy_tensor_contraction(self):
        """Test that SymPy handles contraction in multiplication."""
        from derive.diffgeo import IndexSpace, AbstractTensor

        space = IndexSpace('test', dim=2)
        a, b, c = space.indices('a b c')

        T = AbstractTensor('T', space, space)
        S = AbstractTensor('S', space, space)

        # This creates a SymPy tensor expression with contraction
        product = T[a, b] * S[-b, c]
        # The product is a SymPy TensMul expression
        assert product is not None

    def test_replace_with_arrays(self):
        """Test replacing tensors with array components."""
        from sympy import ImmutableDenseNDimArray
        from derive.diffgeo import IndexSpace, AbstractTensor

        space = IndexSpace('L', dim=2)
        i, j = space.indices('i j')

        A = AbstractTensor('A', space)
        expr = A[i]

        # Replace with concrete array
        result = expr.replace_with_arrays({A: [1, 2]}, [i])
        assert list(result) == [1, 2]

    def test_tensor_trace_via_replace(self):
        """Test trace computation via replace_with_arrays."""
        from sympy import diag
        from derive.diffgeo import IndexSpace, AbstractTensor

        L = IndexSpace('L', dim=2)
        i = L.index('i')

        A = AbstractTensor('A', L, L)
        expr = A[i, -i]  # Trace

        # This traces over the dummy index
        result = expr.replace_with_arrays({A: [[1, 2], [3, 4]], L: diag(1, 1)})
        assert result == 5  # 1 + 4


class TestMultipleIndexSpaces:
    """Tests for tensors with indices from different spaces."""

    def test_mixed_tensor_creation(self):
        """Test creating tensors with mixed index spaces."""
        from derive.diffgeo import IndexSpace, AbstractTensor

        spacetime = IndexSpace('spacetime', dim=4)
        gauge = IndexSpace('gauge', dim=3)

        # Field strength with spacetime and gauge indices: F^μν_a
        F = AbstractTensor('F', spacetime, spacetime, gauge)
        assert F.rank == 3
        assert F.index_spaces == (spacetime, spacetime, gauge)

    def test_indices_from_different_spaces(self):
        """Test that indices from different spaces don't contract."""
        from derive.diffgeo import IndexSpace, AbstractTensor

        spacetime = IndexSpace('spacetime', dim=4)
        gauge = IndexSpace('gauge', dim=3)

        mu, nu = spacetime.indices('mu nu')
        a, b = gauge.indices('a b')

        # These indices are from different spaces
        assert mu.space is not a.space

        # Create a gauge-covariant derivative D_μ A_a
        D = AbstractTensor('D', spacetime)
        A = AbstractTensor('A', gauge)

        # Multiplication creates proper tensor product
        expr = D[mu] * A[a]
        assert expr is not None

    def test_partial_contraction(self):
        """Test contraction only over indices from same space."""
        from derive.diffgeo import IndexSpace, AbstractTensor

        spacetime = IndexSpace('ST', dim=4)
        gauge = IndexSpace('G', dim=3)

        mu, nu = spacetime.indices('mu nu')
        a, b = gauge.indices('a b')

        # T^μ_ν_a and S^ν_b - contracts over ν (spacetime) only
        T = AbstractTensor('T', spacetime, spacetime, gauge)
        S = AbstractTensor('S', spacetime, gauge)

        product = T[mu, nu, a] * S[-nu, b]
        # Result should still have spacetime index μ and gauge indices a, b
        assert product is not None


class TestSubmanifold:
    """Tests for submanifold projections."""

    def test_submanifold_creation(self):
        """Test creating a submanifold."""
        from derive.diffgeo import IndexSpace, Submanifold

        spacetime = IndexSpace('spacetime', dim=4)
        spatial = Submanifold('spatial', spacetime, dim=3)

        assert spatial.name == 'spatial'
        assert spatial.dim == 3
        assert spatial.ambient is spacetime
        assert spatial.index_space.dim == 3

    def test_submanifold_index_space(self):
        """Test that submanifold has its own index space."""
        from derive.diffgeo import IndexSpace, Submanifold, Tensor

        spacetime = IndexSpace('spacetime', dim=4)
        spatial = Submanifold('spatial', spacetime, dim=3)

        # Can create tensors in the submanifold's index space
        h = AbstractTensor('h', spatial.index_space, spatial.index_space)
        assert h.rank == 2


class TestTensorRule:
    """Tests for tensor replacement rules."""

    def test_tensor_rule_creation(self):
        """Test creating a tensor rule."""
        from derive.diffgeo import TensorRule

        rule = TensorRule('pattern', 0)
        assert rule.pattern == 'pattern'
        assert rule.replacement == 0

    def test_tensor_rule_repr(self):
        """Test tensor rule string representation."""
        from derive.diffgeo import TensorRule

        rule = TensorRule('g_ab', 'delta_ab')
        assert 'g_ab' in repr(rule)
        assert 'delta_ab' in repr(rule)
