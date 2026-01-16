"""
symmetry.py - Tensor Symmetry Handling

Provides symmetry tracking and optimized storage for symmetric tensors.
Exploiting symmetries reduces both computation time and storage requirements.
"""

from enum import Enum, auto
from typing import List, Tuple, Dict, Optional, Iterator, Callable
from itertools import product as iterproduct
import sympy as sp
from sympy import MutableDenseNDimArray, ImmutableDenseNDimArray


class Symmetry(Enum):
    """
    Types of tensor symmetry.

    NONE: No symmetry (all components independent)
    SYMMETRIC: Symmetric under index exchange (e.g., g_μν = g_νμ)
    ANTISYMMETRIC: Antisymmetric under exchange (e.g., F_μν = -F_νμ)
    """
    NONE = auto()
    SYMMETRIC = auto()
    ANTISYMMETRIC = auto()


def symmetric_index_pairs(n: int) -> List[Tuple[int, int]]:
    """
    Generate all unique index pairs for a symmetric 2-tensor.

    For n×n symmetric tensor, returns (n*(n+1))/2 pairs.
    Only yields (i, j) where i <= j.

    Args:
        n: Dimension

    Returns:
        List of (i, j) tuples with i <= j

    Example:
        >>> symmetric_index_pairs(3)
        [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
    """
    pairs = []
    for i in range(n):
        for j in range(i, n):
            pairs.append((i, j))
    return pairs


def symmetric_christoffel_indices(n: int) -> List[Tuple[int, int, int]]:
    """
    Generate unique index triplets for Christoffel symbols Γ^ρ_μν.

    Christoffel symbols are symmetric in lower indices: Γ^ρ_μν = Γ^ρ_νμ
    So for each ρ, we only need μ <= ν pairs.

    Total unique components: n * n*(n+1)/2

    Args:
        n: Dimension

    Returns:
        List of (rho, mu, nu) tuples with mu <= nu
    """
    indices = []
    for rho in range(n):
        for mu in range(n):
            for nu in range(mu, n):
                indices.append((rho, mu, nu))
    return indices


class SymmetricMatrix:
    """
    Efficient storage for symmetric matrices.

    Only stores upper triangular elements (n*(n+1)/2 instead of n*n).
    Automatically returns symmetric elements on access.

    Example:
        >>> m = SymmetricMatrix(3)
        >>> m[0, 1] = 5
        >>> m[1, 0]  # Returns 5 (symmetric access)
        5
    """

    def __init__(self, n: int, dtype=None):
        """
        Initialize symmetric matrix.

        Args:
            n: Matrix dimension (n×n)
            dtype: Optional element type
        """
        self.n = n
        self._size = n * (n + 1) // 2
        self._data: Dict[Tuple[int, int], sp.Expr] = {}

    def _canonical_index(self, i: int, j: int) -> Tuple[int, int]:
        """Get canonical (i <= j) form of index pair."""
        return (min(i, j), max(i, j))

    def __getitem__(self, indices: Tuple[int, int]) -> sp.Expr:
        i, j = indices
        key = self._canonical_index(i, j)
        return self._data.get(key, sp.Integer(0))

    def __setitem__(self, indices: Tuple[int, int], value):
        i, j = indices
        key = self._canonical_index(i, j)
        self._data[key] = value

    def to_matrix(self) -> sp.Matrix:
        """Convert to full sympy Matrix."""
        m = sp.Matrix.zeros(self.n, self.n)
        for (i, j), val in self._data.items():
            m[i, j] = val
            m[j, i] = val
        return m

    def unique_elements(self) -> Iterator[Tuple[int, int, sp.Expr]]:
        """Iterate over unique (i, j, value) triples."""
        for (i, j), val in self._data.items():
            yield i, j, val


class SymmetricChristoffel:
    """
    Efficient storage for Christoffel symbols Γ^ρ_μν.

    Exploits symmetry Γ^ρ_μν = Γ^ρ_νμ to reduce storage.
    Only stores components with μ <= ν.
    """

    def __init__(self, n: int):
        """
        Initialize Christoffel symbol storage.

        Args:
            n: Dimension (number of coordinates)
        """
        self.n = n
        # For each ρ, store symmetric matrix in (μ, ν)
        self._data: Dict[Tuple[int, int, int], sp.Expr] = {}

    def _canonical_index(self, rho: int, mu: int, nu: int) -> Tuple[int, int, int]:
        """Get canonical form with mu <= nu."""
        if mu <= nu:
            return (rho, mu, nu)
        return (rho, nu, mu)

    def __getitem__(self, indices: Tuple[int, int, int]) -> sp.Expr:
        rho, mu, nu = indices
        key = self._canonical_index(rho, mu, nu)
        return self._data.get(key, sp.Integer(0))

    def __setitem__(self, indices: Tuple[int, int, int], value):
        rho, mu, nu = indices
        key = self._canonical_index(rho, mu, nu)
        self._data[key] = value

    def to_array(self) -> ImmutableDenseNDimArray:
        """Convert to full SymPy Array."""
        arr = MutableDenseNDimArray.zeros(self.n, self.n, self.n)
        for (rho, mu, nu), val in self._data.items():
            arr[rho, mu, nu] = val
            arr[rho, nu, mu] = val  # Symmetric
        return ImmutableDenseNDimArray(arr)

    def unique_indices(self) -> List[Tuple[int, int, int]]:
        """Get list of unique indices that need to be computed."""
        return symmetric_christoffel_indices(self.n)

    @property
    def num_unique(self) -> int:
        """Number of unique components."""
        return self.n * self.n * (self.n + 1) // 2

    @property
    def num_total(self) -> int:
        """Total components (if not exploiting symmetry)."""
        return self.n ** 3

    @property
    def savings_ratio(self) -> float:
        """Ratio of computation saved by exploiting symmetry."""
        return 1 - self.num_unique / self.num_total


def LeviCivita(*indices: int) -> int:
    """
    Compute the Levi-Civita symbol (totally antisymmetric tensor).

    ε_{i₁i₂...iₙ} = +1 if (i₁,i₂,...,iₙ) is an even permutation of (0,1,...,n-1)
                  = -1 if odd permutation
                  = 0 if any indices are repeated

    Args:
        *indices: Index values (integers)

    Returns:
        +1, -1, or 0

    Examples:
        >>> LeviCivita(0, 1, 2)
        1
        >>> LeviCivita(1, 0, 2)
        -1
        >>> LeviCivita(0, 0, 2)
        0
        >>> LeviCivita(0, 1)  # 2D
        1
        >>> LeviCivita(1, 0)  # 2D
        -1
    """
    n = len(indices)
    idx_list = list(indices)

    # Check for repeated indices
    if len(set(idx_list)) != n:
        return 0

    # Count inversions (number of pairs where larger comes before smaller)
    inversions = 0
    for i in range(n):
        for j in range(i + 1, n):
            if idx_list[i] > idx_list[j]:
                inversions += 1

    return 1 if inversions % 2 == 0 else -1


def levi_civita_tensor(n: int) -> ImmutableDenseNDimArray:
    """
    Create the full Levi-Civita tensor for n dimensions.

    Args:
        n: Number of dimensions

    Returns:
        n-dimensional array with Levi-Civita symbol values

    Examples:
        >>> eps = levi_civita_tensor(3)
        >>> eps[0, 1, 2]
        1
        >>> eps[1, 0, 2]
        -1
    """
    from itertools import product as iterproduct

    shape = tuple([n] * n)
    arr = MutableDenseNDimArray.zeros(*shape)

    for indices in iterproduct(range(n), repeat=n):
        arr[indices] = LeviCivita(*indices)

    return ImmutableDenseNDimArray(arr)


def fill_symmetric_tensor(
    tensor: MutableDenseNDimArray,
    symmetric_pair: Tuple[int, int],
    compute_fn,
) -> None:
    """
    Fill a tensor exploiting symmetry in specified index pair.

    For T_{...i...j...} symmetric in (i,j), computes only i <= j components
    and fills i > j via symmetry.

    Args:
        tensor: Mutable array to fill
        symmetric_pair: Tuple (pos1, pos2) of symmetric index positions
        compute_fn: Function(*indices) -> value for computing components
    """
    from itertools import product as iterproduct

    shape = tensor.shape
    rank = len(shape)
    pos1, pos2 = symmetric_pair

    # Generate all index combinations where index at pos1 <= index at pos2
    ranges = [range(s) for s in shape]

    for indices in iterproduct(*ranges):
        i1, i2 = indices[pos1], indices[pos2]
        if i1 <= i2:
            val = compute_fn(*indices)
            tensor[indices] = val
            if i1 != i2:
                # Swap indices at pos1 and pos2
                swapped = list(indices)
                swapped[pos1], swapped[pos2] = swapped[pos2], swapped[pos1]
                tensor[tuple(swapped)] = val  # Symmetric


def fill_antisymmetric_tensor(
    tensor: MutableDenseNDimArray,
    antisymmetric_pair: Tuple[int, int],
    compute_fn,
) -> None:
    """
    Fill a tensor exploiting antisymmetry in specified index pair.

    For T_{...i...j...} antisymmetric in (i,j), computes only i < j components,
    sets i > j via antisymmetry, and i == j components are zero.

    Args:
        tensor: Mutable array to fill
        antisymmetric_pair: Tuple (pos1, pos2) of antisymmetric index positions
        compute_fn: Function(*indices) -> value for computing components
    """
    from itertools import product as iterproduct

    shape = tensor.shape
    pos1, pos2 = antisymmetric_pair

    ranges = [range(s) for s in shape]

    for indices in iterproduct(*ranges):
        i1, i2 = indices[pos1], indices[pos2]
        if i1 < i2:
            val = compute_fn(*indices)
            tensor[indices] = val
            # Swap indices at pos1 and pos2 for antisymmetric component
            swapped = list(indices)
            swapped[pos1], swapped[pos2] = swapped[pos2], swapped[pos1]
            tensor[tuple(swapped)] = -val  # Antisymmetric
        # i1 == i2 case: already zero from initialization
        # i1 > i2 case: handled by swapping


def fill_with_symmetries(
    tensor: MutableDenseNDimArray,
    compute_fn,
    symmetric: Optional[List[Tuple[int, int]]] = None,
    antisymmetric: Optional[List[Tuple[int, int]]] = None,
) -> None:
    """
    Fill tensor with multiple symmetry constraints.

    Generates only the independent components based on symmetries and fills
    the rest by symmetry relations.

    Args:
        tensor: Mutable array to fill
        compute_fn: Function(*indices) -> value
        symmetric: List of symmetric index pairs
        antisymmetric: List of antisymmetric index pairs

    Example:
        >>> # Riemann tensor R^ρ_{σμν} antisymmetric in (μ,ν) at positions (2,3)
        >>> fill_with_symmetries(R, compute_R, antisymmetric=[(2, 3)])
    """
    from itertools import product as iterproduct

    shape = tensor.shape
    sym_pairs = symmetric or []
    antisym_pairs = antisymmetric or []

    def is_canonical(indices):
        """Check if indices are in canonical form given symmetries."""
        for p1, p2 in sym_pairs:
            if indices[p1] > indices[p2]:
                return False
        for p1, p2 in antisym_pairs:
            if indices[p1] > indices[p2]:
                return False
            if indices[p1] == indices[p2]:
                return False  # Zero by antisymmetry
        return True

    def get_sign_and_canonical(indices):
        """Get sign and canonical form of indices."""
        indices = list(indices)
        sign = 1
        for p1, p2 in antisym_pairs:
            if indices[p1] > indices[p2]:
                indices[p1], indices[p2] = indices[p2], indices[p1]
                sign *= -1
            elif indices[p1] == indices[p2]:
                return 0, None  # Zero
        for p1, p2 in sym_pairs:
            if indices[p1] > indices[p2]:
                indices[p1], indices[p2] = indices[p2], indices[p1]
        return sign, tuple(indices)

    # Compute canonical components
    computed = {}
    for indices in iterproduct(*[range(s) for s in shape]):
        if is_canonical(indices):
            val = compute_fn(*indices)
            computed[indices] = val
            tensor[indices] = val

    # Fill non-canonical components
    for indices in iterproduct(*[range(s) for s in shape]):
        if not is_canonical(indices):
            sign, canonical = get_sign_and_canonical(indices)
            if sign == 0:
                tensor[indices] = 0
            else:
                tensor[indices] = sign * computed[canonical]


__all__ = [
    'Symmetry',
    'symmetric_index_pairs',
    'symmetric_christoffel_indices',
    'SymmetricMatrix',
    'SymmetricChristoffel',
    'LeviCivita',
    'levi_civita_tensor',
    'fill_symmetric_tensor',
    'fill_antisymmetric_tensor',
    'fill_with_symmetries',
]
