"""
indices.py - Abstract Index Notation for Tensors

Thin wrapper around SymPy's tensor module providing xAct-style notation:
- Positive indices (a, +a) are upper/contravariant
- Negative indices (-a) are lower/covariant
- Tensors can be accessed with T[a, -b] notation
- Multiple index types supported (spacetime, spinor, gauge, etc.)
- Automatic Einstein summation contraction
- Metric raising/lowering
- Component replacement with arrays

Example:
    >>> from derive.diffgeo.indices import IndexSpace, indices
    >>> spacetime = IndexSpace('spacetime', dim=4)
    >>> a, b, c = indices('a b c', spacetime)
    >>> T = Tensor('T', spacetime, spacetime)
    >>> T[a, -b]  # T^a_b
    >>> T[a, -a]  # Trace (auto-contracts)
"""

from sympy.tensor.tensor import (
    TensorIndexType as _TensorIndexType,
    TensorIndex as _TensorIndex,
    TensorHead as _TensorHead,
    TensorSymmetry as _TensorSymmetry,
    tensor_indices as _tensor_indices,
)
from sympy import Array, ImmutableDenseNDimArray, Symbol, diag
from typing import List, Optional, Tuple, Union, Any, Sequence

# Re-export SymPy's TensorSymmetry for direct use
TensorSymmetry = _TensorSymmetry


class IndexSpace:
    """
    Defines a space that tensor indices can live in.

    Each IndexSpace represents a distinct vector space with its own dimension
    and optional metric. Indices from different spaces cannot contract.

    Examples:
        >>> spacetime = IndexSpace('spacetime', dim=4)
        >>> spinor = IndexSpace('spinor', dim=2)
        >>> gauge = IndexSpace('SU3', dim=8)  # adjoint rep
    """

    def __init__(
        self,
        name: str,
        dim: Optional[int] = None,
        metric: Optional[Any] = None,
        metric_antisym: bool = False,
        dummy_name: Optional[str] = None,
    ):
        """
        Create an index space.

        Args:
            name: Name of the space (e.g., 'spacetime', 'spinor')
            dim: Dimension of the space (can be symbolic if None)
            metric: Optional metric tensor components for raising/lowering
            metric_antisym: True for antisymmetric metric (e.g., symplectic)
            dummy_name: Prefix for auto-generated dummy indices
        """
        self.name = name
        self._dim = dim
        self._metric_components = metric

        # Create underlying SymPy TensorIndexType
        dummy = dummy_name or name[0].upper()
        metric_symmetry = -1 if metric_antisym else 1
        self._sympy_type = _TensorIndexType(
            name,
            dummy_name=dummy,
            dim=dim,
            metric_symmetry=metric_symmetry,
        )

    @property
    def dim(self) -> Optional[int]:
        """Dimension of this index space."""
        return self._dim

    @property
    def metric(self):
        """The metric tensor for this space (for raising/lowering)."""
        return self._sympy_type.metric

    @property
    def sympy_type(self) -> _TensorIndexType:
        """Access underlying SymPy TensorIndexType."""
        return self._sympy_type

    def index(self, name: str, up: bool = True) -> 'Index':
        """Create a single index in this space."""
        return Index(name, self, up=up)

    def indices(self, names: str) -> Tuple['Index', ...]:
        """
        Create multiple indices in this space.

        Args:
            names: Space-separated index names (e.g., 'a b c' or 'mu nu rho')
        """
        return tuple(Index(n, self) for n in names.split())

    def __repr__(self) -> str:
        dim_str = f", dim={self._dim}" if self._dim else ""
        return f"IndexSpace({self.name!r}{dim_str})"


# Backwards compatibility alias
IndexType = IndexSpace


class Index:
    """
    An abstract tensor index in a specific space.

    Indices have a name, space, and position (up/down).
    Use negation to flip position: -a gives the covariant version.

    Examples:
        >>> a = Index('a', spacetime)
        >>> -a  # Lower index
        >>> a.is_up  # True
        >>> (-a).is_down  # True
    """

    def __init__(self, name: str, space: IndexSpace, up: bool = True):
        """
        Create an abstract index.

        Args:
            name: Index name (e.g., 'a', 'mu', 'nu')
            space: The IndexSpace this index belongs to
            up: True for upper (contravariant), False for lower (covariant)
        """
        self.name = name
        self.space = space
        self.up = up
        # Create underlying SymPy TensorIndex
        self._sympy_index = _TensorIndex(name, space.sympy_type, is_up=up)

    # Backwards compatibility
    @property
    def index_type(self) -> IndexSpace:
        return self.space

    def __neg__(self) -> 'Index':
        """Negate to flip index position: -a gives lower index."""
        return Index(self.name, self.space, up=not self.up)

    def __pos__(self) -> 'Index':
        """Positive sign keeps index as-is."""
        return self

    def __repr__(self) -> str:
        sign = '' if self.up else '-'
        return f"{sign}{self.name}"

    def __eq__(self, other) -> bool:
        if not isinstance(other, Index):
            return False
        return (self.name == other.name and
                self.space is other.space and
                self.up == other.up)

    def __hash__(self) -> int:
        return hash((self.name, id(self.space), self.up))

    @property
    def is_up(self) -> bool:
        """Check if index is upper (contravariant)."""
        return self.up

    @property
    def is_down(self) -> bool:
        """Check if index is lower (covariant)."""
        return not self.up

    @property
    def dim(self) -> Optional[int]:
        """Get dimension of this index's space."""
        return self.space.dim

    @property
    def sympy_index(self) -> _TensorIndex:
        """Access underlying SymPy TensorIndex."""
        return self._sympy_index

    def raised(self) -> 'Index':
        """Return this index in upper position."""
        return Index(self.name, self.space, up=True)

    def lowered(self) -> 'Index':
        """Return this index in lower position."""
        return Index(self.name, self.space, up=False)


def indices(names: str, space: IndexSpace) -> Tuple[Index, ...]:
    """
    Create multiple indices in a space.

    Args:
        names: Space-separated index names (e.g., 'a b c')
        space: The IndexSpace for these indices

    Returns:
        Tuple of Index objects

    Example:
        >>> a, b, c = indices('a b c', spacetime)
    """
    return space.indices(names)


class AbstractTensor:
    """
    Abstract tensor with support for multiple index spaces.

    Tensors can have indices from different spaces, enabling mixed tensors
    like a spacetime vector with gauge indices: V^a_i.

    Examples:
        >>> spacetime = IndexSpace('spacetime', dim=4)
        >>> gauge = IndexSpace('gauge', dim=3)
        >>>
        >>> # Pure spacetime tensor
        >>> T = Tensor('T', spacetime, spacetime)
        >>>
        >>> # Mixed tensor with spacetime and gauge indices
        >>> A = Tensor('A', spacetime, gauge, gauge)
        >>>
        >>> # With symmetry
        >>> g = Tensor('g', spacetime, spacetime, symmetric=True)
        >>> F = Tensor('F', spacetime, spacetime, antisymmetric=True)
    """

    def __init__(
        self,
        name: str,
        *index_spaces: IndexSpace,
        symmetric: bool = False,
        antisymmetric: bool = False,
        symmetry: Optional[_TensorSymmetry] = None,
        components: Optional[Any] = None,
    ):
        """
        Create an abstract tensor.

        Args:
            name: Tensor name
            *index_spaces: IndexSpace for each index position
            symmetric: If True, fully symmetric in all indices
            antisymmetric: If True, fully antisymmetric in all indices
            symmetry: Custom TensorSymmetry (overrides symmetric/antisymmetric)
            components: Optional concrete component array
        """
        self.name = name
        self.index_spaces = tuple(index_spaces)
        self.rank = len(index_spaces)
        self._components = components

        # Determine symmetry
        if symmetry is not None:
            sym = symmetry
        elif symmetric:
            sym = _TensorSymmetry.fully_symmetric(self.rank)
        elif antisymmetric:
            sym = _TensorSymmetry.fully_symmetric(-self.rank)
        else:
            sym = _TensorSymmetry.no_symmetry(self.rank)

        self._symmetry = sym

        # Create underlying SymPy TensorHead
        sympy_types = [s.sympy_type for s in index_spaces]
        self._sympy_head = _TensorHead(name, sympy_types, sym)

    @property
    def components(self) -> Optional[Any]:
        """Get concrete components if set."""
        return self._components

    @components.setter
    def components(self, arr: Any):
        """Set concrete components."""
        self._components = arr

    @property
    def symmetry(self) -> _TensorSymmetry:
        """Get the tensor's symmetry."""
        return self._symmetry

    @property
    def sympy_head(self) -> _TensorHead:
        """Access underlying SymPy TensorHead."""
        return self._sympy_head

    def __getitem__(self, indices: Union[Index, Tuple[Index, ...]]) -> 'IndexedTensor':
        """
        Access tensor with abstract indices.

        T[a, -b] returns an IndexedTensor representing T^a_b.
        Automatic contraction via Einstein summation.
        """
        if isinstance(indices, Index):
            indices = (indices,)
        elif not isinstance(indices, tuple):
            indices = tuple(indices)

        return IndexedTensor(self, indices)

    def __call__(self, *indices: Index) -> 'IndexedTensor':
        """Alternative syntax: T(a, -b) instead of T[a, -b]."""
        return self[indices]

    def __repr__(self) -> str:
        spaces = ', '.join(s.name for s in self.index_spaces)
        return f"Tensor({self.name!r}, [{spaces}])"


# Alias for shorter name
Tensor = AbstractTensor


class IndexedTensor:
    """
    A tensor with specific indices attached.

    Created by T[a, -b] notation. Supports:
    - Automatic contraction (Einstein summation)
    - Multiplication with contraction
    - Component replacement
    - Metric operations

    Examples:
        >>> T[a, -a]  # Trace - auto-contracts
        >>> T[a, b] * S[-b, c]  # Contracts over b
        >>> expr.replace_with_arrays({T: [[1,2],[3,4]]})
    """

    def __init__(self, tensor: Tensor, indices: Tuple[Index, ...]):
        """
        Create an indexed tensor expression.

        Args:
            tensor: The underlying Tensor
            indices: Tuple of Index objects
        """
        if len(indices) != tensor.rank:
            raise ValueError(
                f"Tensor {tensor.name} has rank {tensor.rank}, "
                f"but {len(indices)} indices provided"
            )

        self.tensor = tensor
        self.indices = indices

        # Create underlying SymPy tensor expression
        sympy_indices = [idx.sympy_index for idx in indices]
        self._sympy_expr = tensor.sympy_head(*sympy_indices)

    @property
    def sympy_expr(self):
        """Access underlying SymPy tensor expression."""
        return self._sympy_expr

    @property
    def free_indices(self) -> List[Index]:
        """Get uncontracted (free) indices."""
        # Find indices that appear only once
        from collections import Counter
        name_counts = Counter(idx.name for idx in self.indices)
        return [idx for idx in self.indices if name_counts[idx.name] == 1]

    @property
    def value(self) -> Optional[Any]:
        """
        Compute value if tensor has components and indices contract.

        For auto-contracted expressions like T[a, -a] (trace).
        """
        if self.tensor.components is None:
            return None

        # Check for self-contractions
        contractions = self._find_self_contractions()
        if not contractions:
            return None

        from sympy import tensorcontraction
        result = self.tensor.components
        for i, j in sorted(contractions, reverse=True):
            result = tensorcontraction(result, (i, j))
        return result

    def _find_self_contractions(self) -> List[Tuple[int, int]]:
        """Find pairs of indices to contract within this expression."""
        contractions = []
        used = set()
        for i, idx1 in enumerate(self.indices):
            if i in used:
                continue
            for j, idx2 in enumerate(self.indices):
                if j <= i or j in used:
                    continue
                if idx1.name == idx2.name and idx1.up != idx2.up:
                    contractions.append((i, j))
                    used.update([i, j])
                    break
        return contractions

    def __repr__(self) -> str:
        idx_str = ', '.join(repr(i) for i in self.indices)
        return f"{self.tensor.name}[{idx_str}]"

    def __mul__(self, other: 'IndexedTensor') -> Any:
        """
        Multiply with automatic Einstein summation contraction.

        Repeated indices (one up, one down) are summed over.
        """
        if isinstance(other, IndexedTensor):
            return self._sympy_expr * other._sympy_expr
        return self._sympy_expr * other

    def __rmul__(self, other) -> Any:
        """Right multiplication."""
        return other * self._sympy_expr

    def __add__(self, other: 'IndexedTensor') -> Any:
        """Add tensor expressions."""
        if isinstance(other, IndexedTensor):
            return self._sympy_expr + other._sympy_expr
        return self._sympy_expr + other

    def __radd__(self, other) -> Any:
        """Right addition."""
        return other + self._sympy_expr

    def __sub__(self, other: 'IndexedTensor') -> Any:
        """Subtract tensor expressions."""
        if isinstance(other, IndexedTensor):
            return self._sympy_expr - other._sympy_expr
        return self._sympy_expr - other

    def __neg__(self) -> Any:
        """Negate tensor expression."""
        return -self._sympy_expr

    def replace_with_arrays(
        self,
        replacement_dict: dict,
        free_indices: Optional[List[Index]] = None,
    ) -> Any:
        """
        Replace tensor with concrete array components.

        Args:
            replacement_dict: Maps tensors/spaces to arrays
                {T: [[1,2],[3,4]], spacetime: diag(1,-1,-1,-1)}
            free_indices: Ordering of free indices in result

        Returns:
            Array with contracted components

        Example:
            >>> expr = T[a, -a]  # trace
            >>> expr.replace_with_arrays({T: [[1,2],[3,4]]})
            5  # 1 + 4
        """
        # Convert our types to SymPy types in the dict
        sympy_dict = {}
        for key, val in replacement_dict.items():
            if isinstance(key, Tensor):
                # Need to create a tensor with dummy indices
                dummy_indices = []
                for i, space in enumerate(key.index_spaces):
                    dummy_indices.append(_TensorIndex(f'_i{i}', space.sympy_type))
                sympy_key = key.sympy_head(*dummy_indices)
                sympy_dict[sympy_key] = val
            elif isinstance(key, IndexSpace):
                sympy_dict[key.sympy_type] = val
            else:
                sympy_dict[key] = val

        # Convert free_indices if provided
        sympy_free = None
        if free_indices:
            sympy_free = [idx.sympy_index for idx in free_indices]

        return self._sympy_expr.replace_with_arrays(sympy_dict, sympy_free)

    def contract_metric(self, metric=None) -> Any:
        """
        Contract using the metric to raise/lower indices.

        Args:
            metric: Metric to use (defaults to space's metric)

        Returns:
            Contracted expression
        """
        if metric is None:
            # Use first index's space metric
            metric = self.indices[0].space.metric
        return self._sympy_expr.contract_metric(metric)

    def canon_bp(self) -> Any:
        """Canonicalize to Butler-Portugal normal form."""
        return self._sympy_expr.canon_bp()


# Backwards compatibility alias
IndexedExpr = IndexedTensor


class TensorProduct:
    """Represents a product of indexed tensor expressions."""

    def __init__(self, left: IndexedTensor, right: IndexedTensor):
        self.left = left
        self.right = right
        self._sympy_expr = left.sympy_expr * right.sympy_expr

    @property
    def sympy_expr(self):
        return self._sympy_expr

    def __repr__(self) -> str:
        return f"({self.left} * {self.right})"


# Submanifold projections

class Submanifold:
    """
    Represents a submanifold embedded in a larger space.

    Used for projecting tensors onto subspaces, e.g., projecting
    4D spacetime tensors onto a 3D spatial hypersurface.

    Example:
        >>> spacetime = IndexSpace('spacetime', dim=4)
        >>> spatial = Submanifold('spatial', spacetime, dim=3)
        >>> # Define projection via normal vector or explicit projector
    """

    def __init__(
        self,
        name: str,
        ambient: IndexSpace,
        dim: int,
        projector: Optional[Any] = None,
        normal: Optional[Any] = None,
    ):
        """
        Create a submanifold.

        Args:
            name: Name of the submanifold
            ambient: The ambient IndexSpace
            dim: Dimension of the submanifold
            projector: Optional projection tensor h^a_b
            normal: Optional normal vector n^a (for hypersurfaces)
        """
        self.name = name
        self.ambient = ambient
        self.dim = dim
        self._projector = projector
        self._normal = normal

        # Create an IndexSpace for the submanifold indices
        self.index_space = IndexSpace(name, dim=dim, dummy_name=name[0].lower())

    @property
    def projector(self) -> Optional[Any]:
        """Get the projection tensor."""
        return self._projector

    @projector.setter
    def projector(self, h: Any):
        """Set the projection tensor."""
        self._projector = h

    def project(self, expr: IndexedTensor) -> Any:
        """
        Project a tensor expression onto this submanifold.

        Contracts each free index with the projector.
        """
        if self._projector is None:
            raise ValueError("No projector defined for this submanifold")
        # TODO: Implement full projection
        raise NotImplementedError("Submanifold projection not yet implemented")

    def __repr__(self) -> str:
        return f"Submanifold({self.name!r}, ambient={self.ambient.name}, dim={self.dim})"


class TensorRule:
    """
    A replacement rule for tensor expressions.

    Defines patterns like "g[a,-b] -> delta[a,-b] on the submanifold"
    for simplifying expressions.

    Example:
        >>> # Flat space rule: Riemann vanishes
        >>> rule = TensorRule(R[a,b,c,d], 0)
        >>>
        >>> # Metric is Kronecker delta
        >>> rule = TensorRule(g[a,-b], delta[a,-b])
    """

    def __init__(self, pattern: Any, replacement: Any, condition: Optional[Any] = None):
        """
        Create a tensor replacement rule.

        Args:
            pattern: Tensor expression pattern to match
            replacement: What to replace it with
            condition: Optional condition for when rule applies
        """
        self.pattern = pattern
        self.replacement = replacement
        self.condition = condition

    def apply(self, expr: Any) -> Any:
        """Apply this rule to an expression."""
        # Use SymPy's substitution
        if hasattr(expr, 'subs'):
            return expr.subs(self.pattern, self.replacement)
        return expr

    def __repr__(self) -> str:
        return f"TensorRule({self.pattern} -> {self.replacement})"


def raise_index(expr: IndexedTensor, index_pos: int) -> Any:
    """
    Raise an index using the metric.

    T_a -> g^{ab} T_b = T^a
    """
    idx = expr.indices[index_pos]
    if idx.is_up:
        return expr

    metric = idx.space.metric
    if metric is None:
        raise ValueError(f"No metric defined for space {idx.space.name}")

    return expr.contract_metric(metric)


def lower_index(expr: IndexedTensor, index_pos: int) -> Any:
    """
    Lower an index using the metric.

    T^a -> g_{ab} T^b = T_a
    """
    idx = expr.indices[index_pos]
    if idx.is_down:
        return expr

    metric = idx.space.metric
    if metric is None:
        raise ValueError(f"No metric defined for space {idx.space.name}")

    return expr.contract_metric(metric)


__all__ = [
    # Core classes
    'Index',
    'IndexSpace',
    'IndexType',  # backwards compat
    'Tensor',
    'AbstractTensor',  # backwards compat
    'IndexedTensor',
    'IndexedExpr',  # backwards compat
    'TensorProduct',
    'TensorSymmetry',
    # Submanifolds and rules
    'Submanifold',
    'TensorRule',
    # Factory functions
    'indices',
    # Operations
    'raise_index',
    'lower_index',
]
