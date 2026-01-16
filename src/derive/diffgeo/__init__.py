"""
diffgeo - Differential Geometry Module

This module provides differential geometry functionality:
- Metric tensors for Riemannian/pseudo-Riemannian manifolds
- Coordinate systems and transformations
- Christoffel symbols (with symmetry optimization)
- Riemann, Ricci, and Einstein tensors
- Covariant derivatives
- Tensor symmetry utilities
- Abstract index notation (xAct-style)
"""

from derive.diffgeo.metrics import (
    Metric, Tensor, CovariantDerivative,
    minkowski_metric, schwarzschild_metric, flrw_metric, spherical_metric_3d,
)
from derive.diffgeo.transforms import (
    CoordinateTransformation,
    cartesian_to_spherical_3d, cartesian_to_cylindrical,
)
from derive.diffgeo.symmetry import (
    Symmetry, SymmetricMatrix, SymmetricChristoffel,
    symmetric_index_pairs, symmetric_christoffel_indices,
    LeviCivita, levi_civita_tensor,
    fill_symmetric_tensor, fill_antisymmetric_tensor, fill_with_symmetries,
)
from derive.diffgeo.indices import (
    Index, IndexType, IndexSpace, IndexedExpr, IndexedTensor,
    AbstractTensor, TensorProduct, TensorSymmetry,
    Submanifold, TensorRule,
    indices, raise_index, lower_index,
)
from derive.diffgeo.einsum import (
    Einsum, Contract, TensorProduct as TensorProd,
    OuterProduct, InnerProduct, Trace,
)

__all__ = [
    # Core classes
    'Metric', 'Tensor', 'CoordinateTransformation',
    # Abstract index notation
    'Index', 'IndexType', 'IndexSpace', 'IndexedExpr', 'IndexedTensor',
    'AbstractTensor', 'TensorProduct', 'TensorSymmetry',
    'Submanifold', 'TensorRule',
    'indices', 'raise_index', 'lower_index',
    # Operations
    'CovariantDerivative',
    # Predefined metrics
    'minkowski_metric', 'schwarzschild_metric', 'flrw_metric', 'spherical_metric_3d',
    # Predefined transformations
    'cartesian_to_spherical_3d', 'cartesian_to_cylindrical',
    # Symmetry utilities
    'Symmetry', 'SymmetricMatrix', 'SymmetricChristoffel',
    'symmetric_index_pairs', 'symmetric_christoffel_indices',
    'LeviCivita', 'levi_civita_tensor',
    'fill_symmetric_tensor', 'fill_antisymmetric_tensor', 'fill_with_symmetries',
    # Einstein summation and tensor operations
    'Einsum', 'Contract', 'TensorProd',
    'OuterProduct', 'InnerProduct', 'Trace',
]
