"""
einsum.py - Einstein Summation and Tensor Contractions

Provides Einstein summation notation for tensor operations using
SymPy's native tensor machinery.
"""

from typing import Any, Dict, List, Tuple, Union
import sympy as sp
from sympy import (
    Array, MutableDenseNDimArray, ImmutableDenseNDimArray,
    tensorcontraction, tensorproduct, permutedims,
)
from itertools import product as iterproduct


def Einsum(subscripts: str, *operands: Array) -> Array:
    """
    Evaluate Einstein summation convention on operands.

    Uses SymPy's tensorproduct and tensorcontraction for efficient computation.

    Args:
        subscripts: Specifies the subscripts for summation.
                   Format: "ij,jk->ik" for matrix multiplication
                   Repeated indices are summed over.
        *operands: Input arrays to contract.

    Returns:
        Contracted tensor result.

    Examples:
        >>> A = Array([[1, 2], [3, 4]])
        >>> B = Array([[5, 6], [7, 8]])
        >>> Einsum("ij,jk->ik", A, B)  # Matrix multiplication
        >>> Einsum("ii->", A)          # Trace
        >>> Einsum("ij->ji", A)        # Transpose
    """
    if "->" in subscripts:
        input_str, output_str = subscripts.split("->")
    else:
        input_str = subscripts
        output_str = None

    input_subscripts = input_str.split(",")

    if len(input_subscripts) != len(operands):
        raise ValueError(
            f"Number of subscript groups ({len(input_subscripts)}) "
            f"must match number of operands ({len(operands)})"
        )

    # Build index info
    index_dims = _build_index_dims(input_subscripts, operands)

    # Determine output indices
    all_input = "".join(input_subscripts)
    if output_str is None:
        counts = {c: all_input.count(c) for c in set(all_input)}
        output_str = "".join(sorted(c for c in counts if counts[c] == 1))

    # Single operand: handle trace/transpose
    if len(operands) == 1:
        return _einsum_single(input_subscripts[0], output_str, operands[0])

    # Multi-operand: tensorproduct then contract
    return _einsum_via_product(input_subscripts, output_str, operands)


def _build_index_dims(subscripts: List[str], operands: List[Array]) -> Dict[str, int]:
    """Build mapping from index character to dimension."""
    index_dims = {}
    for subscript, operand in zip(subscripts, operands):
        shape = operand.shape
        if len(subscript) != len(shape):
            raise ValueError(f"Subscript '{subscript}' doesn't match rank {len(shape)}")
        for idx_char, dim in zip(subscript, shape):
            if idx_char in index_dims and index_dims[idx_char] != dim:
                raise ValueError(f"Dimension mismatch for '{idx_char}'")
            index_dims[idx_char] = dim
    return index_dims


def _einsum_single(input_sub: str, output_sub: str, operand: Array) -> Array:
    """Handle single-operand einsum."""
    result = operand

    # Find contraction pairs (repeated indices)
    char_pos = {}
    for pos, char in enumerate(input_sub):
        char_pos.setdefault(char, []).append(pos)

    # Contract repeated indices not in output
    contractions = [(pos[0], pos[1]) for char, pos in char_pos.items()
                    if len(pos) == 2 and char not in output_sub]

    # Apply contractions (in reverse order to preserve indices)
    for pair in sorted(contractions, reverse=True):
        result = tensorcontraction(result, pair)

    # Permute if needed
    remaining = [c for c in input_sub if input_sub.count(c) == 1 or c in output_sub]
    remaining = [c for i, c in enumerate(input_sub) if c in output_sub or (c not in output_sub and input_sub.count(c) == 1)]

    # Simpler: rebuild remaining indices after contraction
    contracted_chars = {c for c, pos in char_pos.items() if len(pos) == 2 and c not in output_sub}
    remaining = [c for c in input_sub if c not in contracted_chars]

    if output_sub and remaining != list(output_sub):
        perm = [remaining.index(c) for c in output_sub]
        result = permutedims(result, perm)

    return result


def _einsum_via_product(
    input_subscripts: List[str],
    output_sub: str,
    operands: List[Array]
) -> Array:
    """Multi-operand einsum via tensor product and contraction."""
    # Compute tensor product of all operands
    product = operands[0]
    for op in operands[1:]:
        product = tensorproduct(product, op)

    # Build combined index string
    combined = "".join(input_subscripts)

    # Find all contraction pairs
    char_positions = {}
    for pos, char in enumerate(combined):
        char_positions.setdefault(char, []).append(pos)

    # Contract indices that appear twice and aren't in output
    contractions = []
    for char, positions in char_positions.items():
        if len(positions) == 2 and char not in output_sub:
            contractions.append(tuple(positions))

    # Apply contractions (SymPy can handle multiple at once)
    if contractions:
        product = tensorcontraction(product, *contractions)

    # Determine remaining indices and permute if needed
    contracted_chars = {c for c, pos in char_positions.items()
                        if len(pos) == 2 and c not in output_sub}
    remaining = [c for c in combined if c not in contracted_chars]

    # Remove duplicates while preserving order (for indices that appeared twice)
    seen = set()
    remaining_unique = []
    for c in remaining:
        if c not in seen:
            remaining_unique.append(c)
            seen.add(c)

    if output_sub and remaining_unique != list(output_sub):
        perm = [remaining_unique.index(c) for c in output_sub]
        product = permutedims(product, perm)

    return product


def Contract(tensor: Array, *pairs: Tuple[int, int]) -> Array:
    """
    Contract tensor over specified index pairs.

    Args:
        tensor: Input tensor
        *pairs: Tuples of (i, j) specifying which axes to contract

    Returns:
        Contracted tensor

    Examples:
        >>> T = Array([[[1,2],[3,4]], [[5,6],[7,8]]])
        >>> Contract(T, (0, 1))  # Contract first two indices
    """
    return tensorcontraction(tensor, *pairs)


def TensorProduct(*tensors: Array) -> Array:
    """
    Compute tensor product of multiple tensors.

    Args:
        *tensors: Input tensors

    Returns:
        Tensor product
    """
    if len(tensors) == 0:
        raise ValueError("Need at least one tensor")
    if len(tensors) == 1:
        return tensors[0]

    result = tensors[0]
    for t in tensors[1:]:
        result = tensorproduct(result, t)
    return result


def Trace(tensor: Array, idx1: int = 0, idx2: int = 1) -> Array:
    """
    Compute trace of a tensor over specified indices.

    Args:
        tensor: Input tensor
        idx1, idx2: Indices to trace over (default: 0, 1)

    Returns:
        Tensor with traced indices removed
    """
    return tensorcontraction(tensor, (idx1, idx2))


def OuterProduct(a: Array, b: Array) -> Array:
    """Compute outer product of two arrays."""
    return tensorproduct(a, b)


def InnerProduct(a: Array, b: Array) -> sp.Expr:
    """
    Compute inner product (full contraction) of two arrays.

    Arrays must have the same shape.
    """
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")

    product = tensorproduct(a, b)
    # Contract all pairs of corresponding indices
    pairs = [(i, len(a.shape) + i) for i in range(len(a.shape))]
    return tensorcontraction(product, *pairs)


__all__ = [
    'Einsum',
    'Contract',
    'TensorProduct',
    'OuterProduct',
    'InnerProduct',
    'Trace',
]
