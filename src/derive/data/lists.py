"""
lists.py - List Operations.

Provides list manipulation functions.
"""

from typing import Any, Callable, List as ListType, Optional


def Table(expr: Any, *iterators) -> list:
    """
    Table function.

    Table[f, {i, n}] - list of f for i from 1 to n
    Table[f, {i, a, b}] - list of f for i from a to b

    Args:
        expr: Expression to evaluate (can have free symbols to substitute)
        *iterators: Tuples specifying iteration

    Returns:
        List of evaluated expressions

    Examples:
        >>> i = Symbol('i')
        >>> Table(i**2, (i, 1, 5))
        [1, 4, 9, 16, 25]
    """
    if len(iterators) == 0:
        return expr

    iterator = iterators[0]
    remaining = iterators[1:]

    if isinstance(iterator, (list, tuple)):
        if len(iterator) == 2:
            var, n = iterator
            start, end = 1, int(n)
        elif len(iterator) == 3:
            var, start, end = iterator
            start, end = int(start), int(end)
        else:
            raise ValueError("Invalid iterator format")

        result = []
        for val in range(start, end + 1):
            sub_expr = expr.subs(var, val) if hasattr(expr, 'subs') else expr
            if remaining:
                sub_expr = Table(sub_expr, *remaining)
            result.append(sub_expr)
        return result

    raise ValueError("Table requires format: Table(expr, (var, start, end))")


def Range(*args) -> list:
    """
    Range.

    Range[n] - list from 1 to n
    Range[a, b] - list from a to b
    Range[a, b, step] - list from a to b with step

    Args:
        *args: 1-3 arguments specifying range

    Returns:
        List of values

    Examples:
        >>> Range(5)
        [1, 2, 3, 4, 5]
        >>> Range(2, 6)
        [2, 3, 4, 5, 6]
    """
    if len(args) == 1:
        return list(range(1, int(args[0]) + 1))
    elif len(args) == 2:
        return list(range(int(args[0]), int(args[1]) + 1))
    elif len(args) == 3:
        a, b, step = args
        result = []
        val = a
        while val <= b:
            result.append(val)
            val += step
        return result
    raise ValueError("Range requires 1-3 arguments")


def Map(f: Callable, lst: list) -> list:
    """
    Apply function f to each element of lst.

    Args:
        f: Function to apply
        lst: List to map over

    Returns:
        List with f applied to each element
    """
    return [f(x) for x in lst]


def Select(lst: list, cond: Callable) -> list:
    """
    Select elements satisfying condition.

    Args:
        lst: List to filter
        cond: Condition function

    Returns:
        Filtered list
    """
    return [x for x in lst if cond(x)]


def Sort(lst: list, key: Optional[Callable] = None) -> list:
    """
    Sort a list.

    Args:
        lst: List to sort
        key: Optional key function

    Returns:
        Sorted list
    """
    return sorted(lst, key=key)


def Total(lst: list) -> Any:
    """Sum of list elements."""
    return sum(lst)


def Length(lst: list) -> int:
    """Length of list."""
    return len(lst)


def First(lst: list) -> Any:
    """First element of list."""
    return lst[0]


def Last(lst: list) -> Any:
    """Last element of list."""
    return lst[-1]


def Take(lst: list, n: int) -> list:
    """
    Take first n elements (or last n if negative).

    Args:
        lst: List
        n: Number of elements

    Returns:
        Sublist
    """
    if n >= 0:
        return lst[:n]
    return lst[n:]


def Drop(lst: list, n: int) -> list:
    """
    Drop first n elements (or last n if negative).

    Args:
        lst: List
        n: Number of elements

    Returns:
        Sublist
    """
    if n >= 0:
        return lst[n:]
    return lst[:n]


def Append(lst: list, elem: Any) -> list:
    """
    Append element to list.

    Args:
        lst: List
        elem: Element to append

    Returns:
        New list with element appended
    """
    return lst + [elem]


def Prepend(lst: list, elem: Any) -> list:
    """
    Prepend element to list.

    Args:
        lst: List
        elem: Element to prepend

    Returns:
        New list with element prepended
    """
    return [elem] + lst


def Join(*lists) -> list:
    """
    Join multiple lists.

    Args:
        *lists: Lists to join

    Returns:
        Concatenated list
    """
    result = []
    for lst in lists:
        result.extend(lst)
    return result


def Flatten(lst: list, depth: Optional[int] = None) -> list:
    """
    Flatten nested list.

    Args:
        lst: Nested list
        depth: Maximum depth to flatten (None for all)

    Returns:
        Flattened list
    """
    def _flatten(lst, d):
        result = []
        for item in lst:
            if isinstance(item, list) and (d is None or d > 0):
                result.extend(_flatten(item, None if d is None else d - 1))
            else:
                result.append(item)
        return result
    return _flatten(lst, depth)


def Partition(lst: list, n: int) -> ListType[list]:
    """
    Partition list into sublists of length n.

    Args:
        lst: List to partition
        n: Partition size

    Returns:
        List of sublists
    """
    return [lst[i:i+n] for i in range(0, len(lst), n)]


__all__ = [
    'Table', 'Range', 'Map', 'Select', 'Sort', 'Total', 'Length',
    'First', 'Last', 'Take', 'Drop', 'Append', 'Prepend', 'Join',
    'Flatten', 'Partition',
]
