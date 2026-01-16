"""
base.py - Base Classes for Custom Types

Provides CustomType base class and DefineType factory function
for creating new mathematical types.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Type
import sympy as sp
from sympy import Symbol, sympify, latex


class CustomType:
    """
    Base class for custom mathematical types.

    CustomType instances represent mathematical objects with named fields
    and can have custom operations defined on them.
    """

    _type_name: str = "CustomType"
    _fields: List[str] = []
    _operations: Dict[str, Callable] = {}
    _display_func: Optional[Callable] = None
    _latex_func: Optional[Callable] = None

    def __init__(self, *args, **kwargs):
        """
        Initialize a custom type instance.

        Args:
            *args: Positional arguments mapped to fields in order
            **kwargs: Named arguments mapped to fields by name
        """
        # Set field values from args
        for i, arg in enumerate(args):
            if i < len(self._fields):
                setattr(self, self._fields[i], sympify(arg))

        # Set field values from kwargs
        for key, value in kwargs.items():
            if key in self._fields:
                setattr(self, key, sympify(value))

        # Set defaults for missing fields
        for field in self._fields:
            if not hasattr(self, field):
                setattr(self, field, sympify(0))

    def __repr__(self) -> str:
        # Access through class to avoid descriptor protocol (functions become bound methods)
        display_func = type(self).__dict__.get('_display_func')
        if display_func:
            return display_func(self)
        field_vals = [f"{f}={getattr(self, f)}" for f in self._fields]
        return f"{self._type_name}({', '.join(field_vals)})"

    def __str__(self) -> str:
        return self.__repr__()

    def _repr_latex_(self) -> str:
        """LaTeX representation for Jupyter."""
        # Access through class to avoid descriptor protocol
        latex_func = type(self).__dict__.get('_latex_func')
        if latex_func:
            return f"${latex_func(self)}$"
        return f"${self.__repr__()}$"

    @classmethod
    def type_name(cls) -> str:
        """Get the type name."""
        return cls._type_name

    @classmethod
    def fields(cls) -> List[str]:
        """Get the field names."""
        return cls._fields.copy()

    def as_dict(self) -> Dict[str, Any]:
        """Get field values as dictionary."""
        return {f: getattr(self, f) for f in self._fields}

    def as_tuple(self) -> Tuple[Any, ...]:
        """Get field values as tuple."""
        return tuple(getattr(self, f) for f in self._fields)

    # Arithmetic operations - can be overridden by define_op
    def __add__(self, other):
        if '__add__' in self._operations:
            return self._operations['__add__'](self, other)
        raise NotImplementedError(f"Addition not defined for {self._type_name}")

    def __radd__(self, other):
        if '__radd__' in self._operations:
            return self._operations['__radd__'](other, self)
        if '__add__' in self._operations:
            return self._operations['__add__'](self, other)
        raise NotImplementedError(f"Addition not defined for {self._type_name}")

    def __sub__(self, other):
        if '__sub__' in self._operations:
            return self._operations['__sub__'](self, other)
        raise NotImplementedError(f"Subtraction not defined for {self._type_name}")

    def __mul__(self, other):
        if '__mul__' in self._operations:
            return self._operations['__mul__'](self, other)
        raise NotImplementedError(f"Multiplication not defined for {self._type_name}")

    def __rmul__(self, other):
        if '__rmul__' in self._operations:
            return self._operations['__rmul__'](other, self)
        raise NotImplementedError(f"Right multiplication not defined for {self._type_name}")

    def __truediv__(self, other):
        if '__truediv__' in self._operations:
            return self._operations['__truediv__'](self, other)
        raise NotImplementedError(f"Division not defined for {self._type_name}")

    def __neg__(self):
        if '__neg__' in self._operations:
            return self._operations['__neg__'](self)
        raise NotImplementedError(f"Negation not defined for {self._type_name}")

    def __pos__(self):
        return self

    def __abs__(self):
        if '__abs__' in self._operations:
            return self._operations['__abs__'](self)
        raise NotImplementedError(f"Abs not defined for {self._type_name}")

    def __pow__(self, other):
        if '__pow__' in self._operations:
            return self._operations['__pow__'](self, other)
        raise NotImplementedError(f"Power not defined for {self._type_name}")

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        return all(getattr(self, f) == getattr(other, f) for f in self._fields)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self._type_name, self.as_tuple()))


def DefineType(
    name: str,
    fields: List[str],
    display: Optional[Callable] = None,
    latex: Optional[Callable] = None,
    parent: Optional[Type] = None
) -> Type[CustomType]:
    """
    Create a new custom mathematical type.

    Args:
        name: Name of the type
        fields: List of field names
        display: Optional display function (instance) -> str
        latex: Optional LaTeX function (instance) -> str
        parent: Optional parent type to inherit from

    Returns:
        A new type class

    Examples:
        >>> Complex = DefineType('Complex', ['real', 'imag'])
        >>> z = Complex(3, 4)
        >>> z.real
        3
    """
    base = parent if parent else CustomType

    # Create new type class
    new_type = type(name, (base,), {
        '_type_name': name,
        '_fields': fields.copy(),
        '_operations': {},
        '_display_func': display,
        '_latex_func': latex,
    })

    # Add define_op class method
    def define_op(cls, op_name: str, func: Callable) -> Type:
        """
        Define an operation on this type.

        Args:
            op_name: Operation name (e.g., '__add__', '__mul__')
            func: Function (self, other) -> result

        Returns:
            The type class (for chaining)
        """
        cls._operations[op_name] = func
        return cls

    new_type.define_op = classmethod(lambda cls, *args, **kwargs: define_op(cls, *args, **kwargs))

    # Add define_method for custom methods
    def define_method(cls, method_name: str, func: Callable) -> Type:
        """
        Add a custom method to this type.

        Args:
            method_name: Name of the method
            func: Function (self, ...) -> result

        Returns:
            The type class (for chaining)
        """
        setattr(cls, method_name, func)
        return cls

    new_type.define_method = classmethod(lambda cls, *args, **kwargs: define_method(cls, *args, **kwargs))

    return new_type


__all__ = [
    'CustomType',
    'DefineType',
]
