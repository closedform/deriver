"""
builtin.py - Built-in Custom Types

Provides pre-built types: ComplexNumber, Quaternion, Vector3D.
"""

from typing import Type
import sympy as sp

from derive.types.base import CustomType, DefineType


def create_complex_type() -> Type[CustomType]:
    """Create a Complex number type."""
    def display(c):
        if c.imag == 0:
            return str(c.real)
        elif c.real == 0:
            return f"{c.imag}*I"
        else:
            return f"{c.real} + {c.imag}*I"

    def latex_repr(c):
        if c.imag == 0:
            return str(c.real)
        elif c.real == 0:
            return f"{c.imag}i"
        else:
            return f"{c.real} + {c.imag}i"

    Complex = DefineType('Complex', ['real', 'imag'], display=display, latex=latex_repr)

    # Define operations
    Complex.define_op('__add__', lambda a, b:
        Complex(a.real + b.real, a.imag + b.imag) if isinstance(b, Complex)
        else Complex(a.real + b, a.imag))

    Complex.define_op('__sub__', lambda a, b:
        Complex(a.real - b.real, a.imag - b.imag) if isinstance(b, Complex)
        else Complex(a.real - b, a.imag))

    Complex.define_op('__mul__', lambda a, b:
        Complex(a.real * b.real - a.imag * b.imag,
                a.real * b.imag + a.imag * b.real) if isinstance(b, Complex)
        else Complex(a.real * b, a.imag * b))

    Complex.define_op('__neg__', lambda a: Complex(-a.real, -a.imag))

    Complex.define_op('__abs__', lambda a: sp.sqrt(a.real**2 + a.imag**2))

    # Conjugate method
    Complex.define_method('conjugate', lambda self: Complex(self.real, -self.imag))

    return Complex


def create_quaternion_type() -> Type[CustomType]:
    """Create a Quaternion type."""
    def display(q):
        parts = []
        if q.r != 0:
            parts.append(str(q.r))
        if q.i != 0:
            parts.append(f"{q.i}*i")
        if q.j != 0:
            parts.append(f"{q.j}*j")
        if q.k != 0:
            parts.append(f"{q.k}*k")
        return " + ".join(parts) if parts else "0"

    Quaternion = DefineType('Quaternion', ['r', 'i', 'j', 'k'], display=display)

    # Addition
    Quaternion.define_op('__add__', lambda a, b:
        Quaternion(a.r + b.r, a.i + b.i, a.j + b.j, a.k + b.k) if isinstance(b, Quaternion)
        else Quaternion(a.r + b, a.i, a.j, a.k))

    # Subtraction
    Quaternion.define_op('__sub__', lambda a, b:
        Quaternion(a.r - b.r, a.i - b.i, a.j - b.j, a.k - b.k) if isinstance(b, Quaternion)
        else Quaternion(a.r - b, a.i, a.j, a.k))

    # Multiplication (quaternion product)
    def quat_mul(a, b):
        if isinstance(b, Quaternion):
            return Quaternion(
                a.r*b.r - a.i*b.i - a.j*b.j - a.k*b.k,
                a.r*b.i + a.i*b.r + a.j*b.k - a.k*b.j,
                a.r*b.j - a.i*b.k + a.j*b.r + a.k*b.i,
                a.r*b.k + a.i*b.j - a.j*b.i + a.k*b.r
            )
        else:
            return Quaternion(a.r * b, a.i * b, a.j * b, a.k * b)

    Quaternion.define_op('__mul__', quat_mul)

    # Right multiplication by scalar
    Quaternion.define_op('__rmul__', lambda s, q:
        Quaternion(q.r * s, q.i * s, q.j * s, q.k * s))

    # Negation
    Quaternion.define_op('__neg__', lambda q: Quaternion(-q.r, -q.i, -q.j, -q.k))

    # Norm/Abs
    Quaternion.define_op('__abs__', lambda q:
        sp.sqrt(q.r**2 + q.i**2 + q.j**2 + q.k**2))

    # Conjugate
    Quaternion.define_method('conjugate', lambda self:
        Quaternion(self.r, -self.i, -self.j, -self.k))

    # Inverse
    def quat_inverse(q):
        norm_sq = q.r**2 + q.i**2 + q.j**2 + q.k**2
        return Quaternion(q.r/norm_sq, -q.i/norm_sq, -q.j/norm_sq, -q.k/norm_sq)

    Quaternion.define_method('inverse', quat_inverse)

    return Quaternion


def create_vector3d_type() -> Type[CustomType]:
    """Create a 3D Vector type."""
    def display(v):
        return f"Vector3D({v.x}, {v.y}, {v.z})"

    def latex_repr(v):
        return f"\\begin{{pmatrix}} {v.x} \\\\ {v.y} \\\\ {v.z} \\end{{pmatrix}}"

    Vector3D = DefineType('Vector3D', ['x', 'y', 'z'], display=display, latex=latex_repr)

    # Addition
    Vector3D.define_op('__add__', lambda a, b:
        Vector3D(a.x + b.x, a.y + b.y, a.z + b.z))

    # Subtraction
    Vector3D.define_op('__sub__', lambda a, b:
        Vector3D(a.x - b.x, a.y - b.y, a.z - b.z))

    # Scalar multiplication
    Vector3D.define_op('__mul__', lambda v, s:
        Vector3D(v.x * s, v.y * s, v.z * s))

    Vector3D.define_op('__rmul__', lambda s, v:
        Vector3D(v.x * s, v.y * s, v.z * s))

    # Negation
    Vector3D.define_op('__neg__', lambda v: Vector3D(-v.x, -v.y, -v.z))

    # Magnitude
    Vector3D.define_op('__abs__', lambda v: sp.sqrt(v.x**2 + v.y**2 + v.z**2))

    # Dot product
    Vector3D.define_method('dot', lambda self, other:
        self.x * other.x + self.y * other.y + self.z * other.z)

    # Cross product
    Vector3D.define_method('cross', lambda self, other:
        Vector3D(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        ))

    # Normalize
    Vector3D.define_method('normalize', lambda self:
        Vector3D(self.x / abs(self), self.y / abs(self), self.z / abs(self)))

    return Vector3D


# Create standard types
ComplexNumber = create_complex_type()
Quaternion = create_quaternion_type()
Vector3D = create_vector3d_type()


__all__ = [
    'ComplexNumber',
    'Quaternion',
    'Vector3D',
]
