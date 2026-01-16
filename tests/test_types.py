"""
Tests for Custom Types / Mathematical Objects (Phase 16).

Tests DefineType, CustomType, and pre-built types like Quaternion and Vector3D.
"""

import pytest
from sympy import Symbol, Integer, sqrt, Rational

from derive import (
    DefineType, CustomType,
    ComplexNumber, Quaternion, Vector3D,
)


# Symbols for testing
a, b, c = Symbol('a'), Symbol('b'), Symbol('c')


class TestDefineType:
    """Tests for DefineType function."""

    def test_create_simple_type(self):
        """Create a simple custom type."""
        Point = DefineType('Point', ['x', 'y'])
        p = Point(3, 4)
        assert p.x == 3
        assert p.y == 4

    def test_type_name(self):
        """Type should know its name."""
        Point = DefineType('Point', ['x', 'y'])
        assert Point.type_name() == 'Point'

    def test_fields_list(self):
        """Type should list its fields."""
        Point = DefineType('Point', ['x', 'y'])
        assert Point.fields() == ['x', 'y']

    def test_kwargs_init(self):
        """Initialize with keyword arguments."""
        Point = DefineType('Point', ['x', 'y'])
        p = Point(x=5, y=10)
        assert p.x == 5
        assert p.y == 10

    def test_mixed_init(self):
        """Initialize with mixed args and kwargs."""
        Point = DefineType('Point', ['x', 'y', 'z'])
        p = Point(1, 2, z=3)
        assert p.x == 1
        assert p.y == 2
        assert p.z == 3

    def test_default_values(self):
        """Missing fields default to 0."""
        Point = DefineType('Point', ['x', 'y', 'z'])
        p = Point(1)
        assert p.x == 1
        assert p.y == 0
        assert p.z == 0

    def test_as_dict(self):
        """Convert to dictionary."""
        Point = DefineType('Point', ['x', 'y'])
        p = Point(3, 4)
        d = p.as_dict()
        assert d == {'x': 3, 'y': 4}

    def test_as_tuple(self):
        """Convert to tuple."""
        Point = DefineType('Point', ['x', 'y'])
        p = Point(3, 4)
        t = p.as_tuple()
        assert t == (3, 4)


class TestDefineOp:
    """Tests for defining operations on custom types."""

    def test_define_addition(self):
        """Define addition operation."""
        Point = DefineType('Point', ['x', 'y'])
        Point.define_op('__add__', lambda a, b: Point(a.x + b.x, a.y + b.y))

        p1 = Point(1, 2)
        p2 = Point(3, 4)
        result = p1 + p2

        assert result.x == 4
        assert result.y == 6

    def test_define_subtraction(self):
        """Define subtraction operation."""
        Point = DefineType('Point', ['x', 'y'])
        Point.define_op('__sub__', lambda a, b: Point(a.x - b.x, a.y - b.y))

        p1 = Point(5, 7)
        p2 = Point(2, 3)
        result = p1 - p2

        assert result.x == 3
        assert result.y == 4

    def test_define_scalar_mul(self):
        """Define scalar multiplication."""
        Point = DefineType('Point', ['x', 'y'])
        Point.define_op('__mul__', lambda p, s: Point(p.x * s, p.y * s))

        p = Point(2, 3)
        result = p * 5

        assert result.x == 10
        assert result.y == 15

    def test_define_custom_method(self):
        """Define a custom method."""
        Point = DefineType('Point', ['x', 'y'])
        Point.define_method('magnitude', lambda self: sqrt(self.x**2 + self.y**2))

        p = Point(3, 4)
        assert p.magnitude() == 5


class TestTypeEquality:
    """Tests for type equality."""

    def test_equal_instances(self):
        """Equal instances should be equal."""
        Point = DefineType('Point', ['x', 'y'])
        p1 = Point(3, 4)
        p2 = Point(3, 4)
        assert p1 == p2

    def test_unequal_instances(self):
        """Unequal instances should not be equal."""
        Point = DefineType('Point', ['x', 'y'])
        p1 = Point(3, 4)
        p2 = Point(5, 6)
        assert p1 != p2

    def test_different_types(self):
        """Different types should not be equal."""
        Point = DefineType('Point', ['x', 'y'])
        Vector = DefineType('Vector', ['x', 'y'])
        p = Point(3, 4)
        v = Vector(3, 4)
        assert p != v


class TestComplexNumber:
    """Tests for built-in ComplexNumber type."""

    def test_create_complex(self):
        """Create complex number."""
        z = ComplexNumber(3, 4)
        assert z.real == 3
        assert z.imag == 4

    def test_complex_addition(self):
        """Add complex numbers."""
        z1 = ComplexNumber(1, 2)
        z2 = ComplexNumber(3, 4)
        result = z1 + z2
        assert result.real == 4
        assert result.imag == 6

    def test_complex_subtraction(self):
        """Subtract complex numbers."""
        z1 = ComplexNumber(5, 7)
        z2 = ComplexNumber(2, 3)
        result = z1 - z2
        assert result.real == 3
        assert result.imag == 4

    def test_complex_multiplication(self):
        """Multiply complex numbers."""
        z1 = ComplexNumber(1, 2)
        z2 = ComplexNumber(3, 4)
        # (1 + 2i)(3 + 4i) = 3 + 4i + 6i + 8i^2 = 3 + 10i - 8 = -5 + 10i
        result = z1 * z2
        assert result.real == -5
        assert result.imag == 10

    def test_complex_abs(self):
        """Absolute value of complex number."""
        z = ComplexNumber(3, 4)
        assert abs(z) == 5

    def test_complex_conjugate(self):
        """Conjugate of complex number."""
        z = ComplexNumber(3, 4)
        conj = z.conjugate()
        assert conj.real == 3
        assert conj.imag == -4


class TestQuaternion:
    """Tests for built-in Quaternion type."""

    def test_create_quaternion(self):
        """Create quaternion."""
        q = Quaternion(1, 2, 3, 4)
        assert q.r == 1
        assert q.i == 2
        assert q.j == 3
        assert q.k == 4

    def test_quaternion_str(self):
        """Test quaternion string representation."""
        q = Quaternion(1, 2, 3, 4)
        s = str(q)
        assert '1' in s
        assert '2' in s
        assert '3' in s
        assert '4' in s

    def test_quaternion_repr(self):
        """Test quaternion repr."""
        q = Quaternion(1, 0, 0, 0)
        r = repr(q)
        assert '1' in r

    def test_quaternion_addition(self):
        """Add quaternions."""
        q1 = Quaternion(1, 2, 3, 4)
        q2 = Quaternion(5, 6, 7, 8)
        result = q1 + q2
        assert result.r == 6
        assert result.i == 8
        assert result.j == 10
        assert result.k == 12

    def test_quaternion_subtraction(self):
        """Subtract quaternions."""
        q1 = Quaternion(5, 6, 7, 8)
        q2 = Quaternion(1, 2, 3, 4)
        result = q1 - q2
        assert result.r == 4
        assert result.i == 4
        assert result.j == 4
        assert result.k == 4

    def test_quaternion_scalar_mul(self):
        """Scalar multiplication."""
        q = Quaternion(1, 2, 3, 4)
        result = q * 2
        assert result.r == 2
        assert result.i == 4
        assert result.j == 6
        assert result.k == 8

    def test_quaternion_multiplication(self):
        """Quaternion multiplication (non-commutative)."""
        # i * j = k
        i = Quaternion(0, 1, 0, 0)
        j = Quaternion(0, 0, 1, 0)
        result = i * j
        assert result.r == 0
        assert result.i == 0
        assert result.j == 0
        assert result.k == 1

    def test_quaternion_conjugate(self):
        """Quaternion conjugate."""
        q = Quaternion(1, 2, 3, 4)
        conj = q.conjugate()
        assert conj.r == 1
        assert conj.i == -2
        assert conj.j == -3
        assert conj.k == -4

    def test_quaternion_abs(self):
        """Quaternion magnitude."""
        q = Quaternion(1, 0, 0, 0)
        assert abs(q) == 1

        q2 = Quaternion(1, 1, 1, 1)
        assert abs(q2) == 2


class TestVector3D:
    """Tests for built-in Vector3D type."""

    def test_create_vector(self):
        """Create 3D vector."""
        v = Vector3D(1, 2, 3)
        assert v.x == 1
        assert v.y == 2
        assert v.z == 3

    def test_vector_addition(self):
        """Add vectors."""
        v1 = Vector3D(1, 2, 3)
        v2 = Vector3D(4, 5, 6)
        result = v1 + v2
        assert result.x == 5
        assert result.y == 7
        assert result.z == 9

    def test_vector_subtraction(self):
        """Subtract vectors."""
        v1 = Vector3D(4, 5, 6)
        v2 = Vector3D(1, 2, 3)
        result = v1 - v2
        assert result.x == 3
        assert result.y == 3
        assert result.z == 3

    def test_vector_scalar_mul(self):
        """Scalar multiplication."""
        v = Vector3D(1, 2, 3)
        result = v * 2
        assert result.x == 2
        assert result.y == 4
        assert result.z == 6

    def test_vector_dot_product(self):
        """Dot product."""
        v1 = Vector3D(1, 2, 3)
        v2 = Vector3D(4, 5, 6)
        # 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        assert v1.dot(v2) == 32

    def test_vector_cross_product(self):
        """Cross product."""
        # i x j = k
        i = Vector3D(1, 0, 0)
        j = Vector3D(0, 1, 0)
        k = i.cross(j)
        assert k.x == 0
        assert k.y == 0
        assert k.z == 1

    def test_vector_magnitude(self):
        """Vector magnitude."""
        v = Vector3D(3, 4, 0)
        assert abs(v) == 5


class TestSymbolicValues:
    """Tests with symbolic values."""

    def test_symbolic_point(self):
        """Create point with symbolic values."""
        Point = DefineType('Point', ['x', 'y'])
        p = Point(a, b)
        assert p.x == a
        assert p.y == b

    def test_symbolic_vector_ops(self):
        """Vector operations with symbols."""
        v1 = Vector3D(a, b, c)
        v2 = Vector3D(1, 2, 3)
        result = v1 + v2
        assert result.x == a + 1
        assert result.y == b + 2
        assert result.z == c + 3

    def test_symbolic_quaternion(self):
        """Quaternion with symbolic values."""
        q = Quaternion(a, b, c, 1)
        conj = q.conjugate()
        assert conj.r == a
        assert conj.i == -b
        assert conj.j == -c
        assert conj.k == -1
