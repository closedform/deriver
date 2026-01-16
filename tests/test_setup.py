"""Test that the project is set up correctly."""

def test_import_derive():
    """Test that derive package can be imported."""
    import derive
    assert derive.__version__ == "0.1.0"


def test_sympy_available():
    """Test that sympy is available."""
    import sympy
    x = sympy.Symbol('x')
    assert sympy.diff(x**2, x) == 2*x
