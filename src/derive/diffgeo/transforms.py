"""
transforms.py - Coordinate Transformations

Provides coordinate transformation functionality for converting
between coordinate systems and transforming metric tensors.
"""

import sympy as sp
from sympy import Symbol, symbols, Matrix, sin, cos
from typing import List, Dict

from derive.diffgeo.metrics import Metric

# Use derive's own APIs for self-consistency
from derive.calculus import D
from derive.algebra import Simplify


class CoordinateTransformation:
    """
    Represents a coordinate transformation between two coordinate systems.
    """

    def __init__(self, old_coords: List[Symbol], new_coords: List[Symbol],
                 transform_eqs: Dict[Symbol, sp.Expr]):
        """
        Initialize a coordinate transformation.

        Args:
            old_coords: Original coordinate symbols
            new_coords: New coordinate symbols
            transform_eqs: Dictionary mapping old coords to expressions in new coords
                          e.g., {x: r*cos(theta), y: r*sin(theta)}
        """
        self.old_coords = list(old_coords)
        self.new_coords = list(new_coords)
        self.transform = transform_eqs

        # Compute Jacobian
        self._jacobian = None
        self._inverse_jacobian = None

    @property
    def jacobian(self) -> Matrix:
        """
        Compute the Jacobian matrix ∂x^i/∂x'^j.
        """
        if self._jacobian is not None:
            return self._jacobian

        n = len(self.old_coords)
        J = Matrix.zeros(n, n)

        for i, old in enumerate(self.old_coords):
            expr = self.transform.get(old, old)
            for j, new in enumerate(self.new_coords):
                J[i, j] = D(expr, new)

        self._jacobian = J
        return self._jacobian

    @property
    def jacobian_determinant(self) -> sp.Expr:
        """Get the Jacobian determinant."""
        return self.jacobian.det()

    def transform_metric(self, metric: Metric) -> Metric:
        """
        Transform a metric to new coordinates.

        g'_{μν} = (∂x^ρ/∂x'^μ)(∂x^σ/∂x'^ν) g_{ρσ}
        """
        J = self.jacobian
        n = metric.dim

        # New metric components
        g_new = Matrix.zeros(n, n)
        for mu in range(n):
            for nu in range(n):
                val = 0
                for rho in range(n):
                    for sigma in range(n):
                        # Substitute old coords with transform expressions
                        g_comp = metric.g[rho, sigma]
                        for old, expr in self.transform.items():
                            g_comp = g_comp.subs(old, expr)
                        val += J[rho, mu] * J[sigma, nu] * g_comp
                g_new[mu, nu] = Simplify(val)

        return Metric(self.new_coords, g_new)


def cartesian_to_spherical_3d() -> CoordinateTransformation:
    """
    Create transformation from Cartesian (x, y, z) to spherical (r, θ, φ).
    """
    x, y, z = symbols('x y z')
    r, theta, phi = symbols('r theta phi')

    transform = {
        x: r * sin(theta) * cos(phi),
        y: r * sin(theta) * sin(phi),
        z: r * cos(theta)
    }

    return CoordinateTransformation([x, y, z], [r, theta, phi], transform)


def cartesian_to_cylindrical() -> CoordinateTransformation:
    """
    Create transformation from Cartesian (x, y, z) to cylindrical (ρ, φ, z).
    """
    x, y, z_cart = symbols('x y z')
    rho, phi, z_cyl = symbols('rho phi z')

    transform = {
        x: rho * cos(phi),
        y: rho * sin(phi),
        z_cart: z_cyl
    }

    return CoordinateTransformation([x, y, z_cart], [rho, phi, z_cyl], transform)


__all__ = [
    'CoordinateTransformation',
    'cartesian_to_spherical_3d', 'cartesian_to_cylindrical',
]
