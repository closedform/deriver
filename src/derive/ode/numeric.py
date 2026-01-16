"""
numeric.py - Numerical ODE Solver.

Provides NDSolve for numerical differential equation solving.
"""

from typing import Any, Callable, Tuple, Union
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp


def NDSolve(
    ode_func: Union[Callable, Any],
    y0: Union[float, np.ndarray],
    var_range: Tuple[Any, float, float],
    **kwargs
) -> Any:
    """
    Numerical ODE solver.

    NDSolve[f, y0, {t, t0, tf}] - solve ODE numerically

    Args:
        ode_func: Function f(t, y) returning dy/dt, or symbolic expression
        y0: Initial condition y(t0)
        var_range: (var, t0, tf) tuple specifying integration range

    Keyword Args:
        Method: Solver method. Options:
            - "ExplicitRungeKutta" or "RK45" (default): 4th order Runge-Kutta
            - "ImplicitRungeKutta" or "Radau": Implicit Runge-Kutta (stiff ODEs)
            - "BDF": Backward Differentiation Formula (stiff ODEs)
            - "LSODA": Auto-switching stiff/non-stiff
            - "DOP853": High-order explicit Runge-Kutta
        MaxSteps: Maximum number of integration steps
        AccuracyGoal: Number of digits of accuracy (maps to rtol)
        PrecisionGoal: Number of digits of precision (maps to atol)
        StartingStepSize: Initial step size
        MaxStepSize: Maximum step size
        WorkingPrecision: Working precision for computations

    Returns:
        InterpolatingFunction-like object with .t and .y attributes

    Examples:
        >>> # dy/dt = -y with y(0) = 1
        >>> sol = NDSolve(lambda t, y: -y, 1.0, ('t', 0, 5))
        >>> sol.y[0][-1]  # y at t=5, approximately exp(-5)

        >>> # Stiff ODE with BDF method
        >>> sol = NDSolve(ode, y0, (t, 0, 10), Method="BDF")
    """
    if isinstance(var_range, (list, tuple)) and len(var_range) == 3:
        var, t0, tf = var_range
        t0_val = float(t0)
        tf_val = float(tf)

        # If ode_func is symbolic, convert to numerical
        if hasattr(ode_func, 'free_symbols'):
            # It's a SymPy expression - need to lambdify
            free_syms = list(ode_func.free_symbols)
            if len(free_syms) == 2:
                # Assume (t, y) format
                t_sym = var if isinstance(var, sp.Symbol) else sp.Symbol(str(var))
                y_sym = [s for s in free_syms if s != t_sym][0] if len(free_syms) > 1 else free_syms[0]
                f = sp.lambdify((t_sym, y_sym), ode_func, modules=['numpy'])
                ode_func = lambda t, y: f(t, y)
            else:
                f = sp.lambdify(free_syms[0], ode_func, modules=['numpy'])
                ode_func = lambda t, y: f(y)

        # Ensure y0 is array-like for solve_ivp
        y0_arr = np.atleast_1d(y0)

        # Map method names to scipy equivalents
        method_map = {
            'ExplicitRungeKutta': 'RK45',
            'RK45': 'RK45',
            'ImplicitRungeKutta': 'Radau',
            'Radau': 'Radau',
            'BDF': 'BDF',
            'LSODA': 'LSODA',
            'DOP853': 'DOP853',
            'RK23': 'RK23',
        }

        # Extract options
        method = kwargs.pop('Method', kwargs.pop('method', 'RK45'))
        method = method_map.get(method, method)

        # Convert options to scipy format
        scipy_kwargs = {}

        if 'AccuracyGoal' in kwargs:
            scipy_kwargs['rtol'] = 10 ** (-kwargs.pop('AccuracyGoal'))
        if 'PrecisionGoal' in kwargs:
            scipy_kwargs['atol'] = 10 ** (-kwargs.pop('PrecisionGoal'))
        if 'MaxSteps' in kwargs:
            scipy_kwargs['max_step'] = (tf_val - t0_val) / kwargs.pop('MaxSteps')
        if 'StartingStepSize' in kwargs:
            scipy_kwargs['first_step'] = kwargs.pop('StartingStepSize')
        if 'MaxStepSize' in kwargs:
            scipy_kwargs['max_step'] = kwargs.pop('MaxStepSize')

        # Remove options not supported by scipy
        kwargs.pop('WorkingPrecision', None)

        # Merge remaining kwargs
        scipy_kwargs.update(kwargs)

        # Solve the ODE
        result = solve_ivp(
            ode_func,
            (t0_val, tf_val),
            y0_arr,
            method=method,
            dense_output=True,
            **scipy_kwargs
        )

        return result

    raise ValueError("NDSolve requires format: NDSolve(ode_func, y0, (var, t0, tf))")


__all__ = ['NDSolve']
