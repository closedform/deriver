"""
Utility helpers for function aliases.

Provides a lightweight wrapper so aliased functions print using the
name of the alias rather than the underlying library function.
"""

from typing import Any, Callable


class FunctionAlias:
    """Callable wrapper that preserves a friendly name when printed."""

    def __init__(self, name: str, func: Callable[..., Any]):
        self._name = name
        self._func = func
        self.__doc__ = getattr(func, "__doc__", None)
        self.__wrapped__ = func

    def __call__(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - trivial delegate
        return self._func(*args, **kwargs)

    def __repr__(self) -> str:
        return self._name

    __str__ = __repr__

    @property
    def __name__(self) -> str:
        return self._name


_PRINTERS_PATCHED = False


def _patch_sympy_printers() -> None:
    """Patch SymPy printers so they respect alias names when available."""
    global _PRINTERS_PATCHED
    if _PRINTERS_PATCHED:
        return

    try:
        from sympy.printing.latex import LatexPrinter
        from sympy.printing.pretty.pretty import PrettyPrinter
        from sympy.printing.str import StrPrinter
    except ImportError:  # pragma: no cover - sympy always present in runtime
        return

    def _wrap(printer_cls):
        original = printer_cls._print_Function

        def _print_Function(self, expr):
            func = expr.func
            alias = getattr(func, "_alias_name", None)
            if alias:
                original_name = getattr(func, "__name__", None)
                try:
                    func.__name__ = alias
                    return original(self, expr)
                finally:
                    if original_name is not None:
                        func.__name__ = original_name
            return original(self, expr)

        printer_cls._print_Function = _print_Function

    for printer in (StrPrinter, PrettyPrinter, LatexPrinter):
        _wrap(printer)

    _PRINTERS_PATCHED = True


def alias_function(name: str, func: Callable[..., Any]) -> FunctionAlias:
    """
    Wrap a function so its string representation matches the alias name.

    Args:
        name: The public alias (e.g., "Sin")
        func: The underlying callable (e.g., sympy.sin)

    Returns:
        FunctionAlias: a callable wrapper that prints as ``name``.
    """
    _patch_sympy_printers()

    if getattr(func, "_alias_name", None) is None:
        try:
            setattr(func, "_alias_name", name)
        except Exception:
            pass

    return FunctionAlias(name, func)


__all__ = ["alias_function", "FunctionAlias"]
