"""
strings.py - String Operations.

Provides string manipulation functions.
"""

from typing import Any, Dict


def StringJoin(*args: Any) -> str:
    """
    Join strings.

    Args:
        *args: Strings to join

    Returns:
        Concatenated string
    """
    return ''.join(str(a) for a in args)


def StringLength(s: str) -> int:
    """
    Length of string.

    Args:
        s: String

    Returns:
        String length
    """
    return len(s)


def StringTake(s: str, n: int) -> str:
    """
    Take first n characters.

    Args:
        s: String
        n: Number of characters

    Returns:
        First n characters
    """
    return s[:n]


def StringDrop(s: str, n: int) -> str:
    """
    Drop first n characters.

    Args:
        s: String
        n: Number of characters

    Returns:
        String with first n characters removed
    """
    return s[n:]


def StringReplace(s: str, rules: Dict[str, str]) -> str:
    """
    Replace in string according to rules.

    Args:
        s: String to modify
        rules: Dictionary of replacements

    Returns:
        Modified string

    Examples:
        >>> StringReplace("hello world", {"hello": "hi", "world": "there"})
        "hi there"
    """
    result = s
    if isinstance(rules, dict):
        for old, new in rules.items():
            result = result.replace(old, new)
    return result


def ToString(expr: Any) -> str:
    """
    Convert to string.

    Args:
        expr: Expression to convert

    Returns:
        String representation
    """
    return str(expr)


__all__ = [
    'StringJoin', 'StringLength', 'StringTake', 'StringDrop',
    'StringReplace', 'ToString',
]
