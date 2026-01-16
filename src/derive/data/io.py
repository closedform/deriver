"""
io.py - Data Import and Export.

Provides Import and Export functions.
"""

from typing import Any, Optional
from pathlib import Path
import json
import polars as pl


def Import(path: str, format: Optional[str] = None) -> Any:
    """
    Import function.

    Auto-detects file format and loads data into appropriate structure.

    Supported formats:
    - CSV, TSV -> Polars DataFrame
    - Parquet -> Polars DataFrame
    - JSON -> Python dict/list
    - Text -> String

    Args:
        path: Path to the file
        format: Optional format override

    Returns:
        Loaded data

    Examples:
        >>> data = Import("data.csv")  # Returns Polars DataFrame
        >>> data = Import("data.parquet")
        >>> data = Import("config.json")
    """
    path = Path(path)

    if format is None:
        # Auto-detect format from extension
        ext = path.suffix.lower()
        format_map = {
            '.csv': 'CSV',
            '.tsv': 'TSV',
            '.parquet': 'Parquet',
            '.json': 'JSON',
            '.txt': 'Text',
            '.text': 'Text',
        }
        format = format_map.get(ext, 'Text')

    format = format.upper() if format else 'TEXT'

    if format == 'CSV':
        return pl.read_csv(path)
    elif format == 'TSV':
        return pl.read_csv(path, separator='\t')
    elif format == 'PARQUET':
        return pl.read_parquet(path)
    elif format == 'JSON':
        with open(path) as f:
            return json.load(f)
    elif format == 'TEXT':
        return path.read_text()
    else:
        raise ValueError(f"Unknown format: {format}")


def Export(path: str, data: Any, format: Optional[str] = None) -> Path:
    """
    Export function.

    Exports data to file in specified format.

    Args:
        path: Path to save to
        data: Data to export
        format: Optional format override

    Returns:
        Path to the exported file

    Examples:
        >>> Export("output.csv", df)
        >>> Export("result.json", {"a": 1, "b": 2})
    """
    path = Path(path)

    if format is None:
        ext = path.suffix.lower()
        format_map = {
            '.csv': 'CSV',
            '.tsv': 'TSV',
            '.parquet': 'Parquet',
            '.json': 'JSON',
            '.txt': 'Text',
            '.pdf': 'PDF',
        }
        format = format_map.get(ext, 'Text')

    format = format.upper() if format else 'TEXT'

    if format == 'CSV':
        if isinstance(data, pl.DataFrame):
            data.write_csv(path)
        else:
            pl.DataFrame(data).write_csv(path)
    elif format == 'TSV':
        if isinstance(data, pl.DataFrame):
            data.write_csv(path, separator='\t')
        else:
            pl.DataFrame(data).write_csv(path, separator='\t')
    elif format == 'PARQUET':
        if isinstance(data, pl.DataFrame):
            data.write_parquet(path)
        else:
            pl.DataFrame(data).write_parquet(path)
    elif format == 'JSON':
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    elif format == 'TEXT':
        path.write_text(str(data))
    elif format == 'PDF':
        # PDF export for matplotlib figures
        if hasattr(data, 'savefig'):
            data.savefig(path, format='pdf')
        else:
            raise ValueError("PDF export requires a matplotlib figure")
    else:
        raise ValueError(f"Unknown format: {format}")

    return path


__all__ = ['Import', 'Export']
