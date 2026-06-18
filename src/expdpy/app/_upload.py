"""Read user-uploaded data files for the ExPdPy app."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

__all__ = ["read_uploaded", "SUPPORTED_SUFFIXES"]

SUPPORTED_SUFFIXES = (".csv", ".xlsx", ".xls", ".parquet")


def read_uploaded(path: str, name: str | None = None) -> pd.DataFrame:
    """Read an uploaded data file into a DataFrame.

    Parameters
    ----------
    path
        Filesystem path to the uploaded file.
    name
        Original file name (used to infer the format when ``path`` lacks a suffix).

    Returns
    -------
    pandas.DataFrame
        The parsed data.

    Raises
    ------
    ValueError
        If the file format is not supported.
    """
    suffix = Path(name or path).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(
        f"unsupported file format '{suffix}'; supported: {', '.join(SUPPORTED_SUFFIXES)}"
    )
