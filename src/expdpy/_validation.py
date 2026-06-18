"""Shared input-validation helpers used across the analytical functions."""

from __future__ import annotations

import pandas as pd
from pandas.api import types as pdt

__all__ = [
    "ensure_dataframe",
    "is_numeric_or_logical",
    "numeric_logical_columns",
]


def ensure_dataframe(df: object) -> pd.DataFrame:
    """Return ``df`` as a DataFrame or raise ``TypeError``.

    Parameters
    ----------
    df
        Object expected to be a :class:`pandas.DataFrame`.

    Returns
    -------
    pandas.DataFrame
        The validated data frame (a shallow copy is *not* made).
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df needs to be a pandas DataFrame")
    return df


def is_numeric_or_logical(series: pd.Series) -> bool:
    """Return ``True`` if ``series`` is numeric or boolean (R's numeric-or-logical)."""
    return bool(pdt.is_numeric_dtype(series) or pdt.is_bool_dtype(series))


def numeric_logical_columns(df: pd.DataFrame) -> list[str]:
    """Return the names of columns that are numeric or boolean.

    Mirrors R's ``df[sapply(df, is.logical) | sapply(df, is.numeric)]``.
    """
    return [c for c in df.columns if is_numeric_or_logical(df[c])]
