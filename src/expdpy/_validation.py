"""Shared input-validation helpers used across the analytical functions."""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import pandas as pd
from pandas.api import types as pdt

__all__ = [
    "ensure_dataframe",
    "is_numeric_or_logical",
    "numeric_logical_columns",
    "ExpdpyWarning",
    "drop_missing",
    "require_columns",
]


class ExpdpyWarning(UserWarning):
    """Advisory warning raised by expdpy (e.g. dropped rows, sampling).

    A subclass of :class:`UserWarning`, so existing ``pytest.warns(UserWarning)``
    callers keep matching while users can silence *only* expdpy's advisory notices
    with ``warnings.filterwarnings("ignore", category=expdpy.ExpdpyWarning)``.
    """


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


def drop_missing(
    df: pd.DataFrame,
    subset: Sequence[str],
    *,
    func: str,
    stacklevel: int = 3,
) -> pd.DataFrame:
    """Drop rows with missing values in ``subset`` and warn if any were dropped.

    A consistent, advisory replacement for a silent ``df.dropna(subset=...)``: the
    complete-case frame is returned, and when rows are lost an :class:`ExpdpyWarning`
    naming the calling function, the count/percentage and the offending columns is
    emitted — in the same style as the library's sampling notices.

    Parameters
    ----------
    df
        Data frame to filter.
    subset
        Column names that must be non-missing for a row to be kept.
    func
        Name of the public function reporting the drop (prefixed in the message).
    stacklevel
        Stack level passed to :func:`warnings.warn` so the warning points at the
        user's call. The default ``3`` is correct when ``drop_missing`` is called
        directly inside a public function; pass ``4`` from a helper one frame deeper.

    Returns
    -------
    pandas.DataFrame
        The complete-case frame (rows with no missing value in ``subset``).
    """
    cols = list(subset)
    n_before = len(df)
    out = df.dropna(subset=cols)
    n_dropped = n_before - len(out)
    if n_dropped:
        pct = n_dropped / n_before if n_before else 0.0
        warnings.warn(
            f"{func}: dropped {n_dropped} of {n_before} row(s) "
            f"({pct:.0%}) with missing values in {cols}",
            ExpdpyWarning,
            stacklevel=stacklevel,
        )
    return out


def require_columns(df: pd.DataFrame, cols: Sequence[str], *, where: str) -> None:
    """Raise ``ValueError`` naming any of ``cols`` not present in ``df``.

    Parameters
    ----------
    df
        Data frame whose columns are checked.
    cols
        Column names that must all be present.
    where
        Name of the calling function, prefixed in the error message.
    """
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{where}: column(s) not found in df: {missing}")
