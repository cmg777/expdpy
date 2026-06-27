"""Shared low-level helpers used across the analytical modules.

These are pure, dependency-light utilities (numeric-aware level sorting, a sample-size
default opacity, time-axis coercion, the standard error, and an x-axis layout builder) that
several feature modules need. Centralizing them here keeps the feature modules
(``scatter``, ``trends``, ``by_group`` …) from importing private helpers out of one another.

This module imports only :mod:`numpy` / :mod:`pandas`, so it can be imported anywhere without
risking a cycle.
"""

from __future__ import annotations

import re
from math import log

import numpy as np
import pandas as pd
from pandas.api import types as pdt

__all__ = [
    "sorted_levels",
    "argsort_levels",
    "default_alpha",
    "try_convert_ts_id",
    "se",
    "xaxis",
]

# Full date strings only (YYYY-MM-DD / YYYY/MM/DD) — bare-year strings like "2013" must fall
# through to numeric (R's ``as.Date("2013")`` fails).
_FULL_DATE = re.compile(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}")


def sorted_levels(values: pd.Series) -> list[str]:
    """Return the distinct levels of ``values`` sorted numerically when possible.

    Group labels like ``"2"`` and ``"10"`` must order as 2 < 10, not lexically.
    """
    levels = list(dict.fromkeys(values.astype(str)))
    num = pd.to_numeric(pd.Series(levels), errors="coerce")
    if not num.isna().any():
        return [lvl for _, lvl in sorted(zip(num, levels, strict=True))]
    return sorted(levels)


def argsort_levels(index: pd.Index) -> np.ndarray:
    """Return a stable sort order (argsort indices) for ``index``, numeric-aware.

    Numbers sort numerically (``2`` before ``10``); a non-numeric index sorts lexically.
    """
    idx = index.astype(str)
    num = pd.to_numeric(pd.Series(idx), errors="coerce")
    keys = num.to_numpy() if not num.isna().any() else idx.to_numpy()
    return np.asarray(np.argsort(keys, kind="stable"))


def default_alpha(n: int) -> float:
    """Sample-size-based default opacity (ExPanDaR's formula)."""
    if n <= 0:
        return 1.0
    return min(1.0, 1.0 / (1.0 + max(0.0, log(n) - log(100))))


def try_convert_ts_id(s: pd.Series) -> tuple[pd.Series, bool]:
    """Coerce a time identifier to a nicer type for axis ticks.

    Cascade (mirrors ExPanDaR's ``try_convert_ts_id``): keep existing datetime/numeric
    types, else try full-date parsing, else numeric, else an ordered categorical.

    Returns
    -------
    tuple of (pandas.Series, bool)
        The converted series and whether it is an ordered categorical (discrete axis).
    """
    if pdt.is_datetime64_any_dtype(s):
        return s, False
    if pdt.is_numeric_dtype(s) and not pdt.is_bool_dtype(s):
        return s, False

    # For factor/categorical/object indices, try the same cascade R applies to the
    # character values: full-date -> numeric -> ordered categorical.
    str_vals = s.astype(str)
    if str_vals.str.match(_FULL_DATE).all():
        try:
            return pd.to_datetime(str_vals), False
        except (ValueError, TypeError):
            pass
    num = pd.to_numeric(str_vals, errors="coerce")
    if not num.isna().any():
        return pd.Series(num.to_numpy(), index=s.index), False
    cats = sorted(s.dropna().astype(str).unique(), key=str)
    return s.astype(str).astype(pd.CategoricalDtype(cats, ordered=True)), True


def se(s: pd.Series) -> float:
    """Return the standard error of the mean: sd / sqrt(n_non_missing)."""
    cnt = int(s.notna().sum())
    if cnt == 0:
        return np.nan
    return float(s.std(ddof=1) / np.sqrt(cnt))


def xaxis(
    time: str, ordered: bool, ts_values: pd.Series, title: str | None = None
) -> dict:
    """Build x-axis layout kwargs, fixing category order when discrete.

    ``title`` overrides the axis title (default: the bare ``time`` name).
    """
    axis: dict = {"title": title if title is not None else time}
    if ordered:
        cats = [str(c) for c in ts_values.cat.categories]
        axis.update(type="category", categoryorder="array", categoryarray=cats)
    return axis
