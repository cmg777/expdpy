"""Outlier treatment by winsorizing or truncating (port of ``treat_outliers``)."""

from __future__ import annotations

from typing import overload

import numpy as np
import pandas as pd
from pandas.api import types as pdt

__all__ = ["treat_outliers"]


def _treat_vector(
    values: np.ndarray, *, truncate: bool, percentile: float, method: str
) -> np.ndarray:
    """Winsorize/truncate a 1-D float array using symmetric percentile cut-offs."""
    out = values.astype(float, copy=True)
    out[~np.isfinite(out)] = np.nan
    if np.all(np.isnan(out)):
        return out
    lo, hi = np.nanquantile(out, [percentile, 1.0 - percentile], method=method)  # type: ignore[call-overload]
    if truncate:
        out[(out < lo) | (out > hi)] = np.nan
    else:
        out = np.clip(out, lo, hi)
        # np.clip leaves existing NaNs untouched, which matches R's behaviour.
    return out


@overload
def treat_outliers(
    x: pd.Series,
    percentile: float = ...,
    *,
    truncate: bool = ...,
    by: pd.Series | np.ndarray | None = ...,
    method: str = ...,
) -> pd.Series: ...


@overload
def treat_outliers(
    x: pd.DataFrame,
    percentile: float = ...,
    *,
    truncate: bool = ...,
    by: str | pd.Series | np.ndarray | None = ...,
    method: str = ...,
) -> pd.DataFrame: ...


@overload
def treat_outliers(
    x: np.ndarray,
    percentile: float = ...,
    *,
    truncate: bool = ...,
    by: pd.Series | np.ndarray | None = ...,
    method: str = ...,
) -> np.ndarray: ...


def treat_outliers(x, percentile=0.01, *, truncate=False, by=None, method="linear"):
    """Treat numerical outliers by winsorizing or truncating.

    For each numeric variable, values outside the ``[percentile, 1 - percentile]``
    quantile range are either clipped to the boundary (winsorizing, the default) or set
    to ``NaN`` (truncating). Non-finite values (``inf``/``-inf``) are set to ``NaN`` first.

    Parameters
    ----------
    x
        A :class:`pandas.Series`, :class:`pandas.DataFrame`, or :class:`numpy.ndarray`.
        For a DataFrame, only numeric columns are treated; other columns pass through
        unchanged.
    percentile
        The (symmetric) tail probability defining the cut-offs. Must satisfy
        ``0 < percentile < 0.5``. Defaults to ``0.01``.
    truncate
        If ``True``, out-of-bounds values are set to ``NaN``; otherwise they are clipped
        to the boundary value (winsorizing). Defaults to ``False``.
    by
        Optional grouping. Either an array/Series of group labels (same length as ``x``)
        or, for a DataFrame ``x``, a column name. Outlier cut-offs are then computed
        within each group. Must not contain missing values.
    method
        Quantile interpolation method passed to :func:`numpy.nanquantile`. Defaults to
        ``"linear"`` (equivalent to R's ``type = 7``). Use ``"averaged_inverted_cdf"`` to
        match Stata / R ``type = 2``.

    Returns
    -------
    Same type as ``x``
        The outlier-treated data. A DataFrame keeps its non-numeric columns, column order
        and index; a Series keeps its index and name; an ndarray keeps its shape.

    Examples
    --------
    >>> import numpy as np
    >>> treat_outliers(np.arange(1, 101, dtype=float), 0.05)[:3]
    array([6., 6., 6.])
    """
    if not isinstance(percentile, (int, float)) or isinstance(percentile, bool):
        raise TypeError("'percentile' needs to be a numeric scalar")
    if not (0.0 < percentile < 0.5):
        raise ValueError("'percentile' needs to be > 0 and < 0.5")
    if not isinstance(truncate, bool):
        raise TypeError("'truncate' needs to be a logical scalar")

    is_df = isinstance(x, pd.DataFrame)
    is_series = isinstance(x, pd.Series)
    is_ndarray = isinstance(x, np.ndarray)
    if not (is_df or is_series or is_ndarray):
        raise TypeError("'x' is of invalid type")

    # Resolve and validate the grouping vector.
    by_vec: np.ndarray | None = None
    if by is not None:
        if isinstance(by, str):
            if not is_df:
                raise ValueError("'by' is a string but no DataFrame provided.")
            by_vec = np.asarray(x[by])
        else:
            by_vec = np.asarray(by)
        n_rows = len(x) if not is_ndarray else x.shape[0]
        if pd.isna(by_vec).any():
            raise ValueError("by vector contains NA values")
        if len(by_vec) != n_rows:
            raise ValueError("by vector number of rows differs from x")

    def _treat_column(col: pd.Series) -> pd.Series:
        if by_vec is None:
            treated = _treat_vector(
                col.to_numpy(), truncate=truncate, percentile=percentile, method=method
            )
            return pd.Series(treated, index=col.index, name=col.name)
        grouper = pd.Series(by_vec, index=col.index)
        return col.groupby(grouper, sort=False).transform(
            lambda s: _treat_vector(
                s.to_numpy(), truncate=truncate, percentile=percentile, method=method
            )
        )

    if is_df:
        out = x.copy()
        for col in x.columns:
            if pdt.is_numeric_dtype(x[col]) and not pdt.is_bool_dtype(x[col]):
                out[col] = _treat_column(x[col])
        return out

    if is_series:
        return _treat_column(x)

    # numpy array (1-D vector or 2-D matrix).
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        frame = pd.DataFrame({"_x": arr})
        return _treat_column(frame["_x"]).to_numpy()
    out_cols = [
        _treat_column(pd.Series(arr[:, j], name=j)).to_numpy()
        for j in range(arr.shape[1])
    ]
    return np.column_stack(out_cols)
