"""Time-trend graphs: variable means and quantiles over an ordered time index."""

from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pandas.api import types as pdt

from expdpy._theme import apply_default_layout, color_for
from expdpy._types import QuantileTrendGraphResult, TrendGraphResult
from expdpy._validation import ensure_dataframe

__all__ = [
    "prepare_quantile_trend_graph",
    "prepare_trend_graph",
]

_FULL_DATE = re.compile(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}")


def _try_convert_ts_id(s: pd.Series) -> tuple[pd.Series, bool]:
    """Coerce a time-series identifier to a nicer type for axis ticks.

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
    # Full date strings only — bare-year strings like "2013" must fall through to numeric
    # (R's as.Date("2013") fails), so we require a YYYY-MM-DD / YYYY/MM/DD pattern.
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


def _se(s: pd.Series) -> float:
    """Return the standard error of the mean: sd / sqrt(n_non_missing)."""
    cnt = int(s.notna().sum())
    if cnt == 0:
        return np.nan
    return float(s.std(ddof=1) / np.sqrt(cnt))


def _xaxis(ts_id: str, ordered: bool, ts_values: pd.Series) -> dict:
    """Build x-axis layout kwargs, fixing category order when discrete."""
    axis: dict = {"title": ts_id}
    if ordered:
        cats = [str(c) for c in ts_values.cat.categories]
        axis.update(type="category", categoryorder="array", categoryarray=cats)
    return axis


def prepare_trend_graph(
    df: pd.DataFrame,
    ts_id: str,
    var: Sequence[str] | None = None,
) -> TrendGraphResult:
    """Line-plot the mean (with standard-error bars) of variables over time.

    Parameters
    ----------
    df
        Data frame containing ``ts_id`` and the numeric variables to plot.
    ts_id
        Column name of the time-series identifier.
    var
        Variables to plot. Defaults to all numeric columns other than ``ts_id``.

    Returns
    -------
    TrendGraphResult
        ``df`` (columns ``variable``, ``ts_id``, ``mean``, ``se``) and the Plotly ``fig``.
    """
    df = ensure_dataframe(df)
    if ts_id not in df.columns:
        raise ValueError("ts_id needs to be in df")
    if var is None:
        var = [
            c
            for c in df.columns
            if c != ts_id
            and pdt.is_numeric_dtype(df[c])
            and not pdt.is_bool_dtype(df[c])
        ]
    else:
        var = list(var)
        missing = [v for v in var if v not in df.columns]
        if missing:
            raise ValueError("var names need to be in df")

    sub = df[[ts_id, *var]].dropna()
    ts_conv, ordered = _try_convert_ts_id(sub[ts_id])
    sub = sub.assign(**{ts_id: ts_conv})

    long = sub.melt(
        id_vars=[ts_id], value_vars=var, var_name="variable", value_name="value"
    )
    gf = (
        long.groupby(["variable", ts_id], observed=True)["value"]
        .agg(mean="mean", se=_se)
        .reset_index()
    )

    fig = go.Figure()
    for idx, v in enumerate(var):
        part = gf[gf["variable"] == v].sort_values(ts_id)
        x = part[ts_id].astype(str) if ordered else part[ts_id]
        fig.add_trace(
            go.Scatter(
                x=x,
                y=part["mean"],
                error_y={"type": "data", "array": part["se"], "visible": True},
                mode="lines+markers",
                name=str(v),
                line={"color": color_for(idx)},
            )
        )
    apply_default_layout(
        fig, xaxis=_xaxis(ts_id, ordered, ts_conv), yaxis={"title": ""}
    )
    return TrendGraphResult(df=gf, fig=fig)


def prepare_quantile_trend_graph(
    df: pd.DataFrame,
    ts_id: str,
    quantiles: Sequence[float] = (0.05, 0.25, 0.5, 0.75, 0.95),
    var: str | None = None,
    *,
    points: bool = True,
) -> QuantileTrendGraphResult:
    """Line-plot quantiles of a single variable over time.

    Parameters
    ----------
    df
        Data frame containing ``ts_id`` and the variable to plot.
    ts_id
        Column name of the time-series identifier.
    quantiles
        Quantile levels to plot (each in ``(0, 1)``).
    var
        Variable to plot. Defaults to the last numeric column that is not ``ts_id``.
    points
        Whether to mark each observation with a point.

    Returns
    -------
    QuantileTrendGraphResult
        ``df`` (long format: ``ts_id``, ``quantile``, value) and the Plotly ``fig``.
    """
    df = ensure_dataframe(df)
    if ts_id not in df.columns:
        raise ValueError("ts_id needs to be in df")
    if var is None:
        candidates = [
            c
            for c in df.columns
            if c != ts_id
            and pdt.is_numeric_dtype(df[c])
            and not pdt.is_bool_dtype(df[c])
        ]
        if not candidates:
            raise ValueError("no numeric variable available")
        var = candidates[-1]
    if var not in df.columns:
        raise ValueError("var needs to be in df")

    sub = df[[ts_id, var]].dropna()
    ts_conv, ordered = _try_convert_ts_id(sub[ts_id])
    sub = sub.assign(**{ts_id: ts_conv})

    labels = [f"q{round(q * 100):02d}" for q in quantiles]
    wide = sub.groupby(ts_id, observed=True)[var].quantile(list(quantiles)).unstack()
    wide.columns = labels
    wide = wide.reset_index()
    gf = wide.melt(
        id_vars=[ts_id], value_vars=labels, var_name="quantile", value_name=var
    )
    gf["quantile"] = pd.Categorical(gf["quantile"], categories=labels, ordered=True)

    mode = "lines+markers" if points else "lines"
    fig = go.Figure()
    for idx, (label, q) in enumerate(zip(labels, quantiles, strict=True)):
        part = gf[gf["quantile"] == label].sort_values(ts_id)
        x = part[ts_id].astype(str) if ordered else part[ts_id]
        fig.add_trace(
            go.Scatter(
                x=x,
                y=part[var],
                mode=mode,
                name=str(q),
                line={"color": color_for(idx)},
            )
        )
    apply_default_layout(
        fig, xaxis=_xaxis(ts_id, ordered, ts_conv), yaxis={"title": var}
    )
    return QuantileTrendGraphResult(df=gf, fig=fig)
