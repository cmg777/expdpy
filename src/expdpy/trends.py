"""Time-trend graphs: variable means and quantiles over an ordered time index."""

from __future__ import annotations

import re
import warnings
from collections.abc import Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pandas.api import types as pdt

from expdpy._labels import resolve_label, resolve_labels
from expdpy._panel import resolve_panel
from expdpy._theme import apply_default_layout, blank_rangeslider, color_for
from expdpy._types import QuantileTrendGraphResult, TrendGraphResult
from expdpy._validation import ensure_dataframe

__all__ = [
    "explore_quantile_trend_plot",
    "explore_trend_plot",
]

_FULL_DATE = re.compile(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}")
_SPAGHETTI_LINE = {"color": "rgba(78,121,167,0.18)", "width": 1}


def _try_convert_ts_id(s: pd.Series) -> tuple[pd.Series, bool]:
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


def _xaxis(
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


def _numeric_vars(df: pd.DataFrame, *exclude: str | None) -> list[str]:
    """Numeric (non-boolean) columns of ``df`` excluding the given identifier columns."""
    drop = {c for c in exclude if c}
    return [
        c
        for c in df.columns
        if c not in drop
        and pdt.is_numeric_dtype(df[c])
        and not pdt.is_bool_dtype(df[c])
    ]


def explore_trend_plot(
    df: pd.DataFrame,
    var: Sequence[str] | None = None,
    *,
    time: str | None = None,
    entity: str | None = None,
    spaghetti: bool = False,
) -> TrendGraphResult:
    """Line-plot the mean (with standard-error bars) of variables over time.

    Parameters
    ----------
    df
        Data frame containing ``time`` and the numeric variables to plot.
    var
        Variables to plot. Defaults to all numeric columns other than ``time``/``entity``.
    time
        Column name of the time identifier. Defaults to the panel ``time`` declared via
        :func:`expdpy.set_panel`.
    entity
        Column name of the cross-sectional (unit) identifier. Only used when
        ``spaghetti=True``. Defaults to the panel ``entity`` declared via
        :func:`expdpy.set_panel`.
    spaghetti
        If ``True`` (and a single ``var`` and an ``entity`` are available), draw a faint
        per-unit trajectory backdrop behind the mean line. For richer per-unit views see
        :func:`expdpy.explore_spaghetti_plot`.

    Returns
    -------
    TrendGraphResult
        ``df`` (columns ``variable``, ``time``, ``mean``, ``se``) and the Plotly ``fig``.

    Examples
    --------
    Basic — mean of a single variable over time, with standard-error bars:

    ```python
    import expdpy as ex
    from expdpy.data import load_kuznets

    df = load_kuznets()
    ex.explore_trend_plot(df, var=["gini_regional"], time="year").fig
    ```

    Advanced — several variables on one chart, with the aggregated frame from ``.df``:

    ```python
    result = ex.explore_trend_plot(
        df, var=["gini_regional", "trade_share"], time="year"
    )
    result.fig
    result.df.head()
    ```
    """
    df = ensure_dataframe(df)
    entity, time = resolve_panel(df, entity, time, require_time=True)
    assert time is not None  # require_time=True guarantees this
    if var is None:
        var = _numeric_vars(df, time, entity)
    else:
        var = list(var)
        missing = [v for v in var if v not in df.columns]
        if missing:
            raise ValueError("var names need to be in df")

    if spaghetti and entity is None:
        warnings.warn(
            "spaghetti=True needs an entity id (pass entity= or set_panel); ignoring it",
            stacklevel=2,
        )
        spaghetti = False
    if spaghetti and len(var) != 1:
        warnings.warn(
            "the spaghetti backdrop is drawn only for a single variable; ignoring it",
            stacklevel=2,
        )
        spaghetti = False

    # Resolve display labels before slicing (column selection drops df.attrs).
    time_label = resolve_label(df, time)
    var_labels = resolve_labels(df, var)

    id_cols = [time, *([entity] if spaghetti else [])]
    sub = df[[*id_cols, *var]].copy()
    ts_conv, ordered = _try_convert_ts_id(sub[time])
    sub = sub.assign(**{time: ts_conv})

    long = sub.melt(
        id_vars=id_cols, value_vars=var, var_name="variable", value_name="value"
    )
    # Drop missing per series (not listwise across all variables) so each variable's mean
    # uses every observation it actually has.
    gf = (
        long.dropna(subset=["value"])
        .groupby(["variable", time], observed=True)["value"]
        .agg(mean="mean", se=_se)
        .reset_index()
    )

    fig = go.Figure()
    if spaghetti:
        v = var[0]
        raw = sub[[entity, time, v]].dropna(subset=[v])
        for _uid, part in raw.groupby(entity, observed=True):
            part = part.sort_values(time)
            x = part[time].astype(str) if ordered else part[time]
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=part[v],
                    mode="lines",
                    line=_SPAGHETTI_LINE,
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
    for idx, v in enumerate(var):
        part = gf[gf["variable"] == v].sort_values(time)
        x = part[time].astype(str) if ordered else part[time]
        fig.add_trace(
            go.Scatter(
                x=x,
                y=part["mean"],
                error_y={"type": "data", "array": part["se"], "visible": True},
                mode="lines+markers",
                name=var_labels[idx],
                line={"color": color_for(idx), "width": 2.5},
                hovertemplate=f"%{{fullData.name}}<br>{time_label}=%{{x}}<br>"
                "mean=%{y:.4g}<extra></extra>",
            )
        )
    xaxis = _xaxis(time, ordered, ts_conv, title=time_label)
    if not ordered:
        xaxis["rangeslider"] = blank_rangeslider(fig)
    yaxis_title = var_labels[0] if len(var) == 1 else "Value"
    apply_default_layout(fig, xaxis=xaxis, yaxis={"title": yaxis_title})
    return TrendGraphResult(df=gf, fig=fig)


def explore_quantile_trend_plot(
    df: pd.DataFrame,
    quantiles: Sequence[float] = (0.05, 0.25, 0.5, 0.75, 0.95),
    var: str | None = None,
    *,
    time: str | None = None,
    points: bool = True,
) -> QuantileTrendGraphResult:
    """Line-plot quantiles of a single variable over time.

    Parameters
    ----------
    df
        Data frame containing ``time`` and the variable to plot.
    quantiles
        Quantile levels to plot (each in ``(0, 1)``).
    var
        Variable to plot. Defaults to the last numeric column that is not ``time``.
    time
        Column name of the time identifier. Defaults to the panel ``time`` declared via
        :func:`expdpy.set_panel`.
    points
        Whether to mark each observation with a point.

    Returns
    -------
    QuantileTrendGraphResult
        ``df`` (long format: ``time``, ``quantile``, value) and the Plotly ``fig``.

    Examples
    --------
    Basic — the default quantiles of a variable over time:

    ```python
    import expdpy as ex
    from expdpy.data import load_kuznets

    df = load_kuznets()
    ex.explore_quantile_trend_plot(df, var="gini_regional", time="year").fig
    ```

    Advanced — custom quantile levels and no per-observation points:

    ```python
    ex.explore_quantile_trend_plot(
        df, quantiles=(0.1, 0.5, 0.9), var="gini_regional", time="year", points=False
    ).fig
    ```
    """
    df = ensure_dataframe(df)
    _entity, time = resolve_panel(df, None, time, require_time=True)
    assert time is not None  # require_time=True guarantees this
    quantiles = list(quantiles)
    if not quantiles or any(not (0 < q < 1) for q in quantiles):
        raise ValueError("quantiles need to be in the open interval (0, 1)")
    if len(set(quantiles)) != len(quantiles):
        raise ValueError("quantiles need to be unique")
    if var is None:
        candidates = _numeric_vars(df, time)
        if not candidates:
            raise ValueError("no numeric variable available")
        var = candidates[-1]
    if var not in df.columns:
        raise ValueError("var needs to be in df")

    # Resolve display labels before slicing (column selection drops df.attrs).
    time_label = resolve_label(df, time)
    var_label = resolve_label(df, var)

    sub = df[[time, var]].dropna()
    ts_conv, ordered = _try_convert_ts_id(sub[time])
    sub = sub.assign(**{time: ts_conv})

    labels = [f"q{round(q * 100):02d}" for q in quantiles]
    wide = sub.groupby(time, observed=True)[var].quantile(list(quantiles)).unstack()
    wide.columns = labels
    wide = wide.reset_index()
    gf = wide.melt(
        id_vars=[time], value_vars=labels, var_name="quantile", value_name=var
    )
    gf["quantile"] = pd.Categorical(gf["quantile"], categories=labels, ordered=True)

    mode = "lines+markers" if points else "lines"
    # A sequential blue ramp conveys the quantile order better than the qualitative palette.
    n_q = len(labels)
    ramp = [
        f"rgb({int(198 - 160 * t)},{int(219 - 130 * t)},{int(239 - 100 * t)})"
        for t in (np.linspace(0, 1, n_q) if n_q > 1 else [1.0])
    ]
    fig = go.Figure()
    for idx, (label, q) in enumerate(zip(labels, quantiles, strict=True)):
        part = gf[gf["quantile"] == label].sort_values(time)
        x = part[time].astype(str) if ordered else part[time]
        # Shade the band between this quantile line and the previous one.
        fill = "tonexty" if idx > 0 else None
        fig.add_trace(
            go.Scatter(
                x=x,
                y=part[var],
                mode=mode,
                name=str(q),
                line={"color": ramp[idx]},
                fill=fill,
                fillcolor="rgba(78,121,167,0.08)",
                hovertemplate=(
                    f"q={q}<br>{time_label}=%{{x}}<br>{var_label}=%{{y:.4g}}<extra></extra>"
                ),
            )
        )
    xaxis = _xaxis(time, ordered, ts_conv, title=time_label)
    if not ordered:
        xaxis["rangeslider"] = blank_rangeslider(fig)
    apply_default_layout(fig, xaxis=xaxis, yaxis={"title": var_label})
    return QuantileTrendGraphResult(df=gf, fig=fig)
