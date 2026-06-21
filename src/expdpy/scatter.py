"""Scatter plot with optional size/color aesthetics and LOESS smoothing."""

from __future__ import annotations

import warnings
from math import log
from typing import Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pandas.api import types as pdt
from statsmodels.nonparametric.smoothers_lowess import lowess

from expdpy._panel import resolve_panel
from expdpy._theme import SEQUENTIAL_SCALE, apply_default_layout, color_for
from expdpy._types import ScatterPlotResult
from expdpy._validation import ensure_dataframe

__all__ = ["prepare_scatter_plot"]


def _default_alpha(n: int) -> float:
    """Sample-size-based default opacity (ExPanDaR's formula)."""
    if n <= 0:
        return 1.0
    return min(1.0, 1.0 / (1.0 + max(0.0, log(n) - log(100))))


def _lowess_curve(
    x: np.ndarray, y: np.ndarray, weights: np.ndarray | None, frac: float = 2 / 3
):
    """Return a smoothed (xs, ys) curve. ``weights`` approximate ggplot's weighted smooth."""
    if weights is None:
        fitted = lowess(y, x, frac=frac, return_sorted=True)
        return fitted[:, 0], fitted[:, 1]
    # Weighted approximation: resample observations proportional to their weight, then
    # run an ordinary lowess on the expanded sample. This mimics geom_smooth(weight=...).
    w = np.clip(weights, 0, None)
    if w.sum() == 0:
        return _lowess_curve(x, y, None, frac)
    probs = w / w.sum()
    reps = np.maximum(1, np.round(probs * len(x) * 3).astype(int))
    xx = np.repeat(x, reps)
    yy = np.repeat(y, reps)
    fitted = lowess(yy, xx, frac=frac, return_sorted=True)
    # collapse duplicate x from the expansion
    _, idx = np.unique(fitted[:, 0], return_index=True)
    return fitted[idx, 0], fitted[idx, 1]


def _lowess_band(x: np.ndarray, y: np.ndarray, xs: np.ndarray, n_boot: int = 50):
    """Pointwise 95% bootstrap band for a lowess fit on grid ``xs``."""
    rng = np.random.default_rng(0)
    n = len(x)
    preds = np.full((n_boot, len(xs)), np.nan)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        fitted = lowess(y[idx], x[idx], frac=2 / 3, return_sorted=True)
        preds[b] = np.interp(xs, fitted[:, 0], fitted[:, 1])
    lo = np.nanpercentile(preds, 2.5, axis=0)
    hi = np.nanpercentile(preds, 97.5, axis=0)
    return lo, hi


def prepare_scatter_plot(
    df: pd.DataFrame,
    x: str,
    y: str,
    *,
    color: str | None = None,
    size: str | None = None,
    loess: Literal[0, 1, 2] = 0,
    alpha: float | None = None,
    entity: str | None = None,
    time: str | None = None,
    connect: bool = False,
) -> ScatterPlotResult:
    """Scatter plot of ``y`` against ``x`` with optional aesthetics and a LOESS smoother.

    Parameters
    ----------
    df
        Data frame containing the variables.
    x, y
        Column names for the axes (both must be numeric).
    color
        Optional column mapped to marker color (numeric -> colorbar, otherwise discrete).
    size
        Optional numeric column mapped to marker size.
    loess
        ``0`` no smoother, ``1`` unweighted LOESS, ``2`` LOESS weighted by ``size``.
    alpha
        Marker opacity. If ``None``, a sample-size-based default is used.
    entity
        Cross-sectional (unit) identifier. Defaults to the panel ``entity`` declared via
        :func:`expdpy.set_panel`. Only used when ``connect=True``.
    time
        Time identifier used to order each unit's trajectory when ``connect=True``. Defaults
        to the panel ``time``.
    connect
        If ``True`` (and an ``entity`` is available), draw a faint line connecting each
        unit's points in time order — turning the scatter into a panel trajectory plot.

    Returns
    -------
    ScatterPlotResult
        ``df`` (the complete-case frame plotted) and ``fig`` (the Plotly scatter).

    Examples
    --------
    Basic — a plain scatter of two variables:

    ```python
    import expdpy as ex
    from expdpy.data import load_kuznets

    df = load_kuznets()
    ex.prepare_scatter_plot(df, x="log_gdp_pc", y="gini_regional").fig
    ```

    Advanced — map color and marker size to other columns, add a size-weighted LOESS
    smoother (the N-shaped Kuznets curve) and tune opacity:

    ```python
    ex.prepare_scatter_plot(
        df, x="log_gdp_pc", y="gini_regional",
        color="continent", size="population", loess=2, alpha=0.6,
    ).fig
    ```
    """
    df = ensure_dataframe(df)
    entity, time = resolve_panel(df, entity, time)
    for axis_name, axis_col in (("x", x), ("y", y)):
        if not pdt.is_numeric_dtype(df[axis_col]):
            raise ValueError(f"{axis_name} ({axis_col!r}) needs to be numeric")
    if connect and entity is None:
        warnings.warn(
            "connect=True needs an entity id (pass entity= or set_panel); ignoring it",
            stacklevel=2,
        )
        connect = False

    extra = [entity, time] if connect else []
    cols = list(dict.fromkeys(c for c in (x, y, color, size, *extra) if c))
    sub = df[cols].dropna()
    n = len(sub)
    if alpha is None:
        alpha = _default_alpha(n)

    xv = sub[x].to_numpy(dtype=float)
    yv = sub[y].to_numpy(dtype=float)
    size_vals = sub[size].to_numpy(dtype=float) if size else None

    def _sizeref(vals: np.ndarray) -> float:
        peak = float(np.nanmax(vals))
        return 2.0 * peak / (22.0**2) if peak > 0 else 1.0

    def _marker(extra: dict | None = None) -> dict:
        m: dict = {
            "opacity": alpha,
            "line": {"color": "rgba(255,255,255,0.5)", "width": 0.5},
        }
        if size_vals is not None:
            m.update(
                size=size_vals, sizemode="area", sizeref=_sizeref(size_vals), sizemin=3
            )
        if extra:
            m.update(extra)
        return m

    fig = go.Figure()
    if connect:
        ordered = sub.sort_values(time) if time is not None else sub
        for _uid, part in ordered.groupby(entity, observed=True):
            fig.add_trace(
                go.Scatter(
                    x=part[x].to_numpy(dtype=float),
                    y=part[y].to_numpy(dtype=float),
                    mode="lines",
                    line={"color": "rgba(78,121,167,0.25)", "width": 1},
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
    if color and not pdt.is_numeric_dtype(sub[color]):
        for idx, level in enumerate(sorted(sub[color].astype(str).unique())):
            mask = sub[color].astype(str).to_numpy() == level
            m = {
                "opacity": alpha,
                "color": color_for(idx),
                "line": {"color": "rgba(255,255,255,0.5)", "width": 0.5},
            }
            if size_vals is not None:
                m.update(
                    size=size_vals[mask],
                    sizemode="area",
                    sizeref=_sizeref(size_vals),
                    sizemin=3,
                )
            fig.add_trace(
                go.Scatter(
                    x=xv[mask], y=yv[mask], mode="markers", name=str(level), marker=m
                )
            )
    elif color:
        fig.add_trace(
            go.Scatter(
                x=xv,
                y=yv,
                mode="markers",
                marker=_marker(
                    {
                        "color": sub[color].to_numpy(dtype=float),
                        "colorscale": SEQUENTIAL_SCALE,
                        "showscale": True,
                        "colorbar": {"title": color},
                    }
                ),
                name=y,
            )
        )
    else:
        fig.add_trace(go.Scatter(x=xv, y=yv, mode="markers", marker=_marker(), name=y))

    if loess > 0 and n >= 4:
        weights = size_vals if loess == 2 and size else None
        xs, ys = _lowess_curve(xv, yv, weights)
        lo, hi = _lowess_band(xv, yv, xs)
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([xs, xs[::-1]]),
                y=np.concatenate([hi, lo[::-1]]),
                fill="toself",
                fillcolor="rgba(78,121,167,0.18)",
                line={"color": "rgba(0,0,0,0)"},
                hoverinfo="skip",
                showlegend=False,
                name="ci",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                line={"color": color_for(0), "width": 2},
                name="loess",
            )
        )

    apply_default_layout(fig, xaxis={"title": x}, yaxis={"title": y})
    return ScatterPlotResult(df=sub, fig=fig)
