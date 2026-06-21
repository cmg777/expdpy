"""Distribution and transition dynamics: how a variable's shape and states evolve over time."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from great_tables import GT
from pandas.api import types as pdt
from scipy.stats import gaussian_kde

from expdpy._panel import resolve_panel
from expdpy._theme import (
    SEQUENTIAL_SCALE,
    apply_default_layout,
    color_for,
)
from expdpy._types import (
    DistributionOverTimeResult,
    TransitionMatrixResult,
    WithinPersistenceResult,
)
from expdpy._validation import ensure_dataframe
from expdpy.scatter import _default_alpha
from expdpy.trends import _try_convert_ts_id

__all__ = [
    "prepare_distribution_over_time",
    "prepare_transition_matrix",
    "prepare_within_persistence",
]


def _ridge_color(t: float) -> str:
    """Return the light-to-dark blue fill for ridge ``t`` in ``[0, 1]`` (earliest lightest)."""
    return f"rgba({int(198 - 120 * t)},{int(219 - 110 * t)},{int(239 - 80 * t)},0.78)"


def _period_density(
    vals: np.ndarray, xs: np.ndarray, bandwidth: float | None, bins: int
) -> np.ndarray:
    """Density of ``vals`` on grid ``xs``: a Gaussian KDE, or a histogram fallback."""
    vals = vals[np.isfinite(vals)]
    if vals.size >= 5 and np.ptp(vals) > 0:
        try:
            return gaussian_kde(vals, bw_method=bandwidth)(xs)
        except (np.linalg.LinAlgError, ValueError):  # pragma: no cover - singular cov
            pass
    if vals.size == 0:
        return np.zeros_like(xs)
    counts, edges = np.histogram(vals, bins=bins, range=(xs[0], xs[-1]), density=True)
    centers = (edges[:-1] + edges[1:]) / 2.0
    return np.interp(xs, centers, counts, left=0.0, right=0.0)


def _even_periods(periods: list, max_periods: int | None) -> list:
    """Evenly subsample ``periods`` to at most ``max_periods`` (keeping order); warn if cut."""
    if not max_periods or len(periods) <= max_periods:
        return periods
    idx = sorted(
        set(np.linspace(0, len(periods) - 1, max_periods).astype(int).tolist())
    )
    warnings.warn(
        f"distribution_over_time: showing {len(idx)} of {len(periods)} periods "
        "(evenly sampled)",
        stacklevel=3,
    )
    return [periods[i] for i in idx]


def prepare_distribution_over_time(
    df: pd.DataFrame,
    var: str,
    *,
    entity: str | None = None,
    time: str | None = None,
    style: Literal["ridgeline", "animated_hist", "animated_violin"] = "ridgeline",
    bins: int = 30,
    bandwidth: float | None = None,
    max_periods: int | None = 24,
) -> DistributionOverTimeResult:
    """Show how the distribution of ``var`` shifts across periods.

    Parameters
    ----------
    df
        Panel data frame.
    var
        Numeric variable whose distribution is tracked.
    entity
        Cross-sectional id (accepted for panel parity; the per-period distribution pools
        across units). Defaults to the panel ``entity``.
    time
        Time identifier. Defaults to the panel ``time``.
    style
        ``"ridgeline"`` (default; stacked per-period densities), ``"animated_hist"`` or
        ``"animated_violin"`` (Plotly animation over periods).
    bins
        Histogram bins (animated_hist and the KDE fallback).
    bandwidth
        KDE bandwidth (``bw_method``) for the ridgeline; ``None`` uses Scott's rule.
    max_periods
        Cap on the number of periods drawn (evenly sampled above it).

    Returns
    -------
    DistributionOverTimeResult
        ``df`` (the complete-case ``(time, var)`` frame) and the Plotly ``fig``.

    Examples
    --------
    ```python
    import expdpy as ex
    from expdpy.data import load_kuznets

    df = load_kuznets()
    ex.prepare_distribution_over_time(df, var="gini_regional", time="year").fig
    ```
    """
    df = ensure_dataframe(df)
    _entity, time = resolve_panel(df, entity, time, require_time=True)
    assert time is not None
    if var not in df.columns:
        raise ValueError("var needs to be in df")
    if not pdt.is_numeric_dtype(df[var]):
        raise ValueError(f"var ({var!r}) needs to be numeric")

    sub = df[[time, var]].dropna()
    if sub.empty:
        raise ValueError(f"var ({var!r}) has no complete observations")
    sub = sub.assign(**{time: _try_convert_ts_id(sub[time])[0]})
    periods = _even_periods(sorted(sub[time].unique()), max_periods)
    sub = sub[sub[time].isin(periods)]
    if len(periods) < 2:
        warnings.warn(
            "distribution_over_time: only one period available — the view is degenerate",
            stacklevel=2,
        )

    lo, hi = float(sub[var].min()), float(sub[var].max())
    if style == "ridgeline":
        fig = _ridgeline(sub, time, var, periods, lo, hi, bandwidth, bins)
    else:
        fig = _animated(sub, time, var, periods, lo, hi, bins, style)
    return DistributionOverTimeResult(df=sub.reset_index(drop=True), fig=fig)


def _ridgeline(sub, time, var, periods, lo, hi, bandwidth, bins) -> go.Figure:
    """Build a ridgeline (joyplot) of per-period densities on a shared x-grid."""
    xs = np.linspace(lo, hi, 200)
    dens = {
        p: _period_density(sub[sub[time] == p][var].to_numpy(), xs, bandwidth, bins)
        for p in periods
    }
    peak = max((float(d.max()) for d in dens.values() if d.size), default=1.0) or 1.0
    spacing = 1.0
    scale = 1.6 * spacing / peak
    fig = go.Figure()
    n = len(periods)
    for i in reversed(range(n)):  # draw top-down so lower ridges overlap on top
        p = periods[i]
        baseline = i * spacing
        y_top = baseline + dens[p] * scale
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([xs, xs[::-1]]),
                y=np.concatenate([y_top, np.full_like(xs, baseline)[::-1]]),
                fill="toself",
                mode="lines",
                line={"color": "rgba(60,60,60,0.5)", "width": 0.8},
                fillcolor=_ridge_color(i / max(1, n - 1)),
                name=str(p),
                hoverinfo="skip",
                showlegend=False,
            )
        )
    apply_default_layout(
        fig,
        xaxis={"title": var},
        yaxis={
            "title": time,
            "tickmode": "array",
            "tickvals": [i * spacing for i in range(n)],
            "ticktext": [str(p) for p in periods],
        },
    )
    return fig


def _animated(sub, time, var, periods, lo, hi, bins, style) -> go.Figure:
    """Build an animated histogram / violin with a play button and a period slider."""

    def _trace(p):
        vals = sub[sub[time] == p][var]
        if style == "animated_hist":
            return go.Histogram(
                x=vals,
                xbins={
                    "start": lo,
                    "end": hi,
                    "size": (hi - lo) / bins if hi > lo else 1,
                },
                marker={"color": color_for(0)},
            )
        return go.Violin(
            y=vals,
            name=str(p),
            fillcolor=color_for(0),
            line_color=color_for(0),
            meanline_visible=True,
        )

    frames = [go.Frame(data=[_trace(p)], name=str(p)) for p in periods]
    fig = go.Figure(data=[_trace(periods[0])], frames=frames)
    if style == "animated_hist":
        fig.update_xaxes(title=var, range=[lo, hi])
        fig.update_yaxes(title="Count")
    else:
        fig.update_yaxes(title=var)
    fig.update_layout(
        updatemenus=[
            {
                "type": "buttons",
                "x": 0,
                "y": 1.15,
                "xanchor": "left",
                "buttons": [
                    {
                        "label": "▶ Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 600, "redraw": True},
                                "fromcurrent": True,
                            },
                        ],
                    },
                    {
                        "label": "❚❚ Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                            },
                        ],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "currentvalue": {"prefix": f"{time} = "},
                "steps": [
                    {
                        "label": str(p),
                        "method": "animate",
                        "args": [
                            [str(p)],
                            {
                                "frame": {"duration": 0, "redraw": True},
                                "mode": "immediate",
                            },
                        ],
                    }
                    for p in periods
                ],
            }
        ],
    )
    apply_default_layout(fig)
    return fig


def _bin_states(
    series: pd.Series,
    n_bins: int,
    bin_method: str,
    bin_labels: Sequence[str] | None,
) -> tuple[pd.Series, list[str]]:
    """Map a variable to ordered discrete state labels (categoricals pass through; numerics bin)."""
    if not pdt.is_numeric_dtype(series):
        states = sorted(series.dropna().astype(str).unique())
        return series.astype(str), states
    if bin_method == "quantile":
        cats = pd.qcut(series, n_bins, duplicates="drop")
    elif bin_method == "equal_width":
        cats = pd.cut(series, n_bins)
    else:  # pragma: no cover - guarded by Literal
        raise ValueError("bin_method needs to be 'quantile' or 'equal_width'")
    k = len(cats.cat.categories)
    if k < n_bins:
        warnings.warn(
            f"transition_matrix: requested {n_bins} bins but only {k} distinct bins "
            "exist (ties dropped)",
            stacklevel=3,
        )
    if bin_labels is not None:
        if len(bin_labels) != k:
            raise ValueError(f"bin_labels must have length {k} (got {len(bin_labels)})")
        labels = list(bin_labels)
    else:
        labels = [f"Q{i + 1}" for i in range(k)]
    cats = cats.cat.rename_categories(labels)
    return cats.astype(str), labels


def prepare_transition_matrix(
    df: pd.DataFrame,
    var: str,
    *,
    entity: str | None = None,
    time: str | None = None,
    n_bins: int = 5,
    bin_method: Literal["quantile", "equal_width"] = "quantile",
    bin_labels: Sequence[str] | None = None,
    lag: int = 1,
    normalize: Literal["row", "none"] = "row",
    caption: str = "Transition Matrix",
) -> TransitionMatrixResult:
    """Period-to-period transition matrix of a discrete (or binned) state within units.

    Parameters
    ----------
    df
        Panel data frame.
    var
        State variable. Categorical/object variables use their categories; numeric variables
        are binned into ``n_bins`` states.
    entity, time
        Panel identifiers (default to those declared via :func:`expdpy.set_panel`).
    n_bins
        Number of bins for a numeric ``var``.
    bin_method
        ``"quantile"`` (equal-mass, default) or ``"equal_width"``.
    bin_labels
        Optional labels for the bins (length must match the resulting number of bins).
    lag
        Number of periods ahead for the transition (default 1). Only consecutive observed
        periods that are exactly ``lag`` apart are counted; gaps are skipped.
    normalize
        ``"row"`` (default) gives conditional transition probabilities; ``"none"`` gives raw
        counts.
    caption
        Great Tables header.

    Returns
    -------
    TransitionMatrixResult
        ``df`` (K-by-K matrix), ``counts``, ``fig``, ``gt`` and ``states``.

    Examples
    --------
    ```python
    import expdpy as ex
    from expdpy.data import load_kuznets

    df = load_kuznets()
    ex.prepare_transition_matrix(
        df, var="gini_regional", entity="country", time="year", n_bins=4
    ).fig
    ```
    """
    df = ensure_dataframe(df)
    entity, time = resolve_panel(
        df, entity, time, require_entity=True, require_time=True
    )
    assert entity is not None and time is not None
    if var not in df.columns:
        raise ValueError("var needs to be in df")

    sub = df[[entity, time, var]].dropna().copy()
    sub[time] = _try_convert_ts_id(sub[time])[0]
    if sub[time].nunique() < 2:
        raise ValueError("transition matrix needs at least two periods")

    state, states = _bin_states(sub[var], n_bins, bin_method, bin_labels)
    sub = sub.assign(_state=state.to_numpy())
    rank = {p: i for i, p in enumerate(sorted(sub[time].unique()))}
    sub["_rank"] = sub[time].map(rank)
    sub = sub.sort_values([entity, "_rank"])
    grp = sub.groupby(entity, observed=True)
    sub["_next_state"] = grp["_state"].shift(-lag)
    sub["_next_rank"] = grp["_rank"].shift(-lag)

    valid = sub["_next_rank"] - sub["_rank"] == lag
    pairs = sub[valid & sub["_next_state"].notna()]
    dropped = int((sub["_next_state"].notna() & ~valid).sum())
    if dropped:
        warnings.warn(
            f"transition_matrix: skipped {dropped} pair(s) spanning a gap larger than "
            f"lag={lag}",
            stacklevel=2,
        )
    if pairs.empty:
        raise ValueError("no consecutive transitions found")

    counts = (
        pd.crosstab(pairs["_state"], pairs["_next_state"])
        .reindex(index=states, columns=states, fill_value=0)
        .astype(int)
    )
    if normalize == "row":
        row_sums = counts.sum(axis=1).replace(0, np.nan)
        matrix = counts.div(row_sums, axis=0)
        text = matrix.to_numpy(dtype=float)
        texttemplate = "%{z:.0%}"
        zmax: float | None = 1.0
        hover = "from %{y} → to %{x}: %{z:.1%}<extra></extra>"
        cbar = "P(next | current)"
    else:
        matrix = counts.astype(float)
        text = counts.to_numpy(dtype=float)
        texttemplate = "%{z:.0f}"
        zmax = None
        hover = "from %{y} → to %{x}: %{z:.0f}<extra></extra>"
        cbar = "count"

    fig = go.Figure(
        go.Heatmap(
            z=matrix.to_numpy(dtype=float),
            x=list(states),
            y=list(states),
            colorscale=SEQUENTIAL_SCALE,
            zmin=0,
            zmax=zmax,
            xgap=1,
            ygap=1,
            text=text,
            texttemplate=texttemplate,
            colorbar={"title": cbar},
            hovertemplate=hover,
        )
    )
    fig.update_yaxes(autorange="reversed")
    apply_default_layout(
        fig, xaxis={"title": "next period"}, yaxis={"title": "current period"}
    )

    disp = matrix.reset_index()
    disp = disp.rename(columns={disp.columns[0]: "from \\ to"})
    gt = GT(disp, rowname_col="from \\ to").tab_header(title=caption)
    if normalize == "row":
        gt = gt.fmt_percent(columns=list(states), decimals=0)
    else:
        gt = gt.fmt_integer(columns=list(states))

    return TransitionMatrixResult(
        df=matrix, counts=counts, fig=fig, gt=gt, states=tuple(states)
    )


def prepare_within_persistence(
    df: pd.DataFrame,
    var: str,
    *,
    entity: str | None = None,
    time: str | None = None,
    lag: int = 1,
    demean: bool = True,
    alpha: float | None = None,
) -> WithinPersistenceResult:
    """Within-unit serial correlation: this period's value against the previous one.

    Parameters
    ----------
    df
        Panel data frame.
    var
        Numeric variable.
    entity, time
        Panel identifiers (default to those declared via :func:`expdpy.set_panel`).
    lag
        Lag (in periods) for the comparison (default 1). Only consecutive observed periods
        exactly ``lag`` apart are paired.
    demean
        If ``True`` (default), remove each unit's mean first, isolating the *within-unit*
        serial correlation (the part fixed-effects models exploit).
    alpha
        Marker opacity. Defaults to a sample-size-based value.

    Returns
    -------
    WithinPersistenceResult
        ``df`` (lagged pairs), ``fig``, ``rho`` (within serial correlation), ``slope`` (AR
        fit), ``n_pairs`` and ``demeaned``.

    Examples
    --------
    ```python
    import expdpy as ex
    from expdpy.data import load_kuznets

    df = load_kuznets()
    ex.prepare_within_persistence(df, var="gini_regional", entity="country", time="year").fig
    ```
    """
    df = ensure_dataframe(df)
    entity, time = resolve_panel(
        df, entity, time, require_entity=True, require_time=True
    )
    assert entity is not None and time is not None
    if var not in df.columns:
        raise ValueError("var needs to be in df")
    if not pdt.is_numeric_dtype(df[var]):
        raise ValueError(f"var ({var!r}) needs to be numeric")

    sub = df[[entity, time, var]].dropna().copy()
    sub[time] = _try_convert_ts_id(sub[time])[0]
    if sub[time].nunique() < 2:
        raise ValueError("within persistence needs at least two periods")
    sub["_v"] = (
        sub[var] - sub.groupby(entity, observed=True)[var].transform("mean")
        if demean
        else sub[var]
    )
    rank = {p: i for i, p in enumerate(sorted(sub[time].unique()))}
    sub["_rank"] = sub[time].map(rank)
    sub = sub.sort_values([entity, "_rank"])
    grp = sub.groupby(entity, observed=True)
    sub["_lag_v"] = grp["_v"].shift(lag)
    sub["_lag_rank"] = grp["_rank"].shift(lag)

    valid = (sub["_rank"] - sub["_lag_rank"] == lag) & sub["_lag_v"].notna()
    pairs = sub[valid]
    if pairs.empty:
        raise ValueError("no consecutive pairs found")
    lag_v = pairs["_lag_v"].to_numpy(dtype=float)
    v = pairs["_v"].to_numpy(dtype=float)
    rho = (
        float(np.corrcoef(lag_v, v)[0, 1])
        if np.ptp(lag_v) > 0 and np.ptp(v) > 0
        else float("nan")
    )
    slope = float(np.polyfit(lag_v, v, 1)[0]) if np.ptp(lag_v) > 0 else float("nan")
    n_pairs = len(pairs)
    alpha = alpha if alpha is not None else _default_alpha(n_pairs)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=lag_v,
            y=v,
            mode="markers",
            marker={"color": color_for(0), "opacity": alpha, "size": 6},
            name="pairs",
            customdata=pairs[entity].astype(str),
            hovertemplate="%{customdata}<br>prev=%{x:.4g}, now=%{y:.4g}<extra></extra>",
        )
    )
    lo, hi = float(min(lag_v.min(), v.min())), float(max(lag_v.max(), v.max()))
    fig.add_trace(
        go.Scatter(
            x=[lo, hi],
            y=[lo, hi],
            mode="lines",
            line={"color": "rgba(120,120,120,0.6)", "dash": "dash"},
            name="45°",
            hoverinfo="skip",
        )
    )
    if not np.isnan(slope):
        intercept = float(np.polyfit(lag_v, v, 1)[1])
        xs = np.array([lag_v.min(), lag_v.max()])
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=slope * xs + intercept,
                mode="lines",
                line={"color": color_for(3), "width": 2.5},
                name="AR fit",
                hoverinfo="skip",
            )
        )
    suffix = " (within, demeaned)" if demean else ""
    apply_default_layout(
        fig,
        xaxis={"title": f"{var} at t-{lag}{suffix}"},
        yaxis={"title": f"{var} at t{suffix}"},
    )
    out = pairs[[entity, time]].copy()
    out["lag_value"] = lag_v
    out["value"] = v
    return WithinPersistenceResult(
        df=out.reset_index(drop=True),
        fig=fig,
        rho=rho,
        slope=slope,
        n_pairs=n_pairs,
        demeaned=demean,
    )
