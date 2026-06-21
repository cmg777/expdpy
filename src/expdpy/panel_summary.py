"""Within/between variation: the xtsum table and the within-vs-between scatter.

These are the cornerstone of panel exploration — they show how much of a variable's variation
(or of a relationship) is **across units** versus **over time within a unit**.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from great_tables import GT
from pandas.api import types as pdt

from expdpy._labels import resolve_label
from expdpy._panel import resolve_panel
from expdpy._panel_math import panel_decompose
from expdpy._theme import apply_default_layout, color_for
from expdpy._types import WithinBetweenScatterResult, XtsumTableResult
from expdpy._validation import ensure_dataframe, numeric_logical_columns
from expdpy.scatter import _default_alpha

__all__ = ["explore_scatter_plot_within_between", "explore_xtsum_table"]


def explore_xtsum_table(
    df: pd.DataFrame,
    var: Sequence[str] | None = None,
    *,
    entity: str | None = None,
    time: str | None = None,
    digits: int = 3,
    caption: str = "Within/Between Variation (xtsum)",
) -> XtsumTableResult:
    """Decompose each variable's variation into overall / between / within (Stata ``xtsum``).

    For every numeric variable the table reports the overall mean and standard deviation, the
    **between** standard deviation (across unit means), and the **within** standard deviation
    (variation over time inside a unit), plus the number of observations, units and the
    average number of periods per unit.

    Parameters
    ----------
    df
        Panel data frame.
    var
        Variables to summarise. Defaults to all numeric/logical columns except the
        identifiers.
    entity
        Cross-sectional (unit) identifier. Defaults to the panel ``entity`` declared via
        :func:`expdpy.set_panel`.
    time
        Time identifier (used only to exclude it from the default ``var`` list). Defaults to
        the panel ``time``.
    digits
        Decimals for the formatted statistics.
    caption
        Great Tables header title.

    Returns
    -------
    XtsumTableResult
        ``df`` (long within/between frame) and ``gt`` (the Great Tables object).

    Examples
    --------
    ```python
    import expdpy as ex
    from expdpy.data import load_kuznets

    df = load_kuznets()
    ex.explore_xtsum_table(df, var=["gini_regional", "log_gdp_pc"], entity="country").gt
    ```
    """
    df = ensure_dataframe(df)
    entity, time = resolve_panel(df, entity, time, require_entity=True)
    assert entity is not None  # require_entity=True guarantees this
    if df[entity].nunique() < 2:
        raise ValueError("xtsum needs at least two entities")
    if var is None:
        var = [c for c in numeric_logical_columns(df) if c not in (entity, time)]
    else:
        var = list(var)
        missing = [v for v in var if v not in df.columns]
        if missing:
            raise ValueError("var names need to be in df")
    if not var:
        raise ValueError("no numeric variables to summarise")

    rows: list[dict] = []
    for v in var:
        d = panel_decompose(df[v], df[entity])
        if d["n_obs"] == 0:
            warnings.warn(
                f"xtsum: variable {v!r} has no observations; skipping", stacklevel=2
            )
            continue
        for comp, sd, lo, hi, mean in (
            (
                "overall",
                d["overall_sd"],
                d["overall_min"],
                d["overall_max"],
                d["overall_mean"],
            ),
            ("between", d["between_sd"], d["between_min"], d["between_max"], np.nan),
            ("within", d["within_sd"], d["within_min"], d["within_max"], np.nan),
        ):
            rows.append(
                {
                    "variable": v,
                    "component": comp,
                    "mean": mean,
                    "sd": sd,
                    "min": lo,
                    "max": hi,
                    "n_obs": d["n_obs"],
                    "n_entities": d["n_entities"],
                    "t_bar": d["t_bar"],
                }
            )
    if not rows:
        raise ValueError("no variables with observations to summarise")
    out = pd.DataFrame(rows)

    disp = out.rename(
        columns={
            "component": "Statistic",
            "mean": "Mean",
            "sd": "Std. dev.",
            "min": "Min.",
            "max": "Max.",
            "n_obs": "N",
            "n_entities": "n units",
            "t_bar": "T-bar",
        }
    )
    disp["Statistic"] = disp["Statistic"].str.capitalize()
    # Group rows by the readable variable label (the returned .df keeps raw names).
    disp["variable"] = disp["variable"].map(lambda v: resolve_label(df, v))
    # N / n / T-bar belong to the variable as a whole — show them only on the Overall row.
    not_overall = disp["Statistic"] != "Overall"
    disp.loc[not_overall, ["N", "n units", "T-bar"]] = np.nan
    gt = (
        GT(disp, rowname_col="Statistic", groupname_col="variable")
        .tab_header(title=caption)
        .fmt_number(
            columns=["Mean", "Std. dev.", "Min.", "Max.", "T-bar"], decimals=digits
        )
        .fmt_integer(columns=["N", "n units"])
        .sub_missing(missing_text="")
        .tab_source_note(
            "Overall = pooled. Between = across unit means. Within = over time inside a unit "
            "(deviations from the unit mean, recentered on the grand mean)."
        )
    )
    return XtsumTableResult(df=out, gt=gt)


def _fit_line(
    xv: np.ndarray, yv: np.ndarray
) -> tuple[float, float, tuple[np.ndarray, np.ndarray] | None]:
    """Return ``(slope, intercept, (xs, ys))`` of an OLS fit, or ``nan`` / ``None`` if undefined."""
    xv = np.asarray(xv, dtype=float)
    yv = np.asarray(yv, dtype=float)
    m = np.isfinite(xv) & np.isfinite(yv)
    xv, yv = xv[m], yv[m]
    if xv.size < 2 or np.ptp(xv) == 0:
        return float("nan"), float("nan"), None
    slope, intercept = np.polyfit(xv, yv, 1)
    xs = np.array([xv.min(), xv.max()])
    return float(slope), float(intercept), (xs, slope * xs + intercept)


def explore_scatter_plot_within_between(
    df: pd.DataFrame,
    x: str,
    y: str,
    *,
    entity: str | None = None,
    time: str | None = None,
    show: Literal["overlay", "pooled", "between", "within"] = "overlay",
    alpha: float | None = None,
) -> WithinBetweenScatterResult:
    """Scatter that decomposes the ``x``-``y`` relationship into between and within parts.

    Three views are drawn (switchable via a dropdown): the **pooled** cloud, the **between**
    cloud of unit means, and the **within** cloud of unit-demeaned deviations (recentered on
    the grand means). Their fitted slopes show how a pooled association blends a cross-unit
    and an over-time relationship.

    Parameters
    ----------
    df
        Panel data frame.
    x, y
        Numeric column names for the axes.
    entity
        Cross-sectional (unit) identifier. Defaults to the panel ``entity``.
    time
        Time identifier (carried into the hover data). Defaults to the panel ``time``.
    show
        Which view is visible initially: ``"overlay"`` (all), ``"pooled"``, ``"between"`` or
        ``"within"``.
    alpha
        Marker opacity for the pooled/within clouds. Defaults to a sample-size-based value.

    Returns
    -------
    WithinBetweenScatterResult
        ``df`` (long plotted frame), ``fig`` and the three slopes ``slope_pooled`` /
        ``slope_between`` / ``slope_within``.

    Examples
    --------
    ```python
    import expdpy as ex
    from expdpy.data import load_kuznets

    df = load_kuznets()
    ex.explore_scatter_plot_within_between(
        df, x="log_gdp_pc", y="gini_regional", entity="country"
    ).fig
    ```
    """
    df = ensure_dataframe(df)
    entity, time = resolve_panel(df, entity, time, require_entity=True)
    assert entity is not None  # require_entity=True guarantees this
    for axis_name, col in (("x", x), ("y", y)):
        if not pdt.is_numeric_dtype(df[col]):
            raise ValueError(f"{axis_name} ({col!r}) needs to be numeric")
    x_label = resolve_label(df, x)
    y_label = resolve_label(df, y)
    cols = list(dict.fromkeys([entity, x, y, *([time] if time else [])]))
    sub = df[cols].dropna(subset=[entity, x, y])
    if sub[entity].nunique() < 2:
        raise ValueError("within/between scatter needs at least two entities")
    n = len(sub)
    alpha = alpha if alpha is not None else _default_alpha(n)

    # Between data: unit means. Within data: deviations recentered on the grand mean.
    em = sub.groupby(entity, observed=True)[[x, y]].mean()
    gx, gy = float(sub[x].mean()), float(sub[y].mean())
    wx = sub[x] - sub.groupby(entity, observed=True)[x].transform("mean") + gx
    wy = sub[y] - sub.groupby(entity, observed=True)[y].transform("mean") + gy

    bp, _ip, line_p = _fit_line(sub[x].to_numpy(), sub[y].to_numpy())
    bb, _ib, line_b = _fit_line(em[x].to_numpy(), em[y].to_numpy())
    bw, _iw, line_w = _fit_line(wx.to_numpy(), wy.to_numpy())

    time_vals = sub[time] if time else pd.Series([np.nan] * n, index=sub.index)
    long = pd.concat(
        [
            pd.DataFrame(
                {
                    "component": "pooled",
                    "x": sub[x].to_numpy(),
                    "y": sub[y].to_numpy(),
                    "entity": sub[entity].astype(str).to_numpy(),
                    "time": time_vals.to_numpy(),
                }
            ),
            pd.DataFrame(
                {
                    "component": "between",
                    "x": em[x].to_numpy(),
                    "y": em[y].to_numpy(),
                    "entity": em.index.astype(str).to_numpy(),
                    "time": np.nan,
                }
            ),
            pd.DataFrame(
                {
                    "component": "within",
                    "x": wx.to_numpy(),
                    "y": wy.to_numpy(),
                    "entity": sub[entity].astype(str).to_numpy(),
                    "time": time_vals.to_numpy(),
                }
            ),
        ],
        ignore_index=True,
    )

    fig = go.Figure()
    # Trace order matters for the visibility map below.
    fig.add_trace(
        go.Scatter(
            x=sub[x],
            y=sub[y],
            mode="markers",
            name="pooled",
            marker={"color": color_for(9), "opacity": alpha, "size": 6},
            customdata=sub[entity].astype(str),
            hovertemplate="%{customdata}<br>x=%{x:.4g}, y=%{y:.4g}<extra>pooled</extra>",
        )
    )
    fig.add_trace(_line_trace(line_p, color_for(9), "pooled fit"))
    fig.add_trace(
        go.Scatter(
            x=em[x],
            y=em[y],
            mode="markers",
            name="between (unit means)",
            marker={
                "color": color_for(1),
                "size": 9,
                "line": {"color": "white", "width": 0.5},
            },
            customdata=em.index.astype(str),
            hovertemplate="%{customdata}<br>mean x=%{x:.4g}, mean y=%{y:.4g}"
            "<extra>between</extra>",
        )
    )
    fig.add_trace(_line_trace(line_b, color_for(1), "between fit", dash="dash"))
    fig.add_trace(
        go.Scatter(
            x=wx,
            y=wy,
            mode="markers",
            name="within (demeaned)",
            marker={"color": color_for(2), "opacity": alpha, "size": 6},
            hovertemplate="x=%{x:.4g}, y=%{y:.4g}<extra>within</extra>",
        )
    )
    fig.add_trace(_line_trace(line_w, color_for(2), "within fit"))

    vis = {
        "overlay": [True] * 6,
        "pooled": [True, True, False, False, False, False],
        "between": [False, False, True, True, False, False],
        "within": [False, False, False, False, True, True],
    }
    for trace, visible in zip(fig.data, vis[show], strict=True):
        trace.visible = True if visible else "legendonly"

    buttons = [
        {
            "label": label.capitalize(),
            "method": "update",
            "args": [{"visible": [v or "legendonly" for v in vis[label]]}],
        }
        for label in ("overlay", "pooled", "between", "within")
    ]
    apply_default_layout(
        fig,
        xaxis={"title": x_label},
        yaxis={"title": y_label},
        updatemenus=[
            {
                "type": "dropdown",
                "x": 1.0,
                "y": 1.15,
                "xanchor": "right",
                "buttons": buttons,
            }
        ],
    )
    fig.add_annotation(
        text=(f"pooled β={_fmt(bp)} · between β={_fmt(bb)} · within β={_fmt(bw)}"),
        xref="paper",
        yref="paper",
        x=0,
        y=1.10,
        showarrow=False,
        font={"size": 13},
    )
    return WithinBetweenScatterResult(
        df=long, fig=fig, slope_pooled=bp, slope_between=bb, slope_within=bw
    )


def _line_trace(
    line: tuple[np.ndarray, np.ndarray] | None,
    color: str,
    name: str,
    *,
    dash: str | None = None,
) -> go.Scatter:
    """Build a fitted-line Scatter trace (empty when the fit is undefined)."""
    if line is None:
        return go.Scatter(x=[], y=[], mode="lines", name=name, line={"color": color})
    xs, ys = line
    return go.Scatter(
        x=xs,
        y=ys,
        mode="lines",
        name=name,
        line={"color": color, "width": 2.5, "dash": dash}
        if dash
        else {"color": color, "width": 2.5},
        hoverinfo="skip",
    )


def _fmt(value: float) -> str:
    """Compact slope label for the annotation."""
    return "n/a" if value != value else f"{value:.3g}"  # value != value catches NaN
