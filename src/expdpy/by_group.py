"""Grouped graphs: bar of a statistic, trend by group, and violin distributions."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from expdpy._panel import resolve_panel
from expdpy._theme import apply_default_layout, color_for
from expdpy._types import (
    ByGroupBarGraphResult,
    ByGroupTrendGraphResult,
    ByGroupViolinResult,
)
from expdpy._validation import ensure_dataframe
from expdpy.trends import _se, _try_convert_ts_id, _xaxis

__all__ = [
    "prepare_by_group_bar_graph",
    "prepare_by_group_trend_graph",
    "prepare_by_group_violin_graph",
]


def _sorted_levels(values: pd.Series) -> list[str]:
    """Return the distinct levels of ``values`` sorted numerically when possible.

    Group labels like ``"2"`` and ``"10"`` must order as 2 < 10, not lexically.
    """
    levels = list(dict.fromkeys(values.astype(str)))
    num = pd.to_numeric(pd.Series(levels), errors="coerce")
    if not num.isna().any():
        return [lvl for _, lvl in sorted(zip(num, levels, strict=True))]
    return sorted(levels)


def prepare_by_group_bar_graph(
    df: pd.DataFrame,
    by_var: str,
    var: str,
    stat_fun: Callable[[np.ndarray], float] = np.nanmean,
    *,
    order_by_stat: bool = False,
    color: str | None = None,
) -> ByGroupBarGraphResult:
    """Bar chart of a statistic of ``var`` computed within each ``by_var`` group.

    Parameters
    ----------
    df
        Data frame containing the grouping factor and the numeric variable.
    by_var
        Grouping column.
    var
        Numeric column to summarise.
    stat_fun
        Statistic applied to the non-missing values of each group. Defaults to
        :func:`numpy.nanmean`. Missing values are dropped before the call, matching R's
        ``na.rm = TRUE``.
    order_by_stat
        If ``True``, bars are ordered by the statistic (largest at the top); otherwise the
        groups keep their order of appearance.
    color
        Bar fill color. Defaults to the primary theme color.

    Returns
    -------
    ByGroupBarGraphResult
        ``df`` (columns ``by_var`` and ``stat_<var>``) and the Plotly ``fig``.

    Examples
    --------
    Basic — mean of a variable within each group:

    ```python
    import expdpy as ex
    from expdpy.data import load_kuznets

    df = load_kuznets()
    ex.prepare_by_group_bar_graph(df, "continent", "gini_regional").fig
    ```

    Advanced — a different statistic, bars ordered by it, a custom color, and the
    per-group values from ``.df``:

    ```python
    import numpy as np

    result = ex.prepare_by_group_bar_graph(
        df, "continent", "gini_regional",
        stat_fun=np.nanmedian, order_by_stat=True, color="#4682b4",
    )
    result.fig
    result.df
    ```
    """
    df = ensure_dataframe(df)
    for col in (by_var, var):
        if col not in df.columns:
            raise ValueError(f"{col} needs to be in df")

    stat_col = f"stat_{var}"

    def _stat(s: pd.Series) -> float:
        vals = s.dropna().to_numpy()
        if vals.size == 0:
            return float("nan")  # avoid nanmean's "empty slice" warning
        return float(stat_fun(vals))

    grouped = (
        df[[by_var, var]]
        .groupby(by_var, sort=False, observed=True)[var]
        .apply(_stat)
        .reset_index()
    )
    grouped.columns = [by_var, stat_col]

    if order_by_stat:
        grouped = grouped.sort_values(stat_col, ascending=False)
    display_order = [str(g) for g in grouped[by_var]]  # top -> bottom

    bar_color = color if color is not None else color_for(0)
    fig = go.Figure(
        go.Bar(
            x=grouped[stat_col],
            y=grouped[by_var].astype(str),
            orientation="h",
            marker={"color": bar_color, "line": {"color": "white", "width": 0.5}},
            hovertemplate=f"{by_var}=%{{y}}<br>{stat_col}=%{{x:.4g}}<extra></extra>",
        )
    )
    apply_default_layout(
        fig,
        xaxis={"title": stat_col},
        yaxis={
            "title": by_var,
            "categoryorder": "array",
            "categoryarray": display_order[::-1],
        },
    )
    return ByGroupBarGraphResult(df=grouped, fig=fig)


def prepare_by_group_trend_graph(
    df: pd.DataFrame,
    group_var: str,
    var: str,
    *,
    time: str | None = None,
    points: bool = True,
    error_bars: bool = False,
) -> ByGroupTrendGraphResult:
    """Line-plot the mean of ``var`` over time, one line per ``group_var`` level.

    Parameters
    ----------
    df
        Data frame containing the time index, grouping factor and variable.
    group_var
        Grouping column.
    var
        Numeric variable to plot.
    time
        Time identifier column. Defaults to the panel ``time`` declared via
        :func:`expdpy.set_panel`.
    points
        Whether to mark each observation with a point.
    error_bars
        Whether to draw standard-error bars (``mean +/- se``).

    Returns
    -------
    ByGroupTrendGraphResult
        ``df`` (columns ``time``, ``group_var``, ``mean``, ``se``) and the Plotly ``fig``.

    Examples
    --------
    Basic — one line per group, tracking a variable's mean over time:

    ```python
    import expdpy as ex
    from expdpy.data import load_kuznets

    df = load_kuznets()
    ex.prepare_by_group_trend_graph(
        df, group_var="continent", var="gini_regional", time="year"
    ).fig
    ```

    Advanced — add standard-error bars and drop the per-observation markers:

    ```python
    ex.prepare_by_group_trend_graph(
        df, group_var="continent", var="gini_regional", time="year",
        error_bars=True, points=False,
    ).fig
    ```
    """
    df = ensure_dataframe(df)
    _entity, time = resolve_panel(df, None, time, require_time=True)
    assert time is not None  # require_time=True guarantees this
    for col in (time, group_var, var):
        if col not in df.columns:
            raise ValueError(f"{col} needs to be in df")

    sub = df[[time, group_var, var]].dropna()
    ts_conv, ordered = _try_convert_ts_id(sub[time])
    sub = sub.assign(**{time: ts_conv})
    sub[group_var] = sub[group_var].astype(str)

    gf = (
        sub.groupby([time, group_var], observed=True)[var]
        .agg(mean="mean", se=_se)
        .reset_index()
    )

    mode = "lines+markers" if points else "lines"
    fig = go.Figure()
    for idx, level in enumerate(_sorted_levels(gf[group_var])):
        part = gf[gf[group_var] == level].sort_values(time)
        x = part[time].astype(str) if ordered else part[time]
        err = (
            {"type": "data", "array": part["se"], "visible": True}
            if error_bars
            else None
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=part["mean"],
                error_y=err,
                mode=mode,
                name=str(level),
                line={"color": color_for(idx)},
                hovertemplate=f"{group_var}=%{{fullData.name}}<br>{time}=%{{x}}<br>"
                "mean=%{y:.4g}<extra></extra>",
            )
        )
    apply_default_layout(
        fig,
        xaxis=_xaxis(time, ordered, ts_conv),
        yaxis={"title": var},
        legend_title_text=group_var,
        hovermode="x unified",
    )
    return ByGroupTrendGraphResult(df=gf, fig=fig)


def prepare_by_group_violin_graph(
    df: pd.DataFrame,
    by_var: str,
    var: str,
    *,
    order_by_mean: bool = False,
    group_on_y: bool = True,
) -> ByGroupViolinResult:
    """Violin plots of ``var`` distribution across ``by_var`` groups.

    Parameters
    ----------
    df
        Data frame containing the grouping factor and the numeric variable.
    by_var
        Grouping column.
    var
        Numeric variable whose distribution is shown.
    order_by_mean
        If ``True``, groups are ordered by descending group mean.
    group_on_y
        If ``True`` (default), violins are oriented horizontally (groups on the y-axis).

    Returns
    -------
    ByGroupViolinResult
        ``df`` (the complete-case ``[by_var, var]`` frame) and the Plotly ``fig``.

    Examples
    --------
    Basic — distribution of a variable across groups:

    ```python
    import expdpy as ex
    from expdpy.data import load_kuznets

    df = load_kuznets()
    ex.prepare_by_group_violin_graph(df, "continent", "gini_regional").fig
    ```

    Advanced — order groups by their mean and orient the violins vertically:

    ```python
    ex.prepare_by_group_violin_graph(
        df, "continent", "gini_regional", order_by_mean=True, group_on_y=False
    ).fig
    ```
    """
    df = ensure_dataframe(df)
    for col in (by_var, var):
        if col not in df.columns:
            raise ValueError(f"{col} needs to be in df")
    sub = df[[by_var, var]].dropna()
    if sub.empty:
        raise ValueError("no complete observations of by_var and var")
    sub[by_var] = sub[by_var].astype(str)

    levels = list(dict.fromkeys(sub[by_var]))
    if order_by_mean:
        means = sub.groupby(by_var, observed=True)[var].mean()
        levels = list(means.sort_values(ascending=False).index)

    fig = go.Figure()
    for idx, level in enumerate(levels):
        vals = sub.loc[sub[by_var] == level, var]
        axis = {"x": vals, "orientation": "h"} if group_on_y else {"y": vals}
        fig.add_trace(
            go.Violin(
                name=str(level),
                fillcolor=color_for(idx),
                line_color=color_for(idx),
                box_visible=True,
                points="outliers",
                **axis,
            )
        )
    fig.update_traces(opacity=0.7, meanline_visible=True)
    apply_default_layout(fig, showlegend=False)
    if group_on_y:
        fig.update_layout(
            yaxis={
                "title": by_var,
                "categoryorder": "array",
                "categoryarray": levels[::-1],
            },
            xaxis={"title": var},
        )
    else:
        fig.update_layout(
            xaxis={"title": by_var, "categoryorder": "array", "categoryarray": levels},
            yaxis={"title": var},
        )
    return ByGroupViolinResult(df=sub, fig=fig)
