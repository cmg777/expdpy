"""Grouped graphs: bar of a statistic, trend by group, and violin distributions."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from expdpy._theme import apply_default_layout, color_for
from expdpy._types import ByGroupBarGraphResult, ByGroupTrendGraphResult
from expdpy._validation import ensure_dataframe
from expdpy.trends import _se, _try_convert_ts_id, _xaxis

__all__ = [
    "prepare_by_group_bar_graph",
    "prepare_by_group_trend_graph",
    "prepare_by_group_violin_graph",
]


def prepare_by_group_bar_graph(
    df: pd.DataFrame,
    by_var: str,
    var: str,
    stat_fun: Callable[[np.ndarray], float] = np.nanmean,
    *,
    order_by_stat: bool = False,
    color: str = "red",
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
        Bar fill color.

    Returns
    -------
    ByGroupBarGraphResult
        ``df`` (columns ``by_var`` and ``stat_<var>``) and the Plotly ``fig``.
    """
    df = ensure_dataframe(df)
    stat_col = f"stat_{var}"
    grouped = (
        df[[by_var, var]]
        .groupby(by_var, sort=False, observed=True)[var]
        .apply(lambda s: float(stat_fun(s.dropna().to_numpy())))
        .reset_index()
    )
    grouped.columns = [by_var, stat_col]

    if order_by_stat:
        grouped = grouped.sort_values(stat_col, ascending=False)
    display_order = [str(g) for g in grouped[by_var]]  # top -> bottom

    fig = go.Figure(
        go.Bar(
            x=grouped[stat_col],
            y=grouped[by_var].astype(str),
            orientation="h",
            marker_color=color,
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
    ts_id: str,
    group_var: str,
    var: str,
    *,
    points: bool = True,
    error_bars: bool = False,
) -> ByGroupTrendGraphResult:
    """Line-plot the mean of ``var`` over time, one line per ``group_var`` level.

    Parameters
    ----------
    df
        Data frame containing the time index, grouping factor and variable.
    ts_id
        Time-series identifier column.
    group_var
        Grouping column.
    var
        Numeric variable to plot.
    points
        Whether to mark each observation with a point.
    error_bars
        Whether to draw standard-error bars (``mean +/- se``).

    Returns
    -------
    ByGroupTrendGraphResult
        ``df`` (columns ``ts_id``, ``group_var``, ``mean``, ``se``) and the Plotly ``fig``.
    """
    df = ensure_dataframe(df)
    for col in (ts_id, group_var, var):
        if col not in df.columns:
            raise ValueError(f"{col} needs to be in df")

    sub = df[[ts_id, group_var, var]].dropna()
    ts_conv, ordered = _try_convert_ts_id(sub[ts_id])
    sub = sub.assign(**{ts_id: ts_conv})
    sub[group_var] = sub[group_var].astype(str)

    gf = (
        sub.groupby([ts_id, group_var], observed=True)[var]
        .agg(mean="mean", se=_se)
        .reset_index()
    )

    mode = "lines+markers" if points else "lines"
    fig = go.Figure()
    for idx, level in enumerate(sorted(gf[group_var].unique())):
        part = gf[gf[group_var] == level].sort_values(ts_id)
        x = part[ts_id].astype(str) if ordered else part[ts_id]
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
            )
        )
    apply_default_layout(
        fig,
        xaxis=_xaxis(ts_id, ordered, ts_conv),
        yaxis={"title": var},
        legend_title_text=group_var,
    )
    return ByGroupTrendGraphResult(df=gf, fig=fig)


def prepare_by_group_violin_graph(
    df: pd.DataFrame,
    by_var: str,
    var: str,
    *,
    order_by_mean: bool = False,
    group_on_y: bool = True,
) -> go.Figure:
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
    plotly.graph_objects.Figure
        The violin figure.
    """
    df = ensure_dataframe(df)
    sub = df[[by_var, var]].dropna()
    sub[by_var] = sub[by_var].astype(str)

    levels = list(dict.fromkeys(sub[by_var]))
    if order_by_mean:
        means = sub.groupby(by_var, observed=True)[var].mean()
        levels = list(means.sort_values(ascending=False).index)

    fig = go.Figure()
    for idx, level in enumerate(levels):
        vals = sub.loc[sub[by_var] == level, var]
        if group_on_y:
            fig.add_trace(
                go.Violin(
                    x=vals,
                    name=str(level),
                    orientation="h",
                    fillcolor=color_for(idx),
                    line_color=color_for(idx),
                )
            )
        else:
            fig.add_trace(
                go.Violin(
                    y=vals,
                    name=str(level),
                    fillcolor=color_for(idx),
                    line_color=color_for(idx),
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
    return fig
