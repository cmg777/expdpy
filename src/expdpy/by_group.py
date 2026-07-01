"""Grouped graphs: bar of a statistic, trend by group, violin, and box/strip distributions."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Literal

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pandas.api import types as pdt

from expdpy._animate import global_range, retheme_px_animation
from expdpy._common import (
    se as _se,
)
from expdpy._common import (
    sorted_levels as _sorted_levels,
)
from expdpy._common import (
    try_convert_ts_id as _try_convert_ts_id,
)
from expdpy._common import (
    xaxis as _xaxis,
)
from expdpy._labels import resolve_label
from expdpy._panel import resolve_panel
from expdpy._theme import active_colorway, apply_default_layout, color_for
from expdpy._types import (
    BoxPlotResult,
    ByGroupBarGraphResult,
    ByGroupTrendGraphResult,
    ByGroupViolinResult,
    StripPlotResult,
)
from expdpy._validation import (
    ExpdpyWarning,
    drop_missing,
    ensure_dataframe,
    require_columns,
)

__all__ = [
    "explore_bar_plot_by_group",
    "explore_trend_plot_by_group",
    "explore_violin_plot_by_group",
    "explore_box_plot",
    "explore_strip_plot",
]


def explore_bar_plot_by_group(
    df: pd.DataFrame,
    by_var: str,
    var: str,
    stat_fun: Callable[[np.ndarray], float] = np.nanmean,
    *,
    order_by_stat: bool = False,
    color: str | None = None,
    title: str | None = None,
    subtitle: str | None = None,
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
    from expdpy.data import load_kuznets, load_kuznets_data_def

    df = ex.set_labels(load_kuznets(), load_kuznets_data_def(), set_panel=True)
    ex.explore_bar_plot_by_group(df, "continent", "gini_regional").fig
    ```

    Advanced — a different statistic, bars ordered by it, a custom color, and the
    per-group values from ``.df``:

    ```python
    import numpy as np
    import expdpy as ex
    from expdpy.data import load_kuznets, load_kuznets_data_def

    df = ex.set_labels(load_kuznets(), load_kuznets_data_def(), set_panel=True)
    result = ex.explore_bar_plot_by_group(
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

    by_label = resolve_label(df, by_var)
    var_label = resolve_label(df, var)
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
            hovertemplate=f"{by_label}=%{{y}}<br>{var_label}=%{{x:.4g}}<extra></extra>",
        )
    )
    apply_default_layout(
        fig,
        xaxis={"title": var_label},
        yaxis={
            "title": by_label,
            "categoryorder": "array",
            "categoryarray": display_order[::-1],
        },
    )
    if title is not None or subtitle is not None:
        apply_default_layout(fig, title=title, subtitle=subtitle)
    return ByGroupBarGraphResult(df=grouped, fig=fig)


def explore_trend_plot_by_group(
    df: pd.DataFrame,
    group_var: str,
    var: str,
    *,
    time: str | None = None,
    points: bool = True,
    error_bars: bool = False,
    title: str | None = None,
    subtitle: str | None = None,
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
    from expdpy.data import load_kuznets, load_kuznets_data_def

    df = ex.set_labels(load_kuznets(), load_kuznets_data_def(), set_panel=True)
    ex.explore_trend_plot_by_group(
        df, group_var="continent", var="gini_regional"
    ).fig
    ```

    Advanced — add standard-error bars and drop the per-observation markers:

    ```python
    import expdpy as ex
    from expdpy.data import load_kuznets, load_kuznets_data_def

    df = ex.set_labels(load_kuznets(), load_kuznets_data_def(), set_panel=True)
    ex.explore_trend_plot_by_group(
        df, group_var="continent", var="gini_regional",
        error_bars=True, points=False,
    ).fig
    ```
    """
    df = ensure_dataframe(df)
    _entity, time = resolve_panel(df, None, time, require_time=True)
    assert time is not None  # require_time=True guarantees this
    require_columns(df, [time, group_var, var], where="explore_trend_plot_by_group")

    time_label = resolve_label(df, time)
    group_label = resolve_label(df, group_var)
    var_label = resolve_label(df, var)

    sub = drop_missing(
        df[[time, group_var, var]],
        [time, group_var, var],
        func="explore_trend_plot_by_group",
    )
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
                hovertemplate=f"{group_label}=%{{fullData.name}}<br>{time_label}=%{{x}}<br>"
                "mean=%{y:.4g}<extra></extra>",
            )
        )
    apply_default_layout(
        fig,
        xaxis=_xaxis(time, ordered, ts_conv, title=time_label),
        yaxis={"title": var_label},
        legend_title_text=group_label,
        hovermode="x unified",
    )
    if title is not None or subtitle is not None:
        apply_default_layout(fig, title=title, subtitle=subtitle)
    return ByGroupTrendGraphResult(df=gf, fig=fig)


def explore_violin_plot_by_group(
    df: pd.DataFrame,
    by_var: str,
    var: str,
    *,
    order_by_mean: bool = False,
    group_on_y: bool = True,
    title: str | None = None,
    subtitle: str | None = None,
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
    from expdpy.data import load_kuznets, load_kuznets_data_def

    df = ex.set_labels(load_kuznets(), load_kuznets_data_def(), set_panel=True)
    ex.explore_violin_plot_by_group(df, "continent", "gini_regional").fig
    ```

    Advanced — order groups by their mean and orient the violins vertically:

    ```python
    import expdpy as ex
    from expdpy.data import load_kuznets, load_kuznets_data_def

    df = ex.set_labels(load_kuznets(), load_kuznets_data_def(), set_panel=True)
    ex.explore_violin_plot_by_group(
        df, "continent", "gini_regional", order_by_mean=True, group_on_y=False
    ).fig
    ```
    """
    df = ensure_dataframe(df)
    require_columns(df, [by_var, var], where="explore_violin_plot_by_group")
    by_label = resolve_label(df, by_var)
    var_label = resolve_label(df, var)
    sub = drop_missing(
        df[[by_var, var]], [by_var, var], func="explore_violin_plot_by_group"
    )
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
                "title": by_label,
                "categoryorder": "array",
                "categoryarray": levels[::-1],
            },
            xaxis={"title": var_label},
        )
    else:
        fig.update_layout(
            xaxis={
                "title": by_label,
                "categoryorder": "array",
                "categoryarray": levels,
            },
            yaxis={"title": var_label},
        )
    if title is not None or subtitle is not None:
        apply_default_layout(fig, title=title, subtitle=subtitle)
    return ByGroupViolinResult(df=sub, fig=fig)


def _grouped_distribution(
    kind: Literal["box", "strip"],
    df: pd.DataFrame,
    by_var: str,
    var: str,
    *,
    time: str | None,
    order_by_mean: bool,
    group_on_y: bool,
    log: bool,
    entity: str | None,
    title: str | None,
    subtitle: str | None,
    px_extra: dict,
    max_units: int | None = None,
    sample_seed: int = 0,
) -> tuple[go.Figure, pd.DataFrame]:
    """Build a box/strip distribution-by-group figure, animated over ``time`` when resolvable."""
    df = ensure_dataframe(df)
    _entity, time = resolve_panel(df, entity, time)
    time_cols = [time] if time else []
    require_columns(df, [by_var, var, *time_cols], where=f"explore_{kind}_plot")
    if not pdt.is_numeric_dtype(df[var]):
        raise ValueError(f"var ({var!r}) needs to be numeric")

    by_label = resolve_label(df, by_var)
    var_label = resolve_label(df, var)
    time_label = resolve_label(df, time) if time else None

    sub = drop_missing(
        df[[by_var, var, *time_cols]],
        [by_var, var, *time_cols],
        func=f"explore_{kind}_plot",
    )
    if sub.empty:
        raise ValueError(
            f"explore_{kind}_plot: no complete observations of by_var and var"
        )
    sub = sub.copy()
    sub[by_var] = sub[by_var].astype(str)

    # Optional point sampling (strip), before plotting so every frame samples one universe.
    if max_units and len(sub) > max_units:
        n_before = len(sub)
        rng = np.random.default_rng(sample_seed)
        keep_idx = rng.choice(sub.index.to_numpy(), size=max_units, replace=False)
        sub = sub.loc[keep_idx]
        warnings.warn(
            f"explore_{kind}_plot: showing {max_units} of {n_before} points "
            f"(sampled, seed={sample_seed})",
            ExpdpyWarning,
            stacklevel=3,
        )

    levels = _sorted_levels(sub[by_var])
    if order_by_mean:
        means = sub.groupby(by_var, observed=True)[var].mean()
        levels = [str(g) for g in means.sort_values(ascending=False).index]

    # px's range_x/range_y are in DATA units (px log-transforms them itself when log_*=True),
    # so pin the range in data units — for a log axis, pad in log space then map back.
    if log:
        _log_lo, _log_hi = global_range(sub[var], log=True)
        value_range = [10.0**_log_lo, 10.0**_log_hi]
    else:
        value_range = global_range(sub[var])
    px_kwargs: dict = {
        "color": by_var,
        "color_discrete_sequence": active_colorway(),
        "category_orders": {by_var: levels},
        "labels": {by_var: by_label, var: var_label},
        **px_extra,
    }
    if time:
        # px orders frames by first appearance, so sort by time to get ascending frames.
        sub = sub.assign(**{time: _try_convert_ts_id(sub[time])[0]}).sort_values(time)
        px_kwargs["animation_frame"] = time
        px_kwargs["labels"][time] = time_label

    if group_on_y:
        px_kwargs.update(y=by_var, x=var, range_x=value_range)
        if log:
            px_kwargs["log_x"] = True
    else:
        px_kwargs.update(x=by_var, y=var, range_y=value_range)
        if log:
            px_kwargs["log_y"] = True

    maker = px.box if kind == "box" else px.strip
    fig = maker(sub, **px_kwargs)
    retheme_px_animation(
        fig, time_label=time_label or "", title=title, subtitle=subtitle
    )
    fig.update_layout(showlegend=False)

    out_cols = [*time_cols, by_var, var]
    return fig, sub[out_cols].reset_index(drop=True)


def explore_box_plot(
    df: pd.DataFrame,
    by_var: str,
    var: str,
    *,
    time: str | None = None,
    order_by_mean: bool = False,
    group_on_y: bool = True,
    log: bool = False,
    points: Literal["outliers", "all", "suspectedoutliers"] | bool = "outliers",
    entity: str | None = None,
    title: str | None = None,
    subtitle: str | None = None,
) -> BoxPlotResult:
    """Box plots of ``var`` across ``by_var`` groups, animated over ``time`` when available.

    One box per group summarises the distribution (median, quartiles, whiskers). On a panel each
    period becomes a frame, stepped through by a play button and a time slider, with the value
    axis pinned so shifts are comparable; without a resolvable time id a single static set of
    boxes is drawn.

    Parameters
    ----------
    df
        Data frame containing the grouping factor, the numeric variable and (optionally) a time
        column.
    by_var
        Grouping column.
    var
        Numeric variable whose distribution is shown.
    time
        Time identifier that drives the animation. Defaults to the panel ``time`` declared via
        :func:`expdpy.set_panel`; when neither is available the boxes are static.
    order_by_mean
        If ``True``, groups are ordered by descending group mean.
    group_on_y
        If ``True`` (default), boxes are horizontal (groups on the y-axis).
    log
        Put the value axis on a log scale (default ``False``).
    points
        Which underlying points to overlay: ``"outliers"`` (default), ``"all"``,
        ``"suspectedoutliers"`` or ``False`` for none.
    entity
        Cross-sectional id (accepted for panel parity; only used to resolve the panel ``time``).

    Returns
    -------
    BoxPlotResult
        ``df`` (the complete-case frame plotted) and ``fig`` (the Plotly box plot).

    Examples
    --------
    Regional inequality by continent, animated across the years:

    ```python
    import expdpy as ex
    from expdpy.data import load_kuznets, load_kuznets_data_def

    df = ex.set_labels(load_kuznets(), load_kuznets_data_def(), set_panel=True)
    ex.explore_box_plot(df, "continent", "gini_regional", time="year").fig
    ```
    """
    fig, plotted = _grouped_distribution(
        "box",
        df,
        by_var,
        var,
        time=time,
        order_by_mean=order_by_mean,
        group_on_y=group_on_y,
        log=log,
        entity=entity,
        title=title,
        subtitle=subtitle,
        px_extra={"points": points},
    )
    return BoxPlotResult(df=plotted, fig=fig)


def explore_strip_plot(
    df: pd.DataFrame,
    by_var: str,
    var: str,
    *,
    time: str | None = None,
    order_by_mean: bool = False,
    group_on_y: bool = True,
    log: bool = False,
    alpha: float = 0.6,
    max_units: int | None = 2000,
    sample_seed: int = 0,
    entity: str | None = None,
    title: str | None = None,
    subtitle: str | None = None,
) -> StripPlotResult:
    """Strip plots of ``var`` across ``by_var`` groups, animated over ``time`` when available.

    A jittered cloud of every observation within each group — complementary to
    :func:`explore_box_plot`, showing the raw points rather than a summary. On a panel each
    period becomes a frame with the value axis pinned; without a resolvable time id the clouds
    are static. Large samples are thinned to ``max_units`` points so the animation stays
    responsive.

    Parameters
    ----------
    df
        Data frame containing the grouping factor, the numeric variable and (optionally) a time
        column.
    by_var
        Grouping column.
    var
        Numeric variable whose observations are shown.
    time
        Time identifier that drives the animation. Defaults to the panel ``time``; when neither
        is available the clouds are static.
    order_by_mean
        If ``True``, groups are ordered by descending group mean.
    group_on_y
        If ``True`` (default), groups are on the y-axis.
    log
        Put the value axis on a log scale (default ``False``).
    alpha
        Point opacity (default ``0.6``).
    max_units
        Cap on the number of plotted points; above it a seeded random sample is drawn and an
        :class:`expdpy.ExpdpyWarning` is emitted. ``None`` disables sampling.
    sample_seed
        Seed for the point sample.
    entity
        Cross-sectional id (accepted for panel parity; only used to resolve the panel ``time``).

    Returns
    -------
    StripPlotResult
        ``df`` (the complete-case frame plotted) and ``fig`` (the Plotly strip plot).

    Examples
    --------
    Every region's inequality by continent, animated across the years:

    ```python
    import expdpy as ex
    from expdpy.data import load_kuznets, load_kuznets_data_def

    df = ex.set_labels(load_kuznets(), load_kuznets_data_def(), set_panel=True)
    ex.explore_strip_plot(df, "continent", "gini_regional", time="year").fig
    ```
    """
    fig, plotted = _grouped_distribution(
        "strip",
        df,
        by_var,
        var,
        time=time,
        order_by_mean=order_by_mean,
        group_on_y=group_on_y,
        log=log,
        entity=entity,
        title=title,
        subtitle=subtitle,
        px_extra={},
        max_units=max_units,
        sample_seed=sample_seed,
    )
    fig.update_traces(marker={"opacity": alpha})
    return StripPlotResult(df=plotted, fig=fig)
