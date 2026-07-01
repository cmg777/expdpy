"""Gapminder-style animated bubble scatter: one frame per period, with a time slider.

``explore_animated_scatter_plot`` turns a panel into a moving picture — each unit is a bubble
whose position (``x``, ``y``), size and color update across periods, driven by a play button
and a time slider. Axis ranges and bubble scaling are held fixed across frames so motion is
comparable over time. The aesthetics mirror :func:`expdpy.explore_scatter_plot`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pandas.api import types as pdt

from expdpy._animate import (
    fmt_period,
    global_range,
    play_controls,
    time_slider,
)
from expdpy._common import entity_display_map as _entity_display_map
from expdpy._labels import resolve_label
from expdpy._panel import resolve_entity_name, resolve_panel
from expdpy._theme import active_sequential_scale, apply_default_layout, color_for
from expdpy._types import AnimatedScatterResult
from expdpy._validation import ensure_dataframe

__all__ = ["explore_animated_scatter_plot"]


def explore_animated_scatter_plot(
    df: pd.DataFrame,
    x: str,
    y: str,
    *,
    size: str | None = None,
    color: str | None = None,
    entity: str | None = None,
    time: str | None = None,
    alpha: float = 0.8,
    log_x: bool = False,
    log_y: bool = False,
    title: str | None = None,
    subtitle: str | None = None,
) -> AnimatedScatterResult:
    """Animated bubble scatter of ``y`` against ``x`` over time (a Gapminder-style view).

    Each period becomes a frame; a play button and slider step through them. Bubbles are sized
    by ``size`` and colored by ``color`` (discrete → one series per level with a legend;
    numeric → a colorbar). Axis ranges and the bubble size scale are fixed across all frames so
    movement is comparable from period to period.

    Parameters
    ----------
    df
        Data frame containing the variables (a panel with a time identifier).
    x, y
        Column names for the axes (both must be numeric).
    size
        Optional numeric column mapped to bubble area.
    color
        Optional column mapped to bubble color (numeric → colorbar, otherwise discrete series).
    entity
        Cross-sectional (unit) identifier, used for hover labels. Defaults to the panel
        ``entity`` declared via :func:`expdpy.set_panel`.
    time
        Time identifier that drives the animation frames. Defaults to the panel ``time``; an
        explicit value or a declared panel is required.
    alpha
        Bubble opacity (default ``0.8``).
    log_x, log_y
        Put the corresponding axis on a log scale (default ``False``). Handy for
        heavy-tailed quantities such as GDP per capita — the canonical Gapminder x-axis.
    title
        Optional figure title.

    Returns
    -------
    AnimatedScatterResult
        ``df`` (the complete-case frame plotted) and ``fig`` (the animated Plotly figure).

    Raises
    ------
    ValueError
        If no time identifier can be resolved, or ``x`` / ``y`` are non-numeric.

    Examples
    --------
    The classic Gapminder animation — income on the x-axis, life expectancy on the y-axis,
    bubbles sized by population and colored by continent, moving across the years:

    ```python
    import expdpy as ex
    from expdpy.data import load_gapminder, load_gapminder_data_def

    df = ex.set_labels(load_gapminder(), load_gapminder_data_def(), set_panel=True)
    ex.explore_animated_scatter_plot(
        df, x="gdpPercap", y="lifeExp", size="pop", color="continent"
    ).fig
    ```
    """
    df = ensure_dataframe(df)
    entity, time = resolve_panel(df, entity, time)
    if time is None:
        raise ValueError(
            "an animated scatter needs a time id (pass time= or call set_panel(df, time=...))"
        )
    for axis_name, axis_col in (("x", x), ("y", y)):
        if not pdt.is_numeric_dtype(df[axis_col]):
            raise ValueError(f"{axis_name} ({axis_col!r}) needs to be numeric")

    x_label, y_label = resolve_label(df, x), resolve_label(df, y)
    color_label = resolve_label(df, color) if color else None
    time_label = resolve_label(df, time)
    disp = _entity_display_map(df, entity, resolve_entity_name(df)) if entity else {}

    cols = list(dict.fromkeys(c for c in (x, y, size, color, entity, time) if c))
    sub = df[cols].dropna().copy()
    if sub.empty:
        raise ValueError("no complete observations to plot")

    periods = sorted(sub[time].unique())
    x_range = global_range(sub[x], log=log_x)
    y_range = global_range(sub[y], log=log_y)

    discrete = bool(color) and not pdt.is_numeric_dtype(sub[color])
    levels = sorted(sub[color].astype(str).unique()) if discrete else []
    size_ref = None
    if size:
        peak = float(np.nanmax(sub[size].to_numpy(dtype=float)))
        size_ref = 2.0 * peak / (42.0**2) if peak > 0 else 1.0
    cmin = float(np.nanmin(sub[color])) if (color and not discrete) else None
    cmax = float(np.nanmax(sub[color])) if (color and not discrete) else None

    def _marker(part: pd.DataFrame, *, fixed_color: str | None = None) -> dict:
        m: dict = {
            "opacity": alpha,
            "line": {"color": "rgba(255,255,255,0.5)", "width": 0.5},
        }
        if size_ref is not None:
            m.update(
                size=part[size].to_numpy(dtype=float),
                sizemode="area",
                sizeref=size_ref,
                sizemin=3,
            )
        if fixed_color is not None:
            m["color"] = fixed_color
        elif color and not discrete:
            m.update(
                color=part[color].to_numpy(dtype=float),
                colorscale=active_sequential_scale(),
                showscale=True,
                cmin=cmin,
                cmax=cmax,
                colorbar={"title": color_label},
            )
        return m

    def _scatter(part: pd.DataFrame, name: str, marker: dict, show: bool) -> go.Scatter:
        text = part[entity].map(lambda u: disp.get(str(u), str(u))) if entity else None
        hover = (
            f"<b>%{{text}}</b><br>{x_label}=%{{x}}<br>{y_label}=%{{y}}<extra></extra>"
            if entity
            else f"{x_label}=%{{x}}<br>{y_label}=%{{y}}<extra></extra>"
        )
        return go.Scatter(
            x=part[x].to_numpy(dtype=float),
            y=part[y].to_numpy(dtype=float),
            mode="markers",
            name=name,
            marker=marker,
            text=text,
            hovertemplate=hover,
            showlegend=show,
        )

    def _traces(period: object) -> list[go.Scatter]:
        part = sub[sub[time] == period]
        if discrete:
            out = []
            for idx, lvl in enumerate(levels):
                p = part[part[color].astype(str) == lvl]
                out.append(
                    _scatter(p, lvl, _marker(p, fixed_color=color_for(idx)), True)
                )
            return out
        return [_scatter(part, y_label, _marker(part), False)]

    frames = [go.Frame(data=_traces(p), name=fmt_period(p)) for p in periods]
    fig = go.Figure(data=_traces(periods[0]), frames=frames)

    xaxis = {"title": x_label, "range": x_range}
    yaxis = {"title": y_label, "range": y_range}
    if log_x:
        xaxis["type"] = "log"
    if log_y:
        yaxis["type"] = "log"
    apply_default_layout(
        fig,
        xaxis=xaxis,
        yaxis=yaxis,
        title=title or f"{y_label} vs {x_label} over {time_label}",
        subtitle=subtitle,
    )
    fig.update_layout(
        sliders=time_slider([fmt_period(p) for p in periods], time_label=time_label),
        updatemenus=play_controls(),
        showlegend=discrete,
    )
    return AnimatedScatterResult(df=sub, fig=fig)
