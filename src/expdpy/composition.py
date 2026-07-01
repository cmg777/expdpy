"""Animated composition views: a treemap and a sunburst of a hierarchy over time.

``explore_treemap_plot`` and ``explore_sunburst_plot`` show how a whole — a total sized by a
value — is divided among a hierarchy of parts (e.g. countries within continents), and, on a
panel, how that composition shifts period by period through a play button and a time slider.

Plotly Express builds each period's figure (``px.treemap`` / ``px.sunburst`` do not accept
``animation_frame``); we assemble the per-period traces into ``go.Frame`` objects and attach the
shared controls from :mod:`expdpy._animate`, so the aesthetics match the rest of the library.
Without a resolvable time id the view degrades to a single static composition.
"""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pandas.api import types as pdt

from expdpy._animate import (
    fmt_period,
    global_color_range,
    play_pause_updatemenus,
    time_slider,
)
from expdpy._common import entity_display_series as _entity_display_series
from expdpy._common import sorted_levels as _sorted_levels
from expdpy._common import try_convert_ts_id as _try_convert_ts_id
from expdpy._labels import resolve_label
from expdpy._panel import resolve_entity_name, resolve_panel
from expdpy._theme import active_colorway, active_sequential_scale, apply_default_layout
from expdpy._types import SunburstPlotResult, TreemapPlotResult
from expdpy._validation import (
    ExpdpyWarning,
    drop_missing,
    ensure_dataframe,
    require_columns,
)

__all__ = ["explore_treemap_plot", "explore_sunburst_plot"]

_MAKERS = {"treemap": px.treemap, "sunburst": px.sunburst}


def _default_title(kind: str, size_label: str | None, time_label: str | None) -> str:
    """Compose a default figure title from the chart kind, value and time labels."""
    noun = "Treemap" if kind == "treemap" else "Sunburst"
    what = f" of {size_label}" if size_label else ""
    when = f" over {time_label}" if time_label else ""
    return f"{noun}{what}{when}"


def _composition_figure(
    kind: Literal["treemap", "sunburst"],
    df: pd.DataFrame,
    *,
    path: list[str] | str | None,
    size: str | None,
    color: str | None,
    entity: str | None,
    time: str | None,
    max_units: int | None,
    sample_seed: int,
    title: str | None,
    subtitle: str | None,
) -> tuple[go.Figure, pd.DataFrame]:
    """Build a treemap/sunburst figure (animated over ``time`` when resolvable) + its frame."""
    df = ensure_dataframe(df)
    entity, time = resolve_panel(df, entity, time)

    if path is None:
        if entity is None:
            raise ValueError(
                f"explore_{kind}_plot needs a hierarchy: pass path=[...] or declare an "
                "entity via set_panel(df, entity=...)"
            )
        path = [entity]
    elif isinstance(path, str):
        path = [path]
    path = list(path)

    cols_needed = [*path, *(c for c in (size, color, time) if c)]
    require_columns(df, cols_needed, where=f"explore_{kind}_plot")
    if size is not None and not pdt.is_numeric_dtype(df[size]):
        raise ValueError(f"size ({size!r}) needs to be numeric")

    # Resolve display labels before slicing (column selection drops df.attrs).
    labels = {c: resolve_label(df, c) for c in dict.fromkeys(cols_needed)}
    size_label = labels.get(size) if size else None
    time_label = resolve_label(df, time) if time else None

    # A "Name (id)" leaf display when the finest level is the bare panel entity.
    leaf = path[-1]
    name_col = resolve_entity_name(df) if leaf == entity else None
    use_display = bool(name_col and name_col != entity and name_col in df.columns)

    keep = list(dict.fromkeys([*cols_needed, *([name_col] if use_display else [])]))
    sub = drop_missing(df[keep], subset=cols_needed, func=f"explore_{kind}_plot")
    if sub.empty:
        raise ValueError(
            f"explore_{kind}_plot: no rows with a complete {path} hierarchy and values"
        )
    sub = sub.copy()

    # Sample distinct leaves down to max_units (stable across periods: sample once, up front).
    n_units = int(sub[leaf].nunique())
    if max_units and n_units > max_units:
        rng = np.random.default_rng(sample_seed)
        chosen = rng.choice(sub[leaf].unique(), size=max_units, replace=False)
        sub = sub[sub[leaf].isin(chosen)]
        warnings.warn(
            f"explore_{kind}_plot: showing {max_units} of {n_units} {leaf!r} units "
            f"(sampled, seed={sample_seed})",
            ExpdpyWarning,
            stacklevel=3,
        )

    plot_path = list(path)
    if use_display:
        # In this branch leaf == entity (a str), so use ``leaf`` for the id column.
        disp = f"{leaf} (display)"
        sub[disp] = _entity_display_series(sub, leaf, name_col)
        labels[disp] = resolve_label(df, leaf)
        plot_path = [*path[:-1], disp]

    px_kwargs: dict = {"path": plot_path, "labels": labels}
    if size is not None:
        px_kwargs["values"] = size

    continuous = bool(color) and pdt.is_numeric_dtype(sub[color])
    if color and continuous:
        px_kwargs.update(
            color=color,
            color_continuous_scale=active_sequential_scale(),
            range_color=global_color_range(sub[color]),
        )
    elif color:
        levels = _sorted_levels(sub[color].astype(str))
        cw = active_colorway()
        sub[color] = sub[color].astype(str)
        px_kwargs.update(
            color=color,
            color_discrete_map={lvl: cw[i % len(cw)] for i, lvl in enumerate(levels)},
        )

    maker = _MAKERS[kind]
    fig_title = title or _default_title(kind, size_label, time_label)

    if time:
        sub = sub.assign(**{time: _try_convert_ts_id(sub[time])[0]})
        periods = sorted(sub[time].unique())
        frames = [
            go.Frame(
                data=maker(sub[sub[time] == p], **px_kwargs).data, name=fmt_period(p)
            )
            for p in periods
        ]
        fig = go.Figure(data=frames[0].data, frames=frames)
        if continuous:
            # Carry the (fixed-range) colorbar from a full-data figure so the scale holds.
            fig.layout.coloraxis = maker(sub, **px_kwargs).layout.coloraxis
        apply_default_layout(fig, title=fig_title, subtitle=subtitle)
        fig.update_layout(
            updatemenus=play_pause_updatemenus(),
            sliders=time_slider(
                [fmt_period(p) for p in periods], time_label=time_label or ""
            ),
        )
    else:
        fig = maker(sub, **px_kwargs)
        apply_default_layout(fig, title=fig_title, subtitle=subtitle)

    return fig, sub.reset_index(drop=True)


def explore_treemap_plot(
    df: pd.DataFrame,
    *,
    path: list[str] | str | None = None,
    size: str | None = None,
    color: str | None = None,
    entity: str | None = None,
    time: str | None = None,
    max_units: int | None = 200,
    sample_seed: int = 0,
    title: str | None = None,
    subtitle: str | None = None,
) -> TreemapPlotResult:
    """Treemap of a hierarchy sized by ``size``, animated over ``time`` when available.

    Nested rectangles show how a total is divided among the levels of ``path`` (e.g. countries
    within continents). On a panel each period becomes a frame, stepped through by a play button
    and a time slider; the color scale is held fixed across frames so shifts are comparable.
    Without a resolvable time id a single static treemap is drawn.

    Parameters
    ----------
    df
        Data frame containing the hierarchy, value and (optionally) time columns.
    path
        Hierarchy columns from root to leaf, e.g. ``["continent", "country"]``. Defaults to the
        panel ``entity`` (a single level) when omitted.
    size
        Numeric column mapped to rectangle area. When omitted, each leaf is sized by its row
        count.
    color
        Optional column mapped to color: numeric → a fixed-range sequential colorbar; otherwise
        a stable discrete palette.
    entity
        Cross-sectional (unit) id. Defaults to the panel ``entity`` declared via
        :func:`expdpy.set_panel`; used to default ``path`` and to label leaves ``"Name (id)"``
        when an ``entity_name`` is declared.
    time
        Time identifier that drives the animation. Defaults to the panel ``time``; when neither
        is available the treemap is static.
    max_units
        Cap on the number of distinct leaf units; above it a seeded random sample is drawn
        (once, so the same leaves appear in every period) and an :class:`expdpy.ExpdpyWarning`
        is emitted. ``None`` disables sampling.
    sample_seed
        Seed for the leaf sample.

    Returns
    -------
    TreemapPlotResult
        ``df`` (the complete-case frame plotted) and ``fig`` (the Plotly treemap).

    Raises
    ------
    ValueError
        If no hierarchy can be resolved, ``size`` is non-numeric, or nothing remains after
        dropping incomplete rows.

    Examples
    --------
    World population by continent and country, colored by life expectancy, animated over the
    years:

    ```python
    import expdpy as ex
    from expdpy.data import load_gapminder, load_gapminder_data_def

    df = ex.set_labels(load_gapminder(), load_gapminder_data_def(), set_panel=True)
    ex.explore_treemap_plot(
        df, path=["continent", "country"], size="pop", color="lifeExp", time="year"
    ).fig
    ```
    """
    fig, plotted = _composition_figure(
        "treemap",
        df,
        path=path,
        size=size,
        color=color,
        entity=entity,
        time=time,
        max_units=max_units,
        sample_seed=sample_seed,
        title=title,
        subtitle=subtitle,
    )
    return TreemapPlotResult(df=plotted, fig=fig)


def explore_sunburst_plot(
    df: pd.DataFrame,
    *,
    path: list[str] | str | None = None,
    size: str | None = None,
    color: str | None = None,
    entity: str | None = None,
    time: str | None = None,
    max_units: int | None = 200,
    sample_seed: int = 0,
    title: str | None = None,
    subtitle: str | None = None,
) -> SunburstPlotResult:
    """Sunburst of a hierarchy sized by ``size``, animated over ``time`` when available.

    The radial counterpart to :func:`explore_treemap_plot`: concentric rings show the nested
    part-to-whole shares of ``path`` (e.g. countries within continents), and on a panel a play
    button and time slider step through the periods with a fixed color scale. Without a
    resolvable time id a single static sunburst is drawn.

    Parameters
    ----------
    df
        Data frame containing the hierarchy, value and (optionally) time columns.
    path
        Hierarchy columns from root to leaf, e.g. ``["continent", "country"]``. Defaults to the
        panel ``entity`` (a single level) when omitted.
    size
        Numeric column mapped to wedge angle. When omitted, each leaf is sized by its row count.
    color
        Optional column mapped to color: numeric → a fixed-range sequential colorbar; otherwise
        a stable discrete palette.
    entity
        Cross-sectional (unit) id. Defaults to the panel ``entity``; used to default ``path``
        and to label leaves ``"Name (id)"`` when an ``entity_name`` is declared.
    time
        Time identifier that drives the animation. Defaults to the panel ``time``; when neither
        is available the sunburst is static.
    max_units
        Cap on the number of distinct leaf units; above it a seeded random sample is drawn and
        an :class:`expdpy.ExpdpyWarning` is emitted. ``None`` disables sampling.
    sample_seed
        Seed for the leaf sample.

    Returns
    -------
    SunburstPlotResult
        ``df`` (the complete-case frame plotted) and ``fig`` (the Plotly sunburst).

    Examples
    --------
    The same hierarchy as a radial sunburst:

    ```python
    import expdpy as ex
    from expdpy.data import load_gapminder, load_gapminder_data_def

    df = ex.set_labels(load_gapminder(), load_gapminder_data_def(), set_panel=True)
    ex.explore_sunburst_plot(
        df, path=["continent", "country"], size="pop", color="lifeExp", time="year"
    ).fig
    ```
    """
    fig, plotted = _composition_figure(
        "sunburst",
        df,
        path=path,
        size=size,
        color=color,
        entity=entity,
        time=time,
        max_units=max_units,
        sample_seed=sample_seed,
        title=title,
        subtitle=subtitle,
    )
    return SunburstPlotResult(df=plotted, fig=fig)
