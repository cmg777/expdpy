"""Spaghetti plot: every unit's trajectory over time, with a central-tendency overlay."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pandas.api import types as pdt
from plotly.subplots import make_subplots

from expdpy._labels import resolve_label
from expdpy._panel import resolve_panel
from expdpy._theme import apply_default_layout, blank_rangeslider, color_for
from expdpy._types import SpaghettiGraphResult
from expdpy._validation import ensure_dataframe
from expdpy.scatter import _default_alpha
from expdpy.trends import _try_convert_ts_id, _xaxis

__all__ = ["explore_spaghetti_plot"]


def _draw_unit_lines(
    fig: go.Figure,
    sub: pd.DataFrame,
    *,
    entity: str,
    time: str,
    var: str,
    ordered: bool,
    highlight: list[str],
    line_alpha: float,
    overlay: str,
    entity_label: str,
    time_label: str,
    var_label: str,
    row: int | None = None,
    col: int | None = None,
    show_overlay_legend: bool = False,
) -> None:
    """Draw the faint per-unit lines, the highlighted units and the overlay into ``fig``."""
    faint = {"color": f"rgba(78,121,167,{line_alpha:.3f})", "width": 1}
    add_kw = {"row": row, "col": col} if row is not None else {}
    hl_idx = 0
    for uid, part in sub.groupby(entity, observed=True):
        part = part.sort_values(time)
        x = part[time].astype(str) if ordered else part[time]
        is_hl = str(uid) in highlight
        fig.add_trace(
            go.Scatter(
                x=x,
                y=part[var],
                mode="lines",
                name=str(uid),
                line=({"color": color_for(hl_idx), "width": 2.5} if is_hl else faint),
                showlegend=is_hl,
                hovertemplate=f"{entity_label}=%{{fullData.name}}<br>{time_label}=%{{x}}<br>"
                f"{var_label}=%{{y:.4g}}<extra></extra>",
                hoverinfo="skip" if not is_hl else None,
            ),
            **add_kw,
        )
        if is_hl:
            hl_idx += 1

    if overlay != "none":
        agg = "median" if overlay == "median" else "mean"
        central = sub.groupby(time, observed=True)[var].agg(agg).reset_index()
        central = central.sort_values(time)
        x = central[time].astype(str) if ordered else central[time]
        fig.add_trace(
            go.Scatter(
                x=x,
                y=central[var],
                mode="lines",
                name=f"{agg} ({var_label})",
                line={"color": "#1a1a1a", "width": 3},
                showlegend=show_overlay_legend,
                hovertemplate=(
                    f"{agg}<br>{time_label}=%{{x}}<br>{var_label}=%{{y:.4g}}<extra></extra>"
                ),
            ),
            **add_kw,
        )


def explore_spaghetti_plot(
    df: pd.DataFrame,
    var: str,
    *,
    entity: str | None = None,
    time: str | None = None,
    overlay: Literal["mean", "median", "none"] = "mean",
    highlight: Sequence[str] | None = None,
    alpha: float | None = None,
    max_units: int | None = 150,
    facet: str | None = None,
    sample_seed: int = 0,
) -> SpaghettiGraphResult:
    """Plot every unit's trajectory of ``var`` over time, with a central-tendency overlay.

    Parameters
    ----------
    df
        Panel data frame.
    var
        Numeric variable to plot.
    entity
        Cross-sectional (unit) identifier. Defaults to the panel ``entity``.
    time
        Time identifier. Defaults to the panel ``time``.
    overlay
        Bold overlay line: ``"mean"`` (default), ``"median"`` or ``"none"``.
    highlight
        Units to draw in saturated colour on top of the faint backdrop.
    alpha
        Opacity of the faint per-unit lines. Defaults to a sample-size-based value.
    max_units
        Cap on the number of units drawn; above it a seeded sample is shown (highlighted
        units are always kept) and a warning reports how many were dropped. ``None`` draws
        all units.
    facet
        Optional grouping column; when given, draws one small-multiple panel per level.
    sample_seed
        Seed for the unit subsample when ``max_units`` is exceeded.

    Returns
    -------
    SpaghettiGraphResult
        ``df`` (plotted long frame), ``fig``, ``n_units`` (in the data) and ``n_shown``.

    Examples
    --------
    ```python
    import expdpy as ex
    from expdpy.data import load_kuznets

    df = load_kuznets()
    ex.explore_spaghetti_plot(df, var="gini_regional", entity="country", time="year").fig
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
    if facet is not None and facet not in df.columns:
        raise ValueError("facet needs to be in df")

    # Resolve display labels before slicing (column selection drops df.attrs).
    entity_label = resolve_label(df, entity)
    time_label = resolve_label(df, time)
    var_label = resolve_label(df, var)
    facet_label = resolve_label(df, facet) if facet else None

    cols = list(dict.fromkeys([entity, time, var, *([facet] if facet else [])]))
    sub = df[cols].dropna(subset=[entity, time, var])
    if sub.empty:
        raise ValueError(f"var ({var!r}) has no complete observations")
    ts_conv, ordered = _try_convert_ts_id(sub[time])
    sub = sub.assign(**{time: ts_conv})

    units = list(sub[entity].unique())
    n_units = len(units)
    highlight = [str(h) for h in highlight] if highlight else []
    unknown = [h for h in highlight if h not in {str(u) for u in units}]
    if unknown:
        warnings.warn(f"spaghetti: highlight units not found: {unknown}", stacklevel=2)

    if max_units and n_units > max_units:
        rng = np.random.default_rng(sample_seed)
        keep = [u for u in units if str(u) in highlight]
        pool = [u for u in units if str(u) not in highlight]
        n_sample = max(0, max_units - len(keep))
        chosen = (
            list(rng.choice(pool, size=min(n_sample, len(pool)), replace=False))
            if n_sample and pool
            else []
        )
        shown = set(keep) | set(chosen)
        sub = sub[sub[entity].isin(shown)]
        warnings.warn(
            f"spaghetti: showing {len(shown)} of {n_units} units; "
            f"{n_units - len(shown)} sampled out (seed={sample_seed})",
            stacklevel=2,
        )
    n_shown = int(sub[entity].nunique())
    base_alpha = alpha if alpha is not None else _default_alpha(n_shown)
    line_alpha = float(min(0.5, max(0.05, base_alpha)))

    if facet is None:
        fig = go.Figure()
        _draw_unit_lines(
            fig,
            sub,
            entity=entity,
            time=time,
            var=var,
            ordered=ordered,
            highlight=highlight,
            line_alpha=line_alpha,
            overlay=overlay,
            entity_label=entity_label,
            time_label=time_label,
            var_label=var_label,
            show_overlay_legend=True,
        )
        xaxis = _xaxis(time, ordered, ts_conv, title=time_label)
        if not ordered:
            xaxis["rangeslider"] = blank_rangeslider(fig)
        apply_default_layout(fig, xaxis=xaxis, yaxis={"title": var_label})
    else:
        levels = sorted(sub[facet].dropna().astype(str).unique())
        ncols = min(3, len(levels)) or 1
        nrows = int(np.ceil(len(levels) / ncols))
        fig = make_subplots(
            rows=nrows,
            cols=ncols,
            subplot_titles=[f"{facet_label} = {lvl}" for lvl in levels],
            shared_yaxes=True,
        )
        for i, lvl in enumerate(levels):
            r, c = divmod(i, ncols)
            part = sub[sub[facet].astype(str) == lvl]
            _draw_unit_lines(
                fig,
                part,
                entity=entity,
                time=time,
                var=var,
                ordered=ordered,
                highlight=highlight,
                line_alpha=line_alpha,
                overlay=overlay,
                entity_label=entity_label,
                time_label=time_label,
                var_label=var_label,
                row=r + 1,
                col=c + 1,
                show_overlay_legend=(i == 0),
            )
        apply_default_layout(fig, yaxis={"title": var_label})

    return SpaghettiGraphResult(
        df=sub[[entity, time, var]].reset_index(drop=True),
        fig=fig,
        n_units=n_units,
        n_shown=n_shown,
    )
