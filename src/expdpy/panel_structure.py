"""Panel structure diagnostics: balance/gaps summary, presence grid and value heatmap."""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from great_tables import GT
from pandas.api import types as pdt

from expdpy._labels import resolve_label
from expdpy._panel import resolve_panel
from expdpy._theme import DIVERGING_SCALE, SEQUENTIAL_SCALE, apply_default_layout
from expdpy._types import PanelStructureResult, ValueHeatmapResult
from expdpy._validation import ensure_dataframe
from expdpy.trends import _try_convert_ts_id

__all__ = ["explore_panel_structure", "explore_value_heatmap"]

_PRESENT_SCALE = [[0.0, "#EDEDED"], [1.0, "#4E79A7"]]


def _even_sample(labels: list, max_units: int | None, *, what: str) -> list:
    """Evenly subsample ``labels`` to at most ``max_units`` (keeping order); warn if cut."""
    n = len(labels)
    if not max_units or n <= max_units:
        return labels
    positions = sorted(set(np.linspace(0, n - 1, max_units).astype(int).tolist()))
    warnings.warn(
        f"{what}: showing {len(positions)} of {n} units (evenly sampled)", stacklevel=3
    )
    return [labels[i] for i in positions]


def explore_panel_structure(
    df: pd.DataFrame,
    *,
    entity: str | None = None,
    time: str | None = None,
    var: str | None = None,
    max_units: int | None = 200,
    caption: str = "Panel Structure",
) -> PanelStructureResult:
    """Summarise the panel's balance and coverage, with a unit-by-period presence grid.

    A general panel-completeness diagnostic. (For *treatment* structure on a staggered-adoption
    design, see :func:`expdpy.analyze_panel_view`.)

    Parameters
    ----------
    df
        Panel data frame.
    entity
        Cross-sectional (unit) identifier. Defaults to the panel ``entity``.
    time
        Time identifier. Defaults to the panel ``time``.
    var
        Optional variable: when given, a cell counts as "present" only if ``var`` is
        non-missing there (rather than merely a row existing).
    max_units
        Cap on the number of units drawn in the presence grid (evenly sampled above it).
    caption
        Great Tables header for the summary.

    Returns
    -------
    PanelStructureResult
        ``df_summary`` (tidy statistics), ``df_grid`` (full presence matrix), ``gt`` and the
        presence-grid ``fig``.

    Examples
    --------
    ```python
    import expdpy as ex
    from expdpy.data import load_kuznets

    df = load_kuznets()
    res = ex.explore_panel_structure(df, entity="country", time="year")
    res.gt
    res.fig
    ```
    """
    df = ensure_dataframe(df)
    entity, time = resolve_panel(
        df, entity, time, require_entity=True, require_time=True
    )
    assert entity is not None and time is not None
    if var is not None and var not in df.columns:
        raise ValueError("var needs to be in df")

    entity_label = resolve_label(df, entity)
    time_label = resolve_label(df, time)
    var_label = resolve_label(df, var) if var else None

    keep = [entity, time, *([var] if var else [])]
    work = df[keep].dropna(subset=[entity, time]).copy()
    work[time] = _try_convert_ts_id(work[time])[0]
    work["_present"] = work[var].notna().astype(int) if var else 1

    grid = work.pivot_table(
        index=entity, columns=time, values="_present", aggfunc="max"
    )
    grid = grid.reindex(columns=sorted(grid.columns))
    presence = grid.fillna(0).to_numpy() == 1

    n_units, n_periods = grid.shape[0], grid.shape[1]
    obs_per_unit = presence.sum(axis=1)
    n_obs = int(presence.sum())
    gaps = 0
    for row in presence:
        idx = np.where(row)[0]
        if idx.size:
            gaps += int((idx[-1] - idx[0] + 1) - idx.size)
    min_obs, max_obs = int(obs_per_unit.min()), int(obs_per_unit.max())
    balanced = bool(min_obs == n_periods and max_obs == n_periods and gaps == 0)

    summary = pd.DataFrame(
        {
            "statistic": [
                "units",
                "periods",
                "observations",
                "balanced",
                "internal gaps",
                "min obs per unit",
                "max obs per unit",
            ],
            "value": [n_units, n_periods, n_obs, balanced, gaps, min_obs, max_obs],
        }
    )
    disp = summary.copy()
    disp["value"] = [
        f"{n_units:,}",
        f"{n_periods:,}",
        f"{n_obs:,}",
        "Yes" if balanced else "No",
        f"{gaps:,}",
        f"{min_obs:,}",
        f"{max_obs:,}",
    ]
    gt = (
        GT(disp, rowname_col="statistic")
        .tab_header(title=caption)
        .tab_source_note(
            "A cell is 'present' when the unit is observed in that period"
            + (f" with a non-missing {var_label}." if var else ".")
            + " An interior gap is a missing period between a unit's first and last."
        )
    )

    # Presence grid: sort units by first-observed period, then by completeness (most on top).
    first_seen = np.array(
        [np.where(row)[0][0] if row.any() else n_periods for row in presence]
    )
    order = np.lexsort((-obs_per_unit, first_seen))
    sorted_units = [str(grid.index[i]) for i in order]
    z_full = presence.astype(int)[order]
    shown_units = _even_sample(sorted_units, max_units, what="panel_structure")
    shown_mask = [u in set(shown_units) for u in sorted_units]
    z = z_full[np.array(shown_mask)]
    y = [u for u, keep_u in zip(sorted_units, shown_mask, strict=True) if keep_u]

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=[str(c) for c in grid.columns],
            y=y,
            colorscale=_PRESENT_SCALE,
            zmin=0,
            zmax=1,
            xgap=1,
            ygap=1,
            colorbar={
                "title": "",
                "tickvals": [0, 1],
                "ticktext": ["absent", "present"],
            },
            hovertemplate=(
                f"{entity_label}=%{{y}}<br>{time_label}=%{{x}}<br>%{{z}}<extra></extra>"
            ),
        )
    )
    apply_default_layout(
        fig,
        xaxis={"title": time_label, "tickangle": -40},
        yaxis={"title": entity_label},
    )
    return PanelStructureResult(df_summary=summary, df_grid=grid, gt=gt, fig=fig)


def explore_value_heatmap(
    df: pd.DataFrame,
    var: str,
    *,
    entity: str | None = None,
    time: str | None = None,
    standardize: Literal["none", "by_time", "by_entity", "global"] = "none",
    aggfunc: str = "mean",
    max_units: int | None = 200,
    sort_by: Literal["mean", "first_value", "label"] = "mean",
) -> ValueHeatmapResult:
    """Heatmap of a variable over the unit-by-time grid (units by periods, colour = value).

    Parameters
    ----------
    df
        Panel data frame.
    var
        Numeric variable to display.
    entity
        Cross-sectional (unit) identifier. Defaults to the panel ``entity``.
    time
        Time identifier. Defaults to the panel ``time``.
    standardize
        ``"none"`` shows raw values; ``"global"`` z-scores over all cells; ``"by_time"`` /
        ``"by_entity"`` z-score within each period / unit (revealing relative position).
    aggfunc
        Aggregator for duplicate ``(entity, time)`` cells (default ``"mean"``).
    max_units
        Cap on the number of units drawn (evenly sampled above it).
    sort_by
        Row order: by ``"mean"`` (default, descending), ``"first_value"`` or ``"label"``.

    Returns
    -------
    ValueHeatmapResult
        ``df`` (the unit-by-time pivot) and the Plotly ``fig``.

    Examples
    --------
    ```python
    import expdpy as ex
    from expdpy.data import load_kuznets

    df = load_kuznets()
    ex.explore_value_heatmap(
        df, var="gini_regional", entity="country", time="year", standardize="by_time"
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
    if not pdt.is_numeric_dtype(df[var]):
        raise ValueError(f"var ({var!r}) needs to be numeric")

    entity_label = resolve_label(df, entity)
    time_label = resolve_label(df, time)
    var_label = resolve_label(df, var)

    work = df[[entity, time, var]].dropna(subset=[entity, time]).copy()
    work[time] = _try_convert_ts_id(work[time])[0]
    pivot = work.pivot_table(index=entity, columns=time, values=var, aggfunc=aggfunc)
    pivot = pivot.reindex(columns=sorted(pivot.columns))
    if pivot.notna().to_numpy().sum() == 0:
        raise ValueError(f"var ({var!r}) has no values to display")

    if sort_by == "mean":
        pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]
    elif sort_by == "first_value":
        pivot = pivot.loc[
            pivot.bfill(axis=1).iloc[:, 0].sort_values(ascending=False).index
        ]
    else:  # label, numeric-aware
        idx = pivot.index.astype(str)
        num = pd.to_numeric(pd.Series(idx), errors="coerce")
        keys = num.to_numpy() if not num.isna().any() else idx.to_numpy()
        pivot = pivot.iloc[np.argsort(keys, kind="stable")]

    shown = _even_sample(list(pivot.index), max_units, what="value_heatmap")
    pivot = pivot.loc[shown]

    z = pivot.to_numpy(dtype=float)
    diverging = standardize != "none"
    if standardize == "global":
        sd = np.nanstd(z)
        z = (z - np.nanmean(z)) / sd if sd else z
        cbar = "z-score"
    elif standardize == "by_time":
        mu, sd = np.nanmean(z, axis=0), np.nanstd(z, axis=0)
        z = np.divide(z - mu, sd, out=np.zeros_like(z), where=sd != 0)
        cbar = "z (within period)"
    elif standardize == "by_entity":
        mu = np.nanmean(z, axis=1, keepdims=True)
        sd = np.nanstd(z, axis=1, keepdims=True)
        z = np.divide(z - mu, sd, out=np.zeros_like(z), where=sd != 0)
        cbar = "z (within unit)"
    else:
        cbar = var_label

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=[str(c) for c in pivot.columns],
            y=[str(i) for i in pivot.index],
            colorscale=DIVERGING_SCALE if diverging else SEQUENTIAL_SCALE,
            zmid=0 if diverging else None,
            xgap=1,
            ygap=1,
            colorbar={"title": cbar},
            hovertemplate=(
                f"{entity_label}=%{{y}}<br>{time_label}=%{{x}}<br>"
                f"{var_label}=%{{z:.3g}}<extra></extra>"
            ),
        )
    )
    apply_default_layout(
        fig,
        xaxis={"title": time_label, "tickangle": -40},
        yaxis={"title": entity_label},
    )
    return ValueHeatmapResult(df=pivot, fig=fig)
