"""Missing-value heatmap across the panel's time or unit dimension."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from expdpy._common import argsort_levels
from expdpy._common import try_convert_ts_id as _try_convert_ts_id
from expdpy._labels import resolve_label, resolve_labels
from expdpy._panel import resolve_panel
from expdpy._theme import active_sequential_scale, apply_default_layout
from expdpy._types import MissingValuesResult
from expdpy._validation import ensure_dataframe, numeric_logical_columns

__all__ = ["explore_missing_values_plot"]


def _ordered_levels(index: pd.Index, *, as_time: bool) -> np.ndarray:
    """Return a stable sort order for heatmap rows (numeric/time-aware, not lexical)."""
    if as_time:
        conv, _ = _try_convert_ts_id(pd.Series(index))
        return np.argsort(conv.to_numpy(), kind="stable")
    return argsort_levels(index)


def explore_missing_values_plot(
    df: pd.DataFrame,
    *,
    time: str | None = None,
    entity: str | None = None,
    by: Literal["time", "entity"] = "time",
    no_factors: bool = False,
    binary: bool = False,
    title: str | None = None,
    subtitle: str | None = None,
) -> MissingValuesResult:
    """Heatmap of missing-value frequency by variable and panel dimension.

    Parameters
    ----------
    df
        Data frame containing the data.
    time
        Time identifier column. Defaults to the panel ``time`` declared via
        :func:`expdpy.set_panel`. Required when ``by="time"``; must not contain missing
        values.
    entity
        Cross-sectional (unit) identifier column. Defaults to the panel ``entity``. Required
        when ``by="entity"``; must not contain missing values.
    by
        Whether to aggregate missingness over ``"time"`` periods (the default) or over
        ``"entity"`` units.
    no_factors
        If ``True``, limit the plot to numeric/logical variables.
    binary
        If ``True``, show only whether values are missing (any) rather than the fraction.

    Returns
    -------
    MissingValuesResult
        ``df`` (the missingness matrix, rows = levels, columns = variables) and ``fig``.

    Examples
    --------
    Basic — fraction of missing values by variable and year:

    ```python
    import expdpy as ex
    from expdpy.data import load_kuznets, load_kuznets_data_def

    df = ex.set_labels(load_kuznets(), load_kuznets_data_def(), set_panel=True)
    ex.explore_missing_values_plot(df).fig
    ```

    Advanced — missingness by unit, restricted to numeric variables, shown as a flag:

    ```python
    import expdpy as ex
    from expdpy.data import load_kuznets, load_kuznets_data_def

    df = ex.set_labels(load_kuznets(), load_kuznets_data_def(), set_panel=True)
    ex.explore_missing_values_plot(
        df, by="entity", no_factors=True, binary=True
    ).fig
    ```
    """
    df = ensure_dataframe(df)
    if by not in ("time", "entity"):
        raise ValueError("by needs to be 'time' or 'entity'")
    entity, time = resolve_panel(
        df, entity, time, require_time=(by == "time"), require_entity=(by == "entity")
    )
    axis_col = time if by == "time" else entity
    assert axis_col is not None  # guaranteed by resolve_panel's require_* above
    if df[axis_col].isna().any():
        raise ValueError(
            f"the {by} column ({axis_col!r}) must not contain missing values"
        )

    cols = [
        c
        for c in (numeric_logical_columns(df) if no_factors else list(df.columns))
        if c != axis_col
    ]
    if not cols:
        raise ValueError("no variables left to assess for missingness")

    grouped = df.groupby(axis_col, observed=True)[cols]
    if binary:
        mat = grouped.apply(lambda g: g.isna().any()).astype(float)
    else:
        mat = grouped.apply(lambda g: g.isna().mean())
    mat = mat.iloc[_ordered_levels(mat.index, as_time=(by == "time"))]

    axis_label = resolve_label(df, axis_col)
    col_labels = resolve_labels(df, cols)
    y = [str(lvl) for lvl in mat.index]
    z = mat.to_numpy(dtype=float)
    if binary:
        heatmap = go.Heatmap(
            z=z,
            x=col_labels,
            y=y,
            colorscale=[[0.0, "#EDEDED"], [1.0, "#4E79A7"]],
            zmin=0,
            zmax=1,
            colorbar={
                "title": "Missing?",
                "tickvals": [0, 1],
                "ticktext": ["No", "Yes"],
            },
            xgap=1,
            ygap=1,
            hovertemplate=f"%{{x}} @ {axis_label}=%{{y}}: %{{z}}<extra></extra>",
        )
    else:
        heatmap = go.Heatmap(
            z=z,
            x=col_labels,
            y=y,
            colorscale=active_sequential_scale(),
            zmin=0,
            zmax=1,
            colorbar={"title": "% missing", "tickformat": ".0%"},
            xgap=1,
            ygap=1,
            hovertemplate=(
                f"%{{x}} @ {axis_label}=%{{y}}: %{{z:.1%}} missing<extra></extra>"
            ),
        )
    fig = go.Figure(heatmap)
    apply_default_layout(fig, xaxis={"tickangle": -40}, yaxis={"title": axis_label})
    if title is not None or subtitle is not None:
        apply_default_layout(fig, title=title, subtitle=subtitle)
    return MissingValuesResult(df=mat, fig=fig)
