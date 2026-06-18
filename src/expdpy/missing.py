"""Missing-value heatmap across the panel's time dimension."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from expdpy._theme import apply_default_layout
from expdpy._validation import ensure_dataframe, numeric_logical_columns

__all__ = ["prepare_missing_values_graph"]


def prepare_missing_values_graph(
    df: pd.DataFrame,
    ts_id: str,
    *,
    no_factors: bool = False,
    binary: bool = False,
) -> go.Figure:
    """Heatmap of missing-value frequency by variable and time period.

    Parameters
    ----------
    df
        Data frame containing the data.
    ts_id
        Column indicating the time dimension (coercible to an ordered factor); must not
        contain missing values.
    no_factors
        If ``True``, limit the plot to numeric/logical variables.
    binary
        If ``True``, show only whether values are missing (any) rather than the fraction.

    Returns
    -------
    plotly.graph_objects.Figure
        The missing-values heatmap.
    """
    df = ensure_dataframe(df)
    if ts_id not in df.columns:
        raise ValueError("'ts_id' needs to be present in data frame 'df'")
    if df[ts_id].isna().any():
        raise ValueError("'ts_id' must not contain missing values")

    levels = sorted(df[ts_id].dropna().unique(), key=str)
    if no_factors:
        cols = [c for c in numeric_logical_columns(df) if c != ts_id]
    else:
        cols = [c for c in df.columns if c != ts_id]

    grouped = df.groupby(ts_id, observed=True)
    z = np.empty((len(levels), len(cols)), dtype=float)
    level_index = {lvl: i for i, lvl in enumerate(levels)}
    for col_j, col in enumerate(cols):
        if binary:
            frac = grouped[col].apply(lambda s: float(s.isna().any()))
        else:
            frac = grouped[col].apply(lambda s: float(s.isna().mean()))
        for lvl, val in frac.items():
            z[level_index[lvl], col_j] = val

    if binary:
        fig = go.Figure(
            go.Heatmap(
                z=z,
                x=cols,
                y=[str(lvl) for lvl in levels],
                colorscale=[[0.0, "#d9d9d9"], [1.0, "#08519c"]],
                zmin=0,
                zmax=1,
                colorbar={
                    "title": "Missing?",
                    "tickvals": [0, 1],
                    "ticktext": ["No", "Yes"],
                },
                xgap=1,
                ygap=1,
            )
        )
    else:
        fig = go.Figure(
            go.Heatmap(
                z=z,
                x=cols,
                y=[str(lvl) for lvl in levels],
                colorscale="Blues",
                zmin=0,
                zmax=1,
                colorbar={"title": "% missing", "tickformat": ".0%"},
                xgap=1,
                ygap=1,
                hovertemplate="%{x} @ %{y}: %{z:.1%} missing<extra></extra>",
            )
        )
    apply_default_layout(fig, xaxis={"tickangle": 90}, yaxis={"title": ts_id})
    return fig
