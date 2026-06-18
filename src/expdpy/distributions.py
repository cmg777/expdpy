"""Distribution graphs: histogram and simple count bar chart."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from expdpy._theme import apply_default_layout
from expdpy._types import BarChartResult, HistogramResult
from expdpy._validation import ensure_dataframe

__all__ = ["prepare_bar_chart", "prepare_histogram"]


def prepare_histogram(
    df: pd.DataFrame,
    var: str,
    *,
    bins: int = 30,
) -> HistogramResult:
    """Histogram of a numeric variable.

    Parameters
    ----------
    df
        Data frame containing ``var``.
    var
        Numeric column to bin.
    bins
        Number of equal-width bins.

    Returns
    -------
    HistogramResult
        ``df`` (columns ``bin_left``, ``bin_right``, ``count``) and the Plotly ``fig``.

    Examples
    --------
    Basic — a 30-bin histogram of a numeric variable:

    ```python
    import expdpy as ex
    from expdpy.data import load_kuznets

    df = load_kuznets()
    ex.prepare_histogram(df, "gini_regional").fig
    ```

    Advanced — finer bins, with the bin/count table from ``.df``:

    ```python
    result = ex.prepare_histogram(df, "gdp_pc", bins=50)
    result.fig
    result.df.head()
    ```
    """
    df = ensure_dataframe(df)
    if var not in df.columns:
        raise ValueError("var needs to be in df")
    values = df[var].to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    counts, edges = np.histogram(values, bins=bins)
    out = pd.DataFrame(
        {"bin_left": edges[:-1], "bin_right": edges[1:], "count": counts}
    )
    centers = (edges[:-1] + edges[1:]) / 2.0
    fig = go.Figure(
        go.Bar(x=centers, y=counts, width=np.diff(edges), marker_color="#1f77b4")
    )
    apply_default_layout(fig, xaxis={"title": var}, yaxis={"title": "Count"}, bargap=0)
    return HistogramResult(df=out, fig=fig)


def prepare_bar_chart(
    df: pd.DataFrame,
    var: str,
    *,
    order_by_count: bool = False,
    color: str = "#4682b4",
) -> BarChartResult:
    """Bar chart of category counts for a (typically categorical) variable.

    Parameters
    ----------
    df
        Data frame containing ``var``.
    var
        Column whose value counts are charted.
    order_by_count
        If ``True``, bars are ordered by descending count; otherwise by category.
    color
        Bar fill color.

    Returns
    -------
    BarChartResult
        ``df`` (columns ``var`` and ``count``) and the Plotly ``fig``.

    Examples
    --------
    Basic — category counts of a (typically categorical) variable:

    ```python
    import expdpy as ex
    from expdpy.data import load_kuznets

    df = load_kuznets()
    ex.prepare_bar_chart(df, "continent").fig
    ```

    Advanced — order bars by descending count and set a custom color:

    ```python
    ex.prepare_bar_chart(df, "continent", order_by_count=True, color="red").fig
    ```
    """
    df = ensure_dataframe(df)
    if var not in df.columns:
        raise ValueError("var needs to be in df")
    counts = df[var].value_counts(dropna=False)
    if not order_by_count:
        counts = counts.sort_index()
    out = counts.rename_axis(var).reset_index(name="count")
    fig = go.Figure(go.Bar(x=out[var].astype(str), y=out["count"], marker_color=color))
    apply_default_layout(fig, xaxis={"title": var}, yaxis={"title": "Count"})
    return BarChartResult(df=out, fig=fig)
