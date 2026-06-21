"""Distribution graphs: histogram and simple count bar chart."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pandas.api import types as pdt

from expdpy._labels import resolve_label
from expdpy._theme import apply_default_layout, color_for
from expdpy._types import BarChartResult, HistogramResult
from expdpy._validation import ensure_dataframe

__all__ = ["explore_bar_plot", "explore_histogram"]


def explore_histogram(
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
    ex.explore_histogram(df, "gini_regional").fig
    ```

    Advanced — finer bins, with the bin/count table from ``.df``:

    ```python
    result = ex.explore_histogram(df, "gdp_pc", bins=50)
    result.fig
    result.df.head()
    ```
    """
    df = ensure_dataframe(df)
    if var not in df.columns:
        raise ValueError("var needs to be in df")
    if not pdt.is_numeric_dtype(df[var]):
        raise ValueError(f"var ({var!r}) needs to be numeric")
    var_label = resolve_label(df, var)
    values = df[var].to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError(f"var ({var!r}) has no finite observations to bin")
    counts, edges = np.histogram(values, bins=bins)
    widths = np.diff(edges)
    out = pd.DataFrame(
        {"bin_left": edges[:-1], "bin_right": edges[1:], "count": counts}
    )
    centers = (edges[:-1] + edges[1:]) / 2.0
    density = counts / (counts.sum() * widths)
    fig = go.Figure(
        go.Bar(
            x=centers,
            y=counts,
            width=widths,
            marker={"color": color_for(0), "line": {"color": "white", "width": 0.5}},
            customdata=np.stack([edges[:-1], edges[1:]], axis=1),
            hovertemplate="[%{customdata[0]:.4g}, %{customdata[1]:.4g})<br>"
            "count=%{y}<extra></extra>",
        )
    )
    apply_default_layout(
        fig,
        xaxis={"title": var_label},
        yaxis={"title": "Count"},
        bargap=0,
        updatemenus=[
            {
                "type": "buttons",
                "direction": "right",
                "x": 0,
                "y": 1.14,
                "xanchor": "left",
                "showactive": True,
                "buttons": [
                    {
                        "label": "Count",
                        "method": "update",
                        "args": [{"y": [counts]}, {"yaxis.title.text": "Count"}, [0]],
                    },
                    {
                        "label": "Density",
                        "method": "update",
                        "args": [
                            {"y": [density]},
                            {"yaxis.title.text": "Density"},
                            [0],
                        ],
                    },
                ],
            }
        ],
    )
    return HistogramResult(df=out, fig=fig)


def explore_bar_plot(
    df: pd.DataFrame,
    var: str,
    *,
    order_by_count: bool = False,
    color: str | None = None,
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
        Bar fill color. Defaults to the primary theme color.

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
    ex.explore_bar_plot(df, "continent").fig
    ```

    Advanced — order bars by descending count and set a custom color:

    ```python
    ex.explore_bar_plot(df, "continent", order_by_count=True, color="red").fig
    ```
    """
    df = ensure_dataframe(df)
    if var not in df.columns:
        raise ValueError("var needs to be in df")
    var_label = resolve_label(df, var)
    counts = df[var].value_counts(dropna=False)
    out = counts.rename_axis(var).reset_index(name="count")
    if not order_by_count:
        # Sort by category as text so a missing/NaN level cannot raise on a mixed-type index.
        out = out.sort_values(
            var, key=lambda s: s.astype(str), kind="stable"
        ).reset_index(drop=True)
    bar_color = color if color is not None else color_for(0)
    fig = go.Figure(
        go.Bar(
            x=out[var].astype(str),
            y=out["count"],
            marker={"color": bar_color, "line": {"color": "white", "width": 0.5}},
            hovertemplate=f"{var_label}=%{{x}}<br>count=%{{y}}<extra></extra>",
        )
    )
    xaxis: dict = {"title": var_label}
    if len(out) > 8:
        xaxis["tickangle"] = -40
    apply_default_layout(fig, xaxis=xaxis, yaxis={"title": "Count"})
    return BarChartResult(df=out, fig=fig)
