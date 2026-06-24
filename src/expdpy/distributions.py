"""Distribution graphs: histogram and simple count bar chart."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pandas.api import types as pdt
from scipy.stats import gaussian_kde, norm

from expdpy._labels import resolve_label
from expdpy._theme import apply_default_layout, color_for
from expdpy._types import BarChartResult, HistogramResult
from expdpy._validation import ensure_dataframe

__all__ = ["explore_bar_plot", "explore_histogram"]


def _kde_on_grid(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Gaussian-KDE density of ``values`` on ``grid`` (zeros if KDE is ill-posed)."""
    if values.size >= 5 and np.ptp(values) > 0:
        try:
            return gaussian_kde(values)(grid)
        except (np.linalg.LinAlgError, ValueError):  # pragma: no cover - singular cov
            return np.zeros_like(grid)
    return np.zeros_like(grid)


def explore_histogram(
    df: pd.DataFrame,
    var: str,
    *,
    bins: int = 30,
    kde: bool = False,
    normal: bool = False,
) -> HistogramResult:
    """Histogram of a numeric variable, optionally with density overlays.

    Parameters
    ----------
    df
        Data frame containing ``var``.
    var
        Numeric column to bin.
    bins
        Number of equal-width bins.
    kde
        Overlay a Gaussian kernel-density estimate of the distribution.
    normal
        Overlay a normal curve with the sample mean and standard deviation.

    Returns
    -------
    HistogramResult
        ``df`` (columns ``bin_left``, ``bin_right``, ``count``) and the Plotly ``fig``.

    Notes
    -----
    The density overlays are drawn on the **Density** scale, so requesting either one opens
    the figure in Density view; the built-in Count/Density toggle hides the overlays in
    Count view and shows them in Density view.

    Examples
    --------
    Basic — a 30-bin histogram of a numeric variable, with readable labels from the data
    dictionary:

    ```python
    import expdpy as ex
    from expdpy.data import load_kuznets, load_kuznets_data_def

    df = ex.set_labels(load_kuznets(), load_kuznets_data_def(), set_panel=True)
    ex.explore_histogram(df, "gini_regional").fig
    ```

    Advanced — overlay a kernel-density estimate and a normal curve:

    ```python
    import expdpy as ex
    from expdpy.data import load_kuznets, load_kuznets_data_def

    df = ex.set_labels(load_kuznets(), load_kuznets_data_def(), set_panel=True)
    ex.explore_histogram(df, "log_gdp_pc", bins=40, kde=True, normal=True).fig
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

    # Overlays are density-scaled, so requesting one opens the figure in Density view.
    start_density = kde or normal
    bar_hover_count = (
        "[%{customdata[0]:.4g}, %{customdata[1]:.4g})<br>count=%{y}<extra></extra>"
    )
    bar_hover_density = "[%{customdata[0]:.4g}, %{customdata[1]:.4g})<br>density=%{y:.4g}<extra></extra>"
    fig = go.Figure(
        go.Bar(
            x=centers,
            y=density if start_density else counts,
            width=widths,
            marker={"color": color_for(0), "line": {"color": "white", "width": 0.5}},
            customdata=np.stack([edges[:-1], edges[1:]], axis=1),
            hovertemplate=bar_hover_density if start_density else bar_hover_count,
            showlegend=False,
        )
    )

    # Density-scaled overlay curves, comparable to the Density-view bars.
    grid = np.linspace(float(values.min()), float(values.max()), 200)
    overlay_idx: list[int] = []
    overlay_y: list[np.ndarray] = []
    overlay_hover: list[str] = []
    curve_hover = f"{var_label}=%{{x:.4g}}<br>density=%{{y:.4g}}<extra></extra>"
    if kde:
        kde_y = _kde_on_grid(values, grid)
        fig.add_trace(
            go.Scatter(
                x=grid,
                y=kde_y,
                mode="lines",
                name="KDE",
                line={"color": color_for(1), "width": 2.5},
                hovertemplate=curve_hover,
            )
        )
        overlay_idx.append(len(fig.data) - 1)
        overlay_y.append(kde_y)
        overlay_hover.append(curve_hover)
    if normal:
        sigma = float(np.std(values, ddof=1))
        norm_y = (
            norm.pdf(grid, float(np.mean(values)), sigma)
            if sigma > 0
            else np.zeros_like(grid)
        )
        fig.add_trace(
            go.Scatter(
                x=grid,
                y=norm_y,
                mode="lines",
                name="Normal",
                line={"color": color_for(3), "width": 2.5, "dash": "dash"},
                hovertemplate=curve_hover,
            )
        )
        overlay_idx.append(len(fig.data) - 1)
        overlay_y.append(norm_y)
        overlay_hover.append(curve_hover)

    # Toggle: Count hides the (density-scaled) overlays; Density shows them.
    idx = [0, *overlay_idx]
    n = len(overlay_idx)
    count_args = {
        "y": [counts, *overlay_y],
        "visible": [True, *([False] * n)],
        "hovertemplate": [bar_hover_count, *overlay_hover],
    }
    density_args = {
        "y": [density, *overlay_y],
        "visible": [True, *([True] * n)],
        "hovertemplate": [bar_hover_density, *overlay_hover],
    }
    apply_default_layout(
        fig,
        xaxis={"title": var_label},
        yaxis={"title": "Density" if start_density else "Count"},
        bargap=0,
        updatemenus=[
            {
                "type": "buttons",
                "direction": "right",
                "x": 0,
                "y": 1.14,
                "xanchor": "left",
                "showactive": True,
                "active": 1 if start_density else 0,
                "buttons": [
                    {
                        "label": "Count",
                        "method": "update",
                        "args": [count_args, {"yaxis.title.text": "Count"}, idx],
                    },
                    {
                        "label": "Density",
                        "method": "update",
                        "args": [density_args, {"yaxis.title.text": "Density"}, idx],
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
    Basic — category counts of a (typically categorical) variable, with readable
    labels from the data dictionary:

    ```python
    import expdpy as ex
    from expdpy.data import load_kuznets, load_kuznets_data_def

    df = ex.set_labels(load_kuznets(), load_kuznets_data_def(), set_panel=True)
    ex.explore_bar_plot(df, "continent").fig
    ```

    Advanced — order bars by descending count and set a custom color:

    ```python
    import expdpy as ex
    from expdpy.data import load_kuznets, load_kuznets_data_def

    df = ex.set_labels(load_kuznets(), load_kuznets_data_def(), set_panel=True)
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
