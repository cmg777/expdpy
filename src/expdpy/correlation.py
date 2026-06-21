"""Correlation graph (Plotly): Pearson above, Spearman below the diagonal."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from expdpy._corr import cor_mat
from expdpy._labels import resolve_labels
from expdpy._theme import DIVERGING_SCALE, apply_default_layout, diverging_color
from expdpy._types import CorrelationGraphResult
from expdpy._validation import ensure_dataframe, numeric_logical_columns

__all__ = ["explore_correlation_plot"]


def _ellipse_points(cx: float, cy: float, r: float, scale: float = 0.45, n: int = 50):
    """Return x/y arrays tracing a correlation ellipse for coefficient ``r``.

    Uses the classic correlation-ellipse parametrization (as in R's ``ellipse`` package):
    a circle for ``r = 0`` collapsing to a 45-degree line as ``|r| -> 1``.
    """
    t = np.linspace(0.0, 2.0 * np.pi, n)
    a = np.arccos(np.clip(r, -1.0, 1.0))
    x = cx + scale * np.cos(t + a / 2.0)
    y = cy + scale * np.cos(t - a / 2.0)
    return x, y


def explore_correlation_plot(
    df: pd.DataFrame,
    *,
    style: Literal["heatmap", "ellipse"] = "heatmap",
) -> CorrelationGraphResult:
    """Visualise a correlation matrix (Pearson above, Spearman below the diagonal).

    Parameters
    ----------
    df
        Data frame with at least two numeric/logical variables and five observations.
    style
        ``"heatmap"`` (default) renders a Plotly heatmap; ``"ellipse"`` reproduces R's
        ``corrplot(method = "ellipse")`` look with one ellipse glyph per cell.

    Returns
    -------
    CorrelationGraphResult
        ``df_corr``/``df_prob``/``df_n`` plus the Plotly ``fig``.

    Examples
    --------
    Basic — a correlation heatmap for a few variables:

    ```python
    import expdpy as ex
    from expdpy.data import load_kuznets

    df = load_kuznets()
    ex.explore_correlation_plot(df[["gini_regional", "gdp_pc", "log_gdp_pc"]]).fig
    ```

    Advanced — the ellipse style (R ``corrplot`` look), with the underlying
    correlation matrix available from ``.df_corr``:

    ```python
    result = ex.explore_correlation_plot(
        df[["gini_regional", "gdp_pc", "log_gdp_pc", "trade_share"]],
        style="ellipse",
    )
    result.fig
    result.df_corr
    ```
    """
    df = ensure_dataframe(df)
    labels_src = df  # resolve labels before the column reslice drops df.attrs
    df = df[numeric_logical_columns(df)]
    if len(df) < 5 or df.shape[1] < 2:
        raise ValueError(
            "'df' needs at least two variables and five observations of numerical data"
        )

    pcorr = cor_mat(df, "pearson")
    scorr = cor_mat(df, "spearman")
    lower = np.tril(np.ones(pcorr.r.shape, dtype=bool), k=-1)
    r = pcorr.r.to_numpy(dtype=float).copy()
    r[lower] = scorr.r.to_numpy(dtype=float)[lower]
    r = np.clip(r, -1.0, 1.0)
    p = pcorr.p.to_numpy(dtype=float).copy()
    p[lower] = scorr.p.to_numpy(dtype=float)[lower]
    n = np.where(lower, scorr.n.to_numpy(), pcorr.n.to_numpy())

    names = list(df.columns)
    name_labels = resolve_labels(labels_src, names)
    corr_r = pd.DataFrame(r, index=names, columns=names)
    corr_p = pd.DataFrame(p, index=names, columns=names)
    corr_n = pd.DataFrame(n, index=names, columns=names).astype("Int64")

    k = len(names)
    if style == "heatmap":
        fig = go.Figure(
            go.Heatmap(
                z=r,
                x=name_labels,
                y=name_labels,
                zmid=0.0,
                zmin=-1.0,
                zmax=1.0,
                colorscale=DIVERGING_SCALE,
                colorbar={"title": "corr"},
                xgap=1,
                ygap=1,
                text=np.round(r, 2),
                texttemplate="%{text}" if k <= 12 else None,
                hovertemplate="%{y} vs %{x}<br>corr=%{z:.3f}<extra></extra>",
            )
        )
        fig.update_yaxes(autorange="reversed")
    else:
        fig = go.Figure()
        for i in range(k):
            for j in range(k):
                if i == j or not np.isfinite(r[i, j]):
                    continue
                # place row i at top: invert y
                cy = k - 1 - i
                ex, ey = _ellipse_points(j, cy, r[i, j])
                fig.add_trace(
                    go.Scatter(
                        x=ex,
                        y=ey,
                        fill="toself",
                        mode="lines",
                        line={"color": "rgba(120,120,120,0.5)", "width": 0.5},
                        fillcolor=diverging_color(r[i, j]),
                        hoverinfo="text",
                        text=f"{name_labels[i]} vs {name_labels[j]}: {r[i, j]:.3f}",
                        showlegend=False,
                    )
                )
        fig.update_xaxes(
            tickmode="array",
            tickvals=list(range(k)),
            ticktext=name_labels,
            range=[-0.6, k - 0.4],
        )
        fig.update_yaxes(
            tickmode="array",
            tickvals=list(range(k)),
            ticktext=list(reversed(name_labels)),
            range=[-0.6, k - 0.4],
            scaleanchor="x",
            scaleratio=1,
        )

    apply_default_layout(fig, title="Correlations")
    fig.add_annotation(
        text="Pearson above the diagonal · Spearman below",
        xref="paper",
        yref="paper",
        x=0,
        y=1.06,
        showarrow=False,
        font={"size": 13},
    )
    return CorrelationGraphResult(df_corr=corr_r, df_prob=corr_p, df_n=corr_n, fig=fig)
