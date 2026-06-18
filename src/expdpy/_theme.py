"""Shared Plotly styling defaults for expdpy figures."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import plotly.graph_objects as go

# A qualitative palette (Plotly's default "D3") used for grouped series so that
# expdpy figures look consistent regardless of the active template.
COLOR_SEQUENCE: list[str] = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


def apply_default_layout(fig: go.Figure, **layout_kwargs: object) -> go.Figure:
    """Apply expdpy's default layout (clean template, tight margins) to ``fig``.

    Parameters
    ----------
    fig
        The figure to style (modified in place and returned).
    **layout_kwargs
        Extra keyword arguments forwarded to :meth:`plotly.graph_objects.Figure.update_layout`.
    """
    fig.update_layout(
        template="plotly_white",
        margin={"l": 60, "r": 30, "t": 50, "b": 50},
        legend={"bgcolor": "rgba(255,255,255,0.6)"},
    )
    if layout_kwargs:
        fig.update_layout(**layout_kwargs)
    return fig


def color_for(index: int) -> str:
    """Return the palette color for a 0-based series ``index`` (wraps around)."""
    return COLOR_SEQUENCE[index % len(COLOR_SEQUENCE)]
