"""Shared Plotly styling for expdpy figures.

This module centralizes the visual identity of every expdpy figure so the look is
consistent across notebooks, scripts, static exports and the Streamlit app:

* a **Tableau 10** qualitative palette for grouped series (:data:`COLOR_SEQUENCE`),
* cohesive Tableau-style continuous scales (:data:`DIVERGING_SCALE`,
  :data:`SEQUENTIAL_SCALE`),
* a presentation-friendly font stack and sizes (Arial/Helvetica, larger labels),
* a registered Plotly template (``"expdpy"``) layered on ``plotly_white`` and set as the
  default, so figures are styled even when a caller forgets :func:`apply_default_layout`,
* a high-resolution export config for crisp slide-ready PNGs (:data:`PLOTLY_CONFIG`).
"""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

__all__ = [
    "COLOR_SEQUENCE",
    "DIVERGING_SCALE",
    "FONT_FAMILY",
    "FONT_SIZE_AXIS_TITLE",
    "FONT_SIZE_BASE",
    "FONT_SIZE_LEGEND",
    "FONT_SIZE_TICK",
    "FONT_SIZE_TITLE",
    "PLOTLY_CONFIG",
    "SEQUENTIAL_SCALE",
    "TEMPLATE_NAME",
    "apply_default_layout",
    "color_for",
    "diverging_color",
]

# --- Qualitative palette -------------------------------------------------------------
# The classic Tableau 10 palette: distinct, muted, and well-suited to projection on
# presentation slides. Used for grouped series via :func:`color_for`.
COLOR_SEQUENCE: list[str] = [
    "#4E79A7",  # blue
    "#F28E2B",  # orange
    "#59A14F",  # green
    "#E15759",  # red
    "#76B7B2",  # teal
    "#EDC948",  # yellow
    "#B07AA1",  # purple
    "#FF9DA7",  # pink
    "#9C755F",  # brown
    "#BAB0AC",  # gray
]

# --- Continuous color scales ---------------------------------------------------------
# A Tableau-flavoured diverging scale (red <-> light neutral <-> blue), anchored at a
# near-white midpoint. Drives the correlation heatmap and the ellipse fill (see
# :func:`diverging_color`) so both styles look the same.
DIVERGING_SCALE: list[list[float | str]] = [
    [0.0, "#E15759"],  # strong negative -> Tableau red
    [0.25, "#F1A7A9"],
    [0.5, "#F5F5F5"],  # zero -> near-white
    [0.75, "#9FB8D4"],
    [1.0, "#4E79A7"],  # strong positive -> Tableau blue
]

# A Tableau-flavoured sequential blue ramp (light -> Tableau blue) for magnitude-only
# encodings such as the missing-values heatmap and continuous scatter color.
SEQUENTIAL_SCALE: list[list[float | str]] = [
    [0.0, "#F7FBFF"],
    [0.25, "#C6DBEF"],
    [0.5, "#90B5D6"],
    [0.75, "#5C8FBC"],
    [1.0, "#2E5C8A"],
]

# --- Fonts -------------------------------------------------------------------------
# Arial/Helvetica renders identically across machines and static exports. Sizes follow
# a "presentation" tier so axis labels remain legible when projected on slides.
FONT_FAMILY: str = "Arial, Helvetica, sans-serif"
FONT_SIZE_BASE: int = 16
FONT_SIZE_TICK: int = 15
FONT_SIZE_AXIS_TITLE: int = 18
FONT_SIZE_TITLE: int = 22
FONT_SIZE_LEGEND: int = 15

TEMPLATE_NAME: str = "expdpy"

# Modebar / export config: emit crisp 2x PNGs suitable for slides.
PLOTLY_CONFIG: dict[str, object] = {
    "displaylogo": False,
    "toImageButtonOptions": {
        "format": "png",
        "filename": "expdpy_figure",
        "scale": 2,
    },
}


def _build_template() -> go.layout.Template:
    """Construct the ``expdpy`` Plotly template (layered on ``plotly_white``)."""
    template = go.layout.Template()
    template.layout = go.Layout(
        font={"family": FONT_FAMILY, "size": FONT_SIZE_BASE, "color": "#2a2a2a"},
        title={"font": {"family": FONT_FAMILY, "size": FONT_SIZE_TITLE}, "x": 0.02},
        colorway=COLOR_SEQUENCE,
        colorscale={"sequential": SEQUENTIAL_SCALE, "diverging": DIVERGING_SCALE},
        margin={"l": 70, "r": 30, "t": 60, "b": 60},
        legend={
            "bgcolor": "rgba(255,255,255,0.6)",
            "font": {"family": FONT_FAMILY, "size": FONT_SIZE_LEGEND},
            "title": {"font": {"family": FONT_FAMILY, "size": FONT_SIZE_LEGEND}},
        },
        hoverlabel={"font": {"family": FONT_FAMILY, "size": FONT_SIZE_TICK}},
        xaxis={
            "title": {"font": {"family": FONT_FAMILY, "size": FONT_SIZE_AXIS_TITLE}},
            "tickfont": {"family": FONT_FAMILY, "size": FONT_SIZE_TICK},
            "automargin": True,
            "gridcolor": "rgba(0,0,0,0.08)",
            "zerolinecolor": "rgba(0,0,0,0.15)",
        },
        yaxis={
            "title": {"font": {"family": FONT_FAMILY, "size": FONT_SIZE_AXIS_TITLE}},
            "tickfont": {"family": FONT_FAMILY, "size": FONT_SIZE_TICK},
            "automargin": True,
            "gridcolor": "rgba(0,0,0,0.08)",
            "zerolinecolor": "rgba(0,0,0,0.15)",
        },
    )
    return template


# Register the template and make ``plotly_white + expdpy`` the process-wide default so
# even figures that bypass ``apply_default_layout`` pick up the expdpy look.
pio.templates[TEMPLATE_NAME] = _build_template()
pio.templates.default = f"plotly_white+{TEMPLATE_NAME}"

# The combined template string applied to every figure for belt-and-suspenders styling.
_COMBINED_TEMPLATE = f"plotly_white+{TEMPLATE_NAME}"


def apply_default_layout(fig: go.Figure, **layout_kwargs: object) -> go.Figure:
    """Apply expdpy's default layout (Tableau theme, presentation fonts) to ``fig``.

    The expdpy template carries the palette, continuous scales, fonts and sizes; this
    function applies it explicitly (so per-figure output is correct regardless of the
    global default) and then forwards any extra ``layout_kwargs`` to
    :meth:`plotly.graph_objects.Figure.update_layout`.

    Parameters
    ----------
    fig
        The figure to style (modified in place and returned).
    **layout_kwargs
        Extra keyword arguments forwarded to
        :meth:`plotly.graph_objects.Figure.update_layout`.
    """
    fig.update_layout(template=_COMBINED_TEMPLATE)
    if layout_kwargs:
        fig.update_layout(**layout_kwargs)
    return fig


def color_for(index: int) -> str:
    """Return the palette color for a 0-based series ``index`` (wraps around)."""
    return COLOR_SEQUENCE[index % len(COLOR_SEQUENCE)]


def diverging_color(value: float) -> str:
    """Map a value in ``[-1, 1]`` to an ``rgb(...)`` string on :data:`DIVERGING_SCALE`.

    Used for the correlation ellipse fills so they match the heatmap's diverging scale.
    ``-1`` is Tableau red, ``0`` near-white, ``+1`` Tableau blue, with linear
    interpolation between the scale's anchor stops.
    """
    v = max(-1.0, min(1.0, value))
    pos = (v + 1.0) / 2.0  # map [-1, 1] -> [0, 1]
    stops = DIVERGING_SCALE
    for i in range(len(stops) - 1):
        p0, c0 = float(stops[i][0]), str(stops[i][1])
        p1, c1 = float(stops[i + 1][0]), str(stops[i + 1][1])
        if pos <= p1:
            t = 0.0 if p1 == p0 else (pos - p0) / (p1 - p0)
            r0, g0, b0 = _hex_to_rgb(c0)
            r1, g1, b1 = _hex_to_rgb(c1)
            r = round(r0 + (r1 - r0) * t)
            g = round(g0 + (g1 - g0) * t)
            b = round(b0 + (b1 - b0) * t)
            return f"rgb({r},{g},{b})"
    return f"rgb{_hex_to_rgb(str(stops[-1][1]))}"


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    """Convert a ``#rrggbb`` hex string to an ``(r, g, b)`` integer tuple."""
    h = value.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
