"""Tests for the shared Plotly theme (palette, fonts, template, export config)."""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

from expdpy._theme import (
    COLOR_SEQUENCE,
    DIVERGING_SCALE,
    FONT_FAMILY,
    FONT_SIZE_AXIS_TITLE,
    FONT_SIZE_BASE,
    FONT_SIZE_TICK,
    FONT_SIZE_TITLE,
    PLOTLY_CONFIG,
    SEQUENTIAL_SCALE,
    TEMPLATE_NAME,
    apply_default_layout,
    color_for,
    diverging_color,
)


def test_palette_is_tableau_10():
    assert len(COLOR_SEQUENCE) == 10
    assert COLOR_SEQUENCE[0] == "#4E79A7"  # Tableau 10 blue
    assert COLOR_SEQUENCE[1] == "#F28E2B"  # Tableau 10 orange


def test_color_for_wraps_around():
    assert color_for(0) == COLOR_SEQUENCE[0]
    assert color_for(len(COLOR_SEQUENCE)) == COLOR_SEQUENCE[0]


def test_template_registered():
    # The template is registered; note a third-party import (e.g. streamlit) may later
    # override the *global* default, which is why apply_default_layout applies the
    # expdpy template explicitly to every figure (see test below).
    assert TEMPLATE_NAME in pio.templates
    tmpl = pio.templates[TEMPLATE_NAME]
    assert list(tmpl.layout.colorway) == COLOR_SEQUENCE


def test_apply_default_layout_uses_expdpy_template():
    fig = apply_default_layout(go.Figure())
    # The combined template carries the expdpy colorway regardless of the global default.
    assert list(fig.layout.template.layout.colorway) == COLOR_SEQUENCE


def test_apply_default_layout_sets_presentation_fonts():
    fig = apply_default_layout(go.Figure())
    assert FONT_FAMILY.split(",")[0] in fig.layout.template.layout.font.family
    assert fig.layout.template.layout.font.size == FONT_SIZE_BASE
    assert fig.layout.template.layout.title.font.size == FONT_SIZE_TITLE
    assert fig.layout.template.layout.xaxis.title.font.size == FONT_SIZE_AXIS_TITLE
    assert fig.layout.template.layout.yaxis.tickfont.size == FONT_SIZE_TICK


def test_apply_default_layout_forwards_kwargs():
    fig = apply_default_layout(go.Figure(), xaxis={"title": "x"}, bargap=0)
    assert fig.layout.xaxis.title.text == "x"
    assert fig.layout.bargap == 0


def test_plotly_config_high_res_export():
    assert PLOTLY_CONFIG["toImageButtonOptions"]["scale"] == 2
    assert PLOTLY_CONFIG["toImageButtonOptions"]["format"] == "png"


def test_continuous_scales_are_explicit_stops():
    for scale in (DIVERGING_SCALE, SEQUENTIAL_SCALE):
        assert scale[0][0] == 0.0
        assert scale[-1][0] == 1.0
        assert all(str(color).startswith("#") for _, color in scale)


def test_diverging_color_endpoints_and_midpoint():
    assert diverging_color(-1.0) == "rgb(225,87,89)"  # Tableau red
    assert diverging_color(1.0) == "rgb(78,121,167)"  # Tableau blue
    assert diverging_color(0.0) == "rgb(245,245,245)"  # near-white midpoint
