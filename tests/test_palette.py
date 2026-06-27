"""Tests for the opt-in colorblind-safe palette toggle (set_palette / get_palette)."""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio
import pytest

import expdpy as ex
from expdpy._theme import (
    COLOR_SEQUENCE,
    COLOR_SEQUENCE_COLORBLIND,
    DIVERGING_SCALE_COLORBLIND,
    SEQUENTIAL_SCALE_COLORBLIND,
    TEMPLATE_NAME,
    active_diverging_scale,
    active_sequential_scale,
    apply_default_layout,
    color_for,
)


@pytest.fixture(autouse=True)
def _reset_palette():
    """Guarantee every test starts and ends on the default palette (global state)."""
    ex.set_palette("default")
    yield
    ex.set_palette("default")


def test_default_is_tableau():
    assert ex.get_palette() == "default"
    assert color_for(0) == COLOR_SEQUENCE[0] == "#4E79A7"


def test_colorblind_changes_color_for_and_reverts():
    before = color_for(0)
    ex.set_palette("colorblind")
    assert ex.get_palette() == "colorblind"
    assert color_for(0) == COLOR_SEQUENCE_COLORBLIND[0] == "#0072B2"
    assert color_for(0) != before
    ex.set_palette("default")
    assert color_for(0) == COLOR_SEQUENCE[0]


def test_template_colorway_updates_and_applies():
    ex.set_palette("colorblind")
    assert (
        list(pio.templates[TEMPLATE_NAME].layout.colorway) == COLOR_SEQUENCE_COLORBLIND
    )
    fig = apply_default_layout(go.Figure())
    assert list(fig.layout.template.layout.colorway) == COLOR_SEQUENCE_COLORBLIND


def test_color_for_wraps_on_active_length():
    ex.set_palette("colorblind")
    n = len(COLOR_SEQUENCE_COLORBLIND)
    assert color_for(n) == color_for(0)  # modulo the ACTIVE length (8), not 10


def test_scales_follow_palette():
    ex.set_palette("colorblind")
    assert active_sequential_scale() == SEQUENTIAL_SCALE_COLORBLIND
    assert active_diverging_scale() == DIVERGING_SCALE_COLORBLIND


def test_heatmap_uses_active_scale():
    import pandas as pd

    df = pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5, 6.0],
            "b": [6, 5, 4, 3, 2, 1.0],
            "c": [1, 3, 2, 5, 4, 6.0],
        }
    )
    ex.set_palette("colorblind")
    cs = ex.explore_correlation_plot(df).fig.data[0].colorscale
    # first diverging stop is the Okabe-Ito vermillion
    assert cs[0][1].lower() in ("#d55e00", "rgb(213,94,0)")


def test_invalid_mode_raises():
    with pytest.raises(ValueError, match="unknown palette"):
        ex.set_palette("rainbow")
