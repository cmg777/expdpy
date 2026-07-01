"""Tests for the Gapminder-style animated bubble scatter."""

from __future__ import annotations

import pandas as pd
import pytest

import expdpy as ex
from expdpy._types import AnimatedScatterResult
from expdpy.data import load_gapminder


@pytest.fixture(scope="module")
def gap() -> pd.DataFrame:
    return ex.set_panel(load_gapminder(), entity="country", time="year")


def test_animated_scatter_builds_frames(gap):
    res = ex.explore_animated_scatter_plot(
        gap, x="gdpPercap", y="lifeExp", size="pop", color="continent"
    )
    assert isinstance(res, AnimatedScatterResult)
    n_periods = gap["year"].nunique()
    # One frame per period, and a slider step per period.
    assert len(res.fig.frames) == n_periods
    assert len(res.fig.layout.sliders[0].steps) == n_periods
    # Discrete color → one series per continent, with a play/pause menu.
    n_levels = gap["continent"].nunique()
    assert len(res.fig.data) == n_levels
    assert res.fig.layout.updatemenus  # play / pause buttons
    # Axis ranges are pinned (so the animation does not jump around).
    assert res.fig.layout.xaxis.range is not None
    assert res.fig.layout.yaxis.range is not None


def test_numeric_color_uses_colorbar(gap):
    res = ex.explore_animated_scatter_plot(gap, x="gdpPercap", y="lifeExp", color="pop")
    # Numeric color → a single bubble series per frame with a colorbar.
    assert len(res.fig.data) == 1
    assert res.fig.data[0].marker.showscale


def test_log_x_axis(gap):
    res = ex.explore_animated_scatter_plot(
        gap, x="gdpPercap", y="lifeExp", size="pop", color="continent", log_x=True
    )
    assert res.fig.layout.xaxis.type == "log"
    assert res.fig.layout.xaxis.range is not None


def test_requires_time_id():
    df = pd.DataFrame({"x": [1.0, 2, 3], "y": [1.0, 2, 3]})
    with pytest.raises(ValueError, match="time id"):
        ex.explore_animated_scatter_plot(df, x="x", y="y")


def test_non_numeric_axis_raises(gap):
    with pytest.raises(ValueError, match="numeric"):
        ex.explore_animated_scatter_plot(gap, x="country", y="lifeExp")
