"""Tests for the animated composition (treemap/sunburst) and box/strip distribution views."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import pytest

import expdpy as ex
from expdpy._types import (
    BoxPlotResult,
    StripPlotResult,
    SunburstPlotResult,
    TreemapPlotResult,
)
from expdpy._validation import ExpdpyWarning
from expdpy.data import load_gapminder


@pytest.fixture(scope="module")
def gap() -> pd.DataFrame:
    return ex.set_panel(load_gapminder(), entity="country", time="year")


# --------------------------------------------------------------- treemap / sunburst ---
@pytest.mark.parametrize(
    ("fn", "rtype"),
    [
        (ex.explore_treemap_plot, TreemapPlotResult),
        (ex.explore_sunburst_plot, SunburstPlotResult),
    ],
)
def test_composition_builds_animation(gap, fn, rtype):
    res = fn(
        gap, path=["continent", "country"], size="pop", color="lifeExp", time="year"
    )
    assert isinstance(res, rtype)
    assert isinstance(res.fig, go.Figure)
    n = gap["year"].nunique()
    # One frame + one slider step per period.
    assert len(res.fig.frames) == n
    steps = res.fig.layout.sliders[0].steps
    assert len(steps) == n
    # Frames are ordered ascending in time (px would otherwise use first-appearance order).
    labels = [s.label for s in steps]
    assert labels == sorted(labels, key=int)
    # Shared expdpy controls: a single ▶ play button, bottom-left, left of the slider.
    menu = res.fig.layout.updatemenus[0]
    assert len(menu.buttons) == 1
    assert menu.buttons[0].label == "▶"
    assert (menu.x, menu.y) == (0.1, 0) and menu.xanchor == "right"
    assert res.fig.layout.sliders[0].currentvalue.prefix.endswith(": ")
    # Continuous color → a colorbar with a range pinned across frames.
    assert res.fig.layout.coloraxis.cmin is not None


def test_composition_static_without_time():
    df = load_gapminder()
    df = df[df["year"] == 2007].copy()  # a cross-section: no panel declared
    df.attrs.clear()
    res = ex.explore_treemap_plot(df, path=["continent", "country"], size="pop")
    assert res.fig.frames == ()
    assert not res.fig.layout.sliders


def test_treemap_df_preserves_values(gap):
    res = ex.explore_treemap_plot(
        gap, path=["continent", "country"], size="pop", time="year"
    )
    # The complete-case frame keeps every row's value: per-year population totals match.
    for yr in (1952, 2007):
        want = float(gap.loc[gap["year"] == yr, "pop"].sum())
        got = float(res.df.loc[res.df["year"] == yr, "pop"].sum())
        assert got == pytest.approx(want)


def test_treemap_defaults_path_to_entity(gap):
    res = ex.explore_treemap_plot(gap, size="pop", time="year")
    assert len(res.fig.frames) == gap["year"].nunique()


def test_treemap_requires_a_hierarchy():
    df = pd.DataFrame({"g": ["a", "b"], "v": [1.0, 2.0]})  # no path, no entity
    with pytest.raises(ValueError, match="hierarchy"):
        ex.explore_treemap_plot(df, size="v")


def test_treemap_samples_large_hierarchies(gap):
    with pytest.warns(ExpdpyWarning, match="sampled"):
        res = ex.explore_treemap_plot(
            gap, path=["continent", "country"], size="pop", time="year", max_units=10
        )
    assert res.df["country"].nunique() == 10


# --------------------------------------------------------------------- box / strip ---
# NB: ``sample_df`` is session-scoped and ``set_panel`` mutates ``attrs`` in place, so these
# tests pass ``time=`` explicitly (never declaring a panel on the shared fixture) and clear
# ``attrs`` where a genuinely panel-free frame is needed.
def test_box_builds_animation(sample_df):
    res = ex.explore_box_plot(sample_df, "grp", "x1", time="year")
    assert isinstance(res, BoxPlotResult)
    n = sample_df["year"].nunique()
    assert len(res.fig.frames) == n
    assert len(res.fig.layout.sliders[0].steps) == n
    # One box trace per group (color=by_var), and the value axis is pinned.
    assert len(res.fig.data) == sample_df["grp"].nunique()
    assert res.fig.layout.xaxis.range is not None
    # A single ▶ play button (Plotly can't do a stateful toggle), bottom-left.
    assert len(res.fig.layout.updatemenus[0].buttons) == 1
    assert res.fig.layout.updatemenus[0].buttons[0].label == "▶"
    # df is stored [time, by_var, var] for the interpreter.
    assert list(res.df.columns) == ["year", "grp", "x1"]


def test_strip_builds_animation_and_samples(sample_df):
    with pytest.warns(ExpdpyWarning, match="sampled"):
        res = ex.explore_strip_plot(
            sample_df, "grp", "x1", time="year", max_units=20, alpha=0.4
        )
    assert isinstance(res, StripPlotResult)
    # After sampling, there is one frame per period still present in the sample.
    assert len(res.fig.frames) == res.df["year"].nunique()
    assert len(res.df) == 20
    assert res.fig.data[0].marker.opacity == pytest.approx(0.4)


def test_box_static_fallback(sample_df):
    # No panel declared and no time= → a single static set of boxes.
    df = sample_df.copy()
    df.attrs.clear()
    res = ex.explore_box_plot(df, "grp", "x1")
    assert res.fig.frames == ()
    assert not res.fig.layout.sliders
    assert list(res.df.columns) == ["grp", "x1"]


def test_box_log_axis(sample_df):
    df = sample_df.copy()
    df["x1"] = df["x1"].abs() + 1.0  # positive for a log axis
    res = ex.explore_box_plot(df, "grp", "x1", time="year", log=True, group_on_y=False)
    assert res.fig.layout.yaxis.type == "log"
    assert res.fig.layout.yaxis.range is not None


def test_box_median_matches_pandas(sample_df):
    res = ex.explore_box_plot(sample_df, "grp", "x1", time="year")
    last = int(sample_df["year"].max())
    grp = str(sample_df["grp"].dropna().iloc[0])
    want = float(
        sample_df.loc[
            (sample_df["year"] == last) & (sample_df["grp"] == grp), "x1"
        ].median()
    )
    got = float(
        res.df.loc[(res.df["year"] == last) & (res.df["grp"] == grp), "x1"].median()
    )
    assert got == pytest.approx(want)


def test_box_interpret_is_associational(sample_df):
    text = ex.explore_box_plot(sample_df, "grp", "x1", time="year").interpret()
    assert isinstance(text, str) and text
    low = text.lower()
    assert "causes" not in low and "effect of" not in low
    assert "correlation_vs_causation" in low


def test_box_requires_numeric_var(sample_df):
    with pytest.raises(ValueError, match="numeric"):
        ex.explore_box_plot(sample_df, "grp", "grp")


def test_strip_hover_names_the_entity(gap):
    # Strip shows individual points, so each point's info box names its unit.
    res = ex.explore_strip_plot(
        gap, "continent", "lifeExp", time="year", max_units=None
    )
    names: set[str] = set()
    for tr in res.fig.data:
        if tr.hovertext is not None:
            names.update(map(str, tr.hovertext))
    assert "Norway" in names  # a gapminder country surfaces in the hover box
    # Box aggregates, so it carries no per-point entity hover.
    box = ex.explore_box_plot(gap, "continent", "lifeExp", time="year")
    assert all(tr.hovertext is None for tr in box.fig.data)


def test_new_chart_code_snippets():
    # The Streamlit "reproduce" snippets for the new charts (pure functions; no Streamlit).
    from expdpy.streamlit_app._export_nb import component_code
    from expdpy.streamlit_app._state import DEFAULT_CONFIG

    box = dict(DEFAULT_CONFIG, box_byvar="continent", box_var="gini", box_anim=True)
    assert component_code("box_plot", box, "year") == (
        "ex.explore_box_plot(df, 'continent', 'gini', time='year').fig"
    )
    strip = dict(DEFAULT_CONFIG, strip_byvar="continent", strip_var="gini")
    assert "explore_strip_plot" in component_code("strip_plot", strip, "year")
    comp = dict(
        DEFAULT_CONFIG,
        comp_path=["continent", "country"],
        comp_size="pop",
        comp_anim=True,
    )
    assert "explore_treemap_plot" in component_code("treemap_plot", comp, "year")
    assert "explore_sunburst_plot" in component_code("sunburst_plot", comp, "year")
    # An incomplete selection yields no snippet.
    assert component_code("box_plot", dict(DEFAULT_CONFIG), "year") is None
