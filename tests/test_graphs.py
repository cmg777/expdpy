"""Figure smoke tests for the graph functions (correlation, trends, by-group, etc.)."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import pytest

from expdpy import (
    explore_bar_plot,
    explore_bar_plot_by_group,
    explore_correlation_plot,
    explore_histogram,
    explore_missing_values_plot,
    explore_quantile_trend_plot,
    explore_scatter_plot,
    explore_trend_plot,
    explore_trend_plot_by_group,
    explore_violin_plot_by_group,
)
from expdpy.trends import _try_convert_ts_id


def test_correlation_graph(sample_df):
    res = explore_correlation_plot(sample_df[["x1", "x2", "x3"]])
    assert isinstance(res.fig, go.Figure)
    assert res.df_corr.shape == (3, 3)
    ell = explore_correlation_plot(sample_df[["x1", "x2", "x3"]], style="ellipse")
    assert len(ell.fig.data) > 0


def test_trend_graph_traces_and_means(sample_df):
    res = explore_trend_plot(sample_df, var=["x1", "x2"], time="year")
    assert len(res.fig.data) == 2
    assert {"variable", "year", "mean", "se"}.issubset(res.df.columns)
    # the mean for one cell matches a direct groupby mean
    direct = sample_df.groupby("year")["x1"].mean()
    cell = res.df[(res.df["variable"] == "x1") & (res.df["year"] == 2014)]["mean"].iloc[
        0
    ]
    assert cell == pytest.approx(direct.loc[2014])


def test_quantile_trend_graph(sample_df):
    res = explore_quantile_trend_plot(sample_df, var="x3", time="year")
    assert len(res.fig.data) == 5
    assert set(res.df["quantile"].cat.categories) == {"q05", "q25", "q50", "q75", "q95"}


def test_quantile_median_matches_r(sample_df, goldens):
    res = explore_quantile_trend_plot(sample_df, quantiles=[0.5], var="x3", time="year")
    q50 = res.df.set_index("year")["x3"]
    for year, gold in goldens["median_x3_by_year"].items():
        assert q50.loc[int(year)] == pytest.approx(gold, rel=1e-9)


def test_by_group_bar(sample_df):
    res = explore_bar_plot_by_group(sample_df, "grp", "x3", np.nanmedian)
    assert "stat_x3" in res.df.columns
    assert isinstance(res.fig, go.Figure)


def test_by_group_trend(sample_df):
    res = explore_trend_plot_by_group(
        sample_df, "grp", "x1", time="year", error_bars=True
    )
    assert len(res.fig.data) == sample_df["grp"].nunique()


def test_by_group_violin(sample_df):
    res = explore_violin_plot_by_group(sample_df, "grp", "x1")
    assert isinstance(res.fig, go.Figure)
    assert len(res.fig.data) == sample_df["grp"].nunique()


def test_histogram_bins(sample_df):
    res = explore_histogram(sample_df, "x3", bins=15)
    assert res.df.shape[0] == 15
    assert res.df["count"].sum() == sample_df["x3"].notna().sum()


def test_histogram_plain_has_no_overlay(sample_df):
    # Off by default: a single trace, Count view, default button active.
    res = explore_histogram(sample_df, "x3")
    assert len(res.fig.data) == 1
    assert res.fig.layout.yaxis.title.text == "Count"
    assert res.fig.layout.updatemenus[0].active == 0


def test_histogram_kde_overlay(sample_df):
    res = explore_histogram(sample_df, "x3", kde=True)
    assert len(res.fig.data) == 2
    assert res.fig.data[1].name == "KDE"
    assert res.fig.data[1].mode == "lines"
    # Overlays are density-scaled, so the figure opens in Density view.
    assert res.fig.layout.yaxis.title.text == "Density"
    assert res.fig.layout.updatemenus[0].active == 1
    # The bin/count table is unchanged by overlays.
    assert list(res.df.columns) == ["bin_left", "bin_right", "count"]


def test_histogram_normal_overlay(sample_df):
    res = explore_histogram(sample_df, "x3", normal=True)
    assert len(res.fig.data) == 2
    assert res.fig.data[1].name == "Normal"
    assert res.fig.data[1].line.dash == "dash"


def test_histogram_both_overlays_toggle_visibility(sample_df):
    res = explore_histogram(sample_df, "x3", kde=True, normal=True)
    assert len(res.fig.data) == 3
    count_btn, density_btn = res.fig.layout.updatemenus[0].buttons
    # Count view hides both overlays; Density view shows them.
    assert list(count_btn.args[0]["visible"]) == [True, False, False]
    assert list(density_btn.args[0]["visible"]) == [True, True, True]


def test_bar_chart_counts(sample_df):
    res = explore_bar_plot(sample_df, "grp")
    assert res.df["count"].sum() == len(sample_df)


def test_missing_values_graph(kuznets):
    res = explore_missing_values_plot(kuznets, time="year")
    assert isinstance(res.fig, go.Figure)
    resb = explore_missing_values_plot(kuznets, time="year", binary=True)
    assert isinstance(resb.fig, go.Figure)


def test_missing_values_by_entity(kuznets):
    res = explore_missing_values_plot(kuznets, entity="country", by="entity")
    assert res.df.shape[0] == kuznets["country"].nunique()


def test_missing_values_requires_clean_ts(sample_df):
    df = sample_df.copy()
    df.loc[0, "year"] = np.nan
    with pytest.raises(ValueError):
        explore_missing_values_plot(df, time="year")


def test_scatter_default_alpha_and_loess(sample_df):
    from expdpy.scatter import _default_alpha

    res = explore_scatter_plot(sample_df, "x1", "x2", loess=1)
    assert isinstance(res.fig, go.Figure)
    assert any(t.name == "loess" for t in res.fig.data)
    # The loess smoother shows only the line — no confidence-band trace.
    assert not any(t.name == "ci" for t in res.fig.data)
    assert res.fig.data[0].marker.opacity == pytest.approx(
        _default_alpha(len(sample_df))
    )


def test_scatter_small_sample_alpha_one(sample_df):
    res = explore_scatter_plot(sample_df.head(50), "x1", "x2")
    assert res.fig.data[0].marker.opacity == pytest.approx(1.0)


def test_scatter_color_categorical(sample_df):
    res = explore_scatter_plot(sample_df, "x1", "x2", color="grp")
    names = {t.name for t in res.fig.data}
    assert {"A", "B", "C"}.issubset(names)


def test_scatter_requires_numeric_axes(sample_df):
    with pytest.raises(ValueError):
        explore_scatter_plot(sample_df, "grp", "x1")


@pytest.mark.parametrize(
    "values,expect_ordered",
    [
        (["2014", "2015", "2016"], False),  # numeric-looking strings -> numeric
        (["2014-01-01", "2014-02-01"], False),  # full dates -> datetime
        (["low", "high", "mid"], True),  # non-numeric -> ordered categorical
    ],
)
def test_try_convert_ts_id_branches(values, expect_ordered):
    import pandas as pd

    s = pd.Series(values * 3)
    _conv, ordered = _try_convert_ts_id(s)
    assert ordered is expect_ordered


# --- unified title/subtitle API (item 13b) -------------------------------------------


def test_plot_title_subtitle_thread_through(kuznets):
    fig = explore_trend_plot(
        kuznets, var=["gini_regional"], time="year", title="Trend", subtitle="by year"
    ).fig
    assert fig.layout.title.text == "Trend"
    # axis titles are unaffected by adding a main title
    assert fig.layout.yaxis.title.text == "gini_regional"


def test_plot_no_title_by_default(kuznets):
    fig = explore_scatter_plot(kuznets, x="log_gdp_pc", y="gini_regional").fig
    assert fig.layout.title.text is None


def test_correlation_plot_default_title_preserved(kuznets):
    cols = ["gini_regional", "log_gdp_pc", "log_gdp_pc_sq"]
    assert (
        explore_correlation_plot(kuznets[cols]).fig.layout.title.text == "Correlations"
    )
    assert (
        explore_correlation_plot(kuznets[cols], title="Custom").fig.layout.title.text
        == "Custom"
    )
