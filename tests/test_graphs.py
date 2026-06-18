"""Figure smoke tests for the graph functions (correlation, trends, by-group, etc.)."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import pytest

from expdpy import (
    prepare_bar_chart,
    prepare_by_group_bar_graph,
    prepare_by_group_trend_graph,
    prepare_by_group_violin_graph,
    prepare_correlation_graph,
    prepare_histogram,
    prepare_missing_values_graph,
    prepare_quantile_trend_graph,
    prepare_scatter_plot,
    prepare_trend_graph,
)
from expdpy.trends import _try_convert_ts_id


def test_correlation_graph(sample_df):
    res = prepare_correlation_graph(sample_df[["x1", "x2", "x3"]])
    assert isinstance(res.fig, go.Figure)
    assert res.df_corr.shape == (3, 3)
    ell = prepare_correlation_graph(sample_df[["x1", "x2", "x3"]], style="ellipse")
    assert len(ell.fig.data) > 0


def test_trend_graph_traces_and_means(sample_df):
    res = prepare_trend_graph(sample_df, ts_id="year", var=["x1", "x2"])
    assert len(res.fig.data) == 2
    assert {"variable", "year", "mean", "se"}.issubset(res.df.columns)
    # the mean for one cell matches a direct groupby mean
    direct = sample_df.groupby("year")["x1"].mean()
    cell = res.df[(res.df["variable"] == "x1") & (res.df["year"] == 2014)]["mean"].iloc[
        0
    ]
    assert cell == pytest.approx(direct.loc[2014])


def test_quantile_trend_graph(sample_df):
    res = prepare_quantile_trend_graph(sample_df, ts_id="year", var="x3")
    assert len(res.fig.data) == 5
    assert set(res.df["quantile"].cat.categories) == {"q05", "q25", "q50", "q75", "q95"}


def test_quantile_median_matches_r(sample_df, goldens):
    res = prepare_quantile_trend_graph(
        sample_df, ts_id="year", var="x3", quantiles=[0.5]
    )
    q50 = res.df.set_index("year")["x3"]
    for year, gold in goldens["median_x3_by_year"].items():
        assert q50.loc[int(year)] == pytest.approx(gold, rel=1e-9)


def test_by_group_bar(sample_df):
    res = prepare_by_group_bar_graph(sample_df, "grp", "x3", np.nanmedian)
    assert "stat_x3" in res.df.columns
    assert isinstance(res.fig, go.Figure)


def test_by_group_trend(sample_df):
    res = prepare_by_group_trend_graph(sample_df, "year", "grp", "x1", error_bars=True)
    assert len(res.fig.data) == sample_df["grp"].nunique()


def test_by_group_violin(sample_df):
    fig = prepare_by_group_violin_graph(sample_df, "grp", "x1")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == sample_df["grp"].nunique()


def test_histogram_bins(sample_df):
    res = prepare_histogram(sample_df, "x3", bins=15)
    assert res.df.shape[0] == 15
    assert res.df["count"].sum() == sample_df["x3"].notna().sum()


def test_bar_chart_counts(sample_df):
    res = prepare_bar_chart(sample_df, "grp")
    assert res.df["count"].sum() == len(sample_df)


def test_missing_values_graph(kuznets):
    fig = prepare_missing_values_graph(kuznets, ts_id="year")
    assert isinstance(fig, go.Figure)
    figb = prepare_missing_values_graph(kuznets, ts_id="year", binary=True)
    assert isinstance(figb, go.Figure)


def test_missing_values_requires_clean_ts(sample_df):
    df = sample_df.copy()
    df.loc[0, "year"] = np.nan
    with pytest.raises(ValueError):
        prepare_missing_values_graph(df, ts_id="year")


def test_scatter_default_alpha_and_loess(sample_df):
    from expdpy.scatter import _default_alpha

    fig = prepare_scatter_plot(sample_df, "x1", "x2", loess=1)
    assert isinstance(fig, go.Figure)
    assert any(t.name == "loess" for t in fig.data)
    assert fig.data[0].marker.opacity == pytest.approx(_default_alpha(len(sample_df)))


def test_scatter_small_sample_alpha_one(sample_df):
    fig = prepare_scatter_plot(sample_df.head(50), "x1", "x2")
    assert fig.data[0].marker.opacity == pytest.approx(1.0)


def test_scatter_color_categorical(sample_df):
    fig = prepare_scatter_plot(sample_df, "x1", "x2", color="grp")
    names = {t.name for t in fig.data}
    assert {"A", "B", "C"}.issubset(names)


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
