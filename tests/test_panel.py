"""Tests for the panel declaration helper and the within/between math primitives."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from expdpy import (
    explore_distribution_over_time,
    explore_panel_structure,
    explore_quantile_trend_plot,
    explore_scatter_plot_within_between,
    explore_spaghetti_plot,
    explore_transition_matrix,
    explore_trend_plot,
    explore_value_heatmap,
    explore_within_persistence,
    explore_xtsum_table,
    resolve_panel,
    set_panel,
)
from expdpy._panel_math import entity_means, panel_decompose, within_demean


# ----------------------------------------------------------------- set_panel / resolve_panel ---
def test_set_panel_stores_and_resolves(sample_df):
    df = set_panel(sample_df.copy(), entity="firm", time="year")
    assert resolve_panel(df) == ("firm", "year")


def test_explicit_args_win_over_attrs(sample_df):
    df = set_panel(sample_df.copy(), entity="firm", time="year")
    # an explicit override beats the stored default
    assert resolve_panel(df, entity="grp") == ("grp", "year")


def test_resolve_without_declaration_returns_none(sample_df):
    assert resolve_panel(sample_df.copy()) == (None, None)


def test_require_raises_when_unresolved(sample_df):
    with pytest.raises(ValueError):
        resolve_panel(sample_df.copy(), require_time=True)
    with pytest.raises(ValueError):
        resolve_panel(sample_df.copy(), require_entity=True)


def test_unknown_column_raises(sample_df):
    with pytest.raises(ValueError):
        set_panel(sample_df.copy(), entity="not_a_column")
    with pytest.raises(ValueError):
        resolve_panel(sample_df.copy(), time="not_a_column")


def test_set_panel_partial_update(sample_df):
    df = set_panel(sample_df.copy(), entity="firm", time="year")
    set_panel(df, entity="grp")  # only entity changes; time is preserved
    assert resolve_panel(df) == ("grp", "year")


# ------------------------------------------------------------------------- decomposition math ---
def test_panel_decompose_matches_direct_pandas(sample_df):
    d = panel_decompose(sample_df["x1"], sample_df["firm"])
    x = sample_df["x1"]
    means = x.groupby(sample_df["firm"]).mean()
    within = x - sample_df["firm"].map(means) + x.mean()

    assert d["n_obs"] == int(x.notna().sum())
    assert d["n_entities"] == sample_df["firm"].nunique()
    assert d["t_bar"] == pytest.approx(len(x) / sample_df["firm"].nunique())
    assert d["overall_mean"] == pytest.approx(x.mean())
    assert d["overall_sd"] == pytest.approx(x.std(ddof=1))
    assert d["between_sd"] == pytest.approx(means.std(ddof=1))
    assert d["within_sd"] == pytest.approx(within.std(ddof=1))


def test_panel_decompose_single_entity_between_nan():
    s = pd.Series([1.0, 2.0, 3.0])
    g = pd.Series(["a", "a", "a"])
    d = panel_decompose(s, g)
    assert d["n_entities"] == 1
    assert np.isnan(d["between_sd"])


def test_panel_decompose_all_missing():
    d = panel_decompose(pd.Series([np.nan, np.nan]), pd.Series(["a", "b"]))
    assert d["n_obs"] == 0
    assert np.isnan(d["overall_mean"])


def test_within_demean_zero_mean_per_entity(sample_df):
    dem = within_demean(sample_df, ["x1"], "firm", add_grand_mean=False)
    per_unit = dem["x1"].groupby(sample_df["firm"]).mean()
    assert np.allclose(per_unit.to_numpy(), 0.0, atol=1e-9)


def test_within_demean_grand_mean_preserved(sample_df):
    dem = within_demean(sample_df, ["x1"], "firm", add_grand_mean=True)
    assert dem["x1"].mean() == pytest.approx(sample_df["x1"].mean())


def test_entity_means_shape(sample_df):
    em = entity_means(sample_df, ["x1", "x2"], "firm")
    assert em.shape == (sample_df["firm"].nunique(), 2)


# ------------------------------------------------------------------------- xtsum + scatter ---
def test_xtsum_table_matches_decomposition(sample_df):
    res = explore_xtsum_table(sample_df, var=["x1", "x3"], entity="firm")
    assert set(res.df["component"]) == {"overall", "between", "within"}
    g = res.df[res.df["variable"] == "x1"].set_index("component")
    d = panel_decompose(sample_df["x1"], sample_df["firm"])
    assert float(g.loc["overall", "sd"]) == pytest.approx(d["overall_sd"])
    assert float(g.loc["between", "sd"]) == pytest.approx(d["between_sd"])
    assert float(g.loc["within", "sd"]) == pytest.approx(d["within_sd"])


def test_xtsum_resolves_from_set_panel(sample_df):
    df = set_panel(sample_df.copy(), entity="firm", time="year")
    res = explore_xtsum_table(df, var=["x1"])  # entity inferred from attrs
    assert res.df.shape[0] == 3


def test_xtsum_needs_two_entities():
    df = pd.DataFrame({"u": ["a", "a"], "v": [1.0, 2.0]})
    with pytest.raises(ValueError):
        explore_xtsum_table(df, var=["v"], entity="u")


def test_within_between_scatter_slopes(sample_df):
    res = explore_scatter_plot_within_between(sample_df, "x1", "x2", entity="firm")
    assert len(res.fig.data) == 6
    # the between slope equals OLS on the unit means
    em = sample_df.groupby("firm")[["x1", "x2"]].mean()
    expected = np.polyfit(em["x1"], em["x2"], 1)[0]
    assert res.slope_between == pytest.approx(expected)


# ---------------------------------------------------------------------------------- spaghetti ---
def test_spaghetti_one_line_per_unit(sample_df):
    res = explore_spaghetti_plot(sample_df, "x1", entity="firm", time="year")
    assert res.n_units == sample_df["firm"].nunique()
    assert res.n_shown == res.n_units
    assert list(res.df.columns) == ["firm", "year", "x1"]


def test_spaghetti_has_no_legend(sample_df):
    # The legend is removed entirely — including when units are highlighted.
    res = explore_spaghetti_plot(sample_df, "x1", entity="firm", time="year")
    assert res.fig.layout.showlegend is False
    hl = explore_spaghetti_plot(
        sample_df, "x1", entity="firm", time="year", highlight=[1]
    )
    assert hl.fig.layout.showlegend is False


def test_spaghetti_sampling_and_highlight(sample_df):
    with pytest.warns(UserWarning):
        res = explore_spaghetti_plot(
            sample_df, "x1", entity="firm", time="year", max_units=5, highlight=[1]
        )
    assert res.n_shown == 5
    assert res.n_units == sample_df["firm"].nunique()


# --------------------------------------------------------------------- no rangeslider ---
@pytest.mark.parametrize(
    "build",
    [
        lambda df: explore_trend_plot(df, var=["x1"], time="year").fig,
        lambda df: explore_quantile_trend_plot(df, var="x1", time="year").fig,
        lambda df: explore_spaghetti_plot(df, "x1", entity="firm", time="year").fig,
    ],
)
def test_no_rangeslider(sample_df, build):
    # The draggable x-axis rangeslider was removed from the time-series plots.
    fig = build(sample_df)
    assert fig.layout.xaxis.rangeslider.visible in (None, False)


# -------------------------------------------------------------------------- panel structure ---
def _summary(res) -> dict:
    return dict(zip(res.df_summary["statistic"], res.df_summary["value"], strict=True))


def test_panel_structure_balanced(sample_df):
    res = explore_panel_structure(sample_df, entity="firm", time="year")
    s = _summary(res)
    assert s["balanced"] is True
    assert int(s["units"]) == sample_df["firm"].nunique()
    assert int(s["periods"]) == sample_df["year"].nunique()
    assert int(s["internal gaps"]) == 0
    assert isinstance(res.fig, go.Figure)


def test_panel_structure_unbalanced_with_gap(sample_df):
    # remove one interior period for one firm to create a gap
    firm0 = sample_df["firm"].iloc[0]
    yr_mid = sorted(sample_df["year"].unique())[3]
    df = sample_df[~((sample_df["firm"] == firm0) & (sample_df["year"] == yr_mid))]
    res = explore_panel_structure(df, entity="firm", time="year")
    s = _summary(res)
    assert s["balanced"] is False
    assert int(s["internal gaps"]) >= 1


def test_value_heatmap_shape_and_standardize(sample_df):
    res = explore_value_heatmap(sample_df, "x1", entity="firm", time="year")
    assert res.df.shape == (sample_df["firm"].nunique(), sample_df["year"].nunique())
    resz = explore_value_heatmap(
        sample_df, "x1", entity="firm", time="year", standardize="by_time"
    )
    assert isinstance(resz.fig, go.Figure)


# ----------------------------------------------------------------------- dynamics + transitions ---
def test_distribution_over_time_ridgeline(sample_df):
    res = explore_distribution_over_time(sample_df, "x3", time="year")
    assert len(res.fig.data) == sample_df["year"].nunique()
    assert list(res.df.columns) == ["year", "x3"]


def test_distribution_over_time_animated_has_frames(sample_df):
    res = explore_distribution_over_time(
        sample_df, "x3", time="year", style="animated_hist"
    )
    assert len(res.fig.frames) == sample_df["year"].nunique()


def test_transition_matrix_known_counts():
    toy = pd.DataFrame(
        {
            "u": [1, 1, 1, 2, 2, 2],
            "t": [1, 2, 3, 1, 2, 3],
            "s": ["A", "B", "B", "A", "A", "B"],
        }
    )
    res = explore_transition_matrix(toy, "s", entity="u", time="t")
    assert res.states == ("A", "B")
    assert int(res.counts.loc["A", "A"]) == 1
    assert int(res.counts.loc["A", "B"]) == 2
    assert int(res.counts.loc["B", "B"]) == 1
    assert int(res.counts.loc["B", "A"]) == 0
    assert res.df.loc["A", "B"] == pytest.approx(2 / 3)
    assert res.df.loc["B", "B"] == pytest.approx(1.0)


def test_transition_matrix_skips_gaps_warns():
    # period 2 exists (unit 2), so unit 1's 1->3 jump spans a real gap and must be skipped
    toy = pd.DataFrame(
        {
            "u": [1, 1, 2, 2, 2],
            "t": [1, 3, 1, 2, 3],
            "s": ["A", "B", "A", "A", "B"],
        }
    )
    with pytest.warns(UserWarning):
        res = explore_transition_matrix(toy, "s", entity="u", time="t")
    # only unit 2's two consecutive steps are counted (A->A, A->B); unit 1's jump dropped
    assert int(res.counts.to_numpy().sum()) == 2
    assert int(res.counts.loc["A", "A"]) == 1
    assert int(res.counts.loc["A", "B"]) == 1


def test_within_persistence_positive_and_pairs(sample_df):
    res = explore_within_persistence(sample_df, "x1", entity="firm", time="year")
    assert -1.0 <= res.rho <= 1.0
    assert res.n_pairs == sample_df["firm"].nunique() * (
        sample_df["year"].nunique() - 1
    )
    assert res.demeaned is True


def test_within_persistence_known_sign():
    toy = pd.DataFrame(
        {"u": [1, 1, 1, 2, 2, 2], "t": [1, 2, 3, 1, 2, 3], "x": [1.0, 2, 3, 2, 4, 6]}
    )
    res = explore_within_persistence(toy, "x", entity="u", time="t")
    assert res.rho > 0  # within units the series trends, so positive serial correlation
    assert res.n_pairs == 4
