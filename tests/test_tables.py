"""Tests for descriptive, correlation and extreme-observation tables."""

from __future__ import annotations

import numpy as np
import pytest
from great_tables import GT

from expdpy import (
    explore_correlation_table,
    explore_descriptive_table,
    explore_ext_obs_table,
)


def test_descriptive_matches_r_goldens(sample_df, goldens):
    res = explore_descriptive_table(sample_df[["x1", "x2", "x3"]])
    assert isinstance(res.gt, GT)
    for var, gold in goldens["descriptive"].items():
        row = res.df.loc[var]
        assert row["N"] == gold["N"]
        assert row["Mean"] == pytest.approx(gold["mean"], rel=1e-9)
        assert row["Std. dev."] == pytest.approx(gold["sd"], rel=1e-9)
        assert row["Min."] == pytest.approx(gold["min"], rel=1e-9)
        assert row["25 %"] == pytest.approx(gold["q25"], rel=1e-9)
        assert row["Median"] == pytest.approx(gold["median"], rel=1e-9)
        assert row["75 %"] == pytest.approx(gold["q75"], rel=1e-9)
        assert row["Max."] == pytest.approx(gold["max"], rel=1e-9)


def test_descriptive_df_always_has_all_stats(sample_df):
    # `.df` carries all eight statistics regardless of which are rendered; no panel here.
    res = explore_descriptive_table(sample_df[["x1", "x2", "x3"]], stats=("Mean",))
    assert list(res.df.columns) == [
        "N",
        "Mean",
        "Std. dev.",
        "Min.",
        "25 %",
        "Median",
        "75 %",
        "Max.",
    ]
    assert res.by_period is None


def test_descriptive_stats_selection_and_validation(sample_df):
    assert isinstance(
        explore_descriptive_table(sample_df[["x1", "x2"]], stats=("Mean", "Median")).gt,
        GT,
    )
    with pytest.raises(ValueError):
        explore_descriptive_table(sample_df[["x1"]], stats=("Mean", "Bogus"))


def test_descriptive_digits_scalar_and_mapping(sample_df):
    # Both a scalar and a per-statistic mapping render without error.
    explore_descriptive_table(sample_df[["x1", "x2"]], digits=2)
    explore_descriptive_table(sample_df[["x1", "x2"]], digits={"Mean": 1})


def test_descriptive_by_period_first_and_last(sample_df):
    res = explore_descriptive_table(sample_df, entity="firm", time="year")
    assert res.by_period is not None
    years = sorted(sample_df["year"].unique())
    assert sorted(res.by_period["period"].unique()) == [years[0], years[-1]]
    # The entity/time ids are coordinates, never summarized as variables.
    summarized = set(res.by_period["variable"].unique())
    assert "year" not in summarized and "firm" not in summarized
    html = res.gt.as_raw_html()
    assert str(years[0]) in html and str(years[-1]) in html


def test_descriptive_by_period_explicit_periods(sample_df):
    chosen = [2014, 2017, 2021]
    res = explore_descriptive_table(
        sample_df, entity="firm", time="year", periods=chosen
    )
    assert sorted(res.by_period["period"].unique()) == chosen


def test_descriptive_by_period_absent_period_warns(sample_df):
    with pytest.warns(UserWarning):
        res = explore_descriptive_table(sample_df, time="year", periods=[2014, 1900])
    assert sorted(res.by_period["period"].unique()) == [2014]
    with pytest.raises(ValueError):
        explore_descriptive_table(sample_df, time="year", periods=[1900, 1901])


def test_descriptive_missing_note(sample_df):
    df = sample_df.copy()
    df.loc[df.index[:3], "x1"] = np.nan
    html = explore_descriptive_table(df[["x1", "x2", "x3"]]).gt.as_raw_html()
    assert "missing data" in html.lower()


def test_descriptive_declared_panel_columns_absent_falls_back(sample_df):
    # A declared panel whose entity/time columns were dropped by a column subset must not
    # raise — the table falls back to the flat (overall) layout.
    from expdpy import set_panel

    panel = set_panel(sample_df.copy(), entity="firm", time="year")
    subset = panel[["x1", "x2", "x3"]]  # carries panel attrs but drops firm/year
    res = explore_descriptive_table(subset)
    assert res.by_period is None
    # Mixed case: a valid explicit id alongside a stored id whose column was dropped must
    # still fall back (the stored, not the explicit, column is the missing one).
    mixed = panel[["firm", "x1", "x2"]]  # keeps entity, drops time
    assert explore_descriptive_table(mixed, entity="firm").by_period is None
    # An explicitly-named missing column, by contrast, still raises.
    with pytest.raises(ValueError):
        explore_descriptive_table(subset, time="year")


def test_descriptive_requires_numeric():
    import pandas as pd

    with pytest.raises(ValueError):
        explore_descriptive_table(pd.DataFrame({"g": ["a", "b", "c"]}))


def test_correlation_matches_r_goldens(sample_df, goldens):
    res = explore_correlation_table(sample_df[["x1", "x2", "x3"]])
    # Pearson is above the diagonal (x1=row0/x2=col1), Spearman below.
    assert res.df_corr.loc["x1", "x2"] == pytest.approx(
        goldens["correlation"]["pearson"]["x1_x2"]["r"], rel=1e-9
    )
    assert res.df_corr.loc["x2", "x1"] == pytest.approx(
        goldens["correlation"]["spearman"]["x1_x2"]["r"], rel=1e-9
    )
    assert res.df_prob.loc["x1", "x2"] == pytest.approx(
        goldens["correlation"]["pearson"]["x1_x2"]["p"], rel=1e-6
    )
    assert (
        int(res.df_n.loc["x1", "x2"]) == goldens["correlation"]["pearson"]["x1_x2"]["n"]
    )


def test_correlation_diagonal_and_shape(sample_df):
    res = explore_correlation_table(sample_df[["x1", "x2", "x3"]])
    assert res.df_corr.shape == (3, 3)
    assert np.allclose(np.diag(res.df_corr.to_numpy()), 1.0)
    assert isinstance(res.gt, GT)


def test_correlation_requires_five_obs():
    import pandas as pd

    df = pd.DataFrame({"a": [1.0, 2, 3], "b": [3.0, 2, 1]})
    with pytest.raises(ValueError):
        explore_correlation_table(df)


def test_ext_obs_top_and_bottom(kuznets):
    res = explore_ext_obs_table(
        kuznets, n=5, var="gini_regional", entity=["country"], time="year"
    )
    assert res.df.shape[0] == 10
    top = res.df["gini_regional"].iloc[:5].to_numpy()
    bottom = res.df["gini_regional"].iloc[5:].to_numpy()
    assert top.min() >= bottom.max()  # the top block dominates the bottom block
    assert isinstance(res.gt, GT)


def test_ext_obs_n_too_large():
    import pandas as pd

    df = pd.DataFrame({"v": [1.0, 2, 3]})
    with pytest.raises(ValueError):
        explore_ext_obs_table(df, n=5)
