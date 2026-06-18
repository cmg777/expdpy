"""Tests for descriptive, correlation and extreme-observation tables."""

from __future__ import annotations

import numpy as np
import pytest
from great_tables import GT

from expdpy import (
    prepare_correlation_table,
    prepare_descriptive_table,
    prepare_ext_obs_table,
)


def test_descriptive_matches_r_goldens(sample_df, goldens):
    res = prepare_descriptive_table(sample_df[["x1", "x2", "x3"]])
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


def test_descriptive_digits_none_drops_column(sample_df):
    res = prepare_descriptive_table(
        sample_df[["x1", "x2"]], digits=(0, 3, None, 3, 3, 3, 3, 3)
    )
    assert "Std. dev." not in res.df.columns


def test_descriptive_requires_numeric():
    import pandas as pd

    with pytest.raises(ValueError):
        prepare_descriptive_table(pd.DataFrame({"g": ["a", "b", "c"]}))


def test_correlation_matches_r_goldens(sample_df, goldens):
    res = prepare_correlation_table(sample_df[["x1", "x2", "x3"]])
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
    res = prepare_correlation_table(sample_df[["x1", "x2", "x3"]])
    assert res.df_corr.shape == (3, 3)
    assert np.allclose(np.diag(res.df_corr.to_numpy()), 1.0)
    assert isinstance(res.gt, GT)


def test_correlation_requires_five_obs():
    import pandas as pd

    df = pd.DataFrame({"a": [1.0, 2, 3], "b": [3.0, 2, 1]})
    with pytest.raises(ValueError):
        prepare_correlation_table(df)


def test_ext_obs_top_and_bottom(russell):
    res = prepare_ext_obs_table(
        russell, n=5, cs_id=["coid", "coname"], ts_id="period", var="sales"
    )
    assert res.df.shape[0] == 10
    top = res.df["sales"].iloc[:5].to_numpy()
    bottom = res.df["sales"].iloc[5:].to_numpy()
    assert top.min() >= bottom.max()  # the top block dominates the bottom block
    assert isinstance(res.gt, GT)


def test_ext_obs_n_too_large():
    import pandas as pd

    df = pd.DataFrame({"v": [1.0, 2, 3]})
    with pytest.raises(ValueError):
        prepare_ext_obs_table(df, n=5)
