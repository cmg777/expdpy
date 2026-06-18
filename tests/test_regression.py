"""Tests for prepare_regression_table (OLS / FE / clustered SEs via pyfixest)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from expdpy import prepare_regression_table


def test_ols_coefficients_match_statsmodels(sample_df):
    res = prepare_regression_table(sample_df, dvs="x2", idvs=["x1"])
    coef = float(res.models[0].coef()["x1"])
    sm_fit = sm.OLS(sample_df["x2"], sm.add_constant(sample_df[["x1"]])).fit()
    assert coef == pytest.approx(sm_fit.params["x1"], rel=1e-6)
    assert "Estimate" in res.df.columns


def test_fixed_effects_and_cluster(sample_df):
    res = prepare_regression_table(
        sample_df, dvs="x2", idvs=["x1", "x3"], feffects=["firm"], clusters=["firm"]
    )
    assert len(res.models) == 1
    m = res.models[0]
    assert len(sample_df) == m._N
    assert "x1" in m.coef().index


def test_fe_equal_to_cluster_no_duplicate_error(sample_df):
    # firm used as both FE and cluster must not raise a duplicate-column error.
    res = prepare_regression_table(
        sample_df, dvs="x2", idvs=["x1"], feffects=["firm"], clusters=["firm"]
    )
    assert len(res.models) == 1


def test_multiple_models(sample_df):
    res = prepare_regression_table(
        sample_df,
        dvs=["x2", "x3"],
        idvs=[["x1"], ["x1", "x2"]],
        feffects=[["firm"], [""]],
        clusters=[["firm"], [""]],
    )
    assert len(res.models) == 2


def test_byvar_path(sample_df):
    res = prepare_regression_table(sample_df, dvs="x2", idvs=["x1"], byvar="grp")
    # full sample + one per group level
    assert len(res.models) == 1 + sample_df["grp"].nunique()
    assert "Full Sample" in set(res.df["byvalue"])


def test_byvar_with_multiple_dvs_raises(sample_df):
    with pytest.raises(ValueError, match="subset multiple models"):
        prepare_regression_table(
            sample_df, dvs=["x2", "x3"], idvs=[["x1"], ["x1"]], byvar="grp"
        )


def test_non_numeric_dv_raises():
    df = pd.DataFrame({"y": ["a", "b", "a", "b"] * 5, "x": np.arange(20.0)})
    with pytest.raises(NotImplementedError):
        prepare_regression_table(df, dvs="y", idvs=["x"])


@pytest.mark.parametrize("fmt", ["gt", "md", "df"])
def test_format_variants(sample_df, fmt):
    res = prepare_regression_table(sample_df, dvs="x2", idvs=["x1"], format=fmt)
    assert res.etable is not None
