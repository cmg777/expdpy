"""Tests for the post-estimation helpers (fixef plot, predictions, joint test)."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import pytest

import expdpy as ex


@pytest.fixture(scope="module")
def fe_model(kuznets):
    return ex.analyze_regression_table(
        kuznets,
        dvs="gini_regional",
        idvs=["log_gdp_pc", "log_gdp_pc_sq", "log_gdp_pc_cu"],
        feffects=["country", "year"],
        clusters=["country"],
    )


def test_fixef_plot_default_dimension(fe_model):
    res = ex.analyze_fixef_plot(fe_model)
    assert set(res.df.columns) == {"fixef", "level", "value"}
    assert isinstance(res.fig, go.Figure)
    assert len(res.df) > 0


def test_fixef_plot_select_dimension_and_top_n(fe_model):
    res = ex.analyze_fixef_plot(fe_model, fixef="country", top_n=10)
    assert (res.df["fixef"] == "country").all()
    assert len(res.df) == 10  # top_n extremes


def test_fixef_plot_unknown_dimension_raises(fe_model):
    with pytest.raises(KeyError, match="nope"):
        ex.analyze_fixef_plot(fe_model, fixef="nope")


def test_fixef_plot_without_fe_raises(kuznets):
    pooled = ex.analyze_regression_table(
        kuznets, dvs="gini_regional", idvs=["log_gdp_pc"]
    )
    with pytest.raises(ValueError, match="no fixed effects"):
        ex.analyze_fixef_plot(pooled)


def test_predictions_in_sample(fe_model):
    res = ex.analyze_predictions(fe_model)
    assert set(res.df.columns) == {"actual", "predicted", "residual"}
    assert len(res.df) == fe_model.models[0]._N
    # actual == predicted + residual by construction
    np.testing.assert_allclose(
        res.df["actual"], res.df["predicted"] + res.df["residual"], atol=1e-8
    )


def test_predictions_newdata(kuznets):
    reg = ex.analyze_regression_table(kuznets, dvs="gini_regional", idvs=["log_gdp_pc"])
    nd = kuznets.head(15)
    res = ex.analyze_predictions(reg, newdata=nd)
    assert list(res.df.columns) == ["predicted"]
    assert len(res.df) == 15


def test_joint_test_named_terms(fe_model):
    res = ex.analyze_joint_test(
        fe_model, ["log_gdp_pc", "log_gdp_pc_sq", "log_gdp_pc_cu"]
    )
    assert res.hypotheses == ("log_gdp_pc", "log_gdp_pc_sq", "log_gdp_pc_cu")
    assert 0.0 <= res.p_value <= 1.0
    assert "joint" in res.summary().lower()


def test_joint_test_all_coefficients(kuznets):
    reg = ex.analyze_regression_table(kuznets, dvs="gini_regional", idvs=["log_gdp_pc"])
    res = ex.analyze_joint_test(reg)
    assert np.isfinite(res.statistic)


def test_joint_test_unknown_coef_raises(fe_model):
    with pytest.raises(KeyError, match="not in model"):
        ex.analyze_joint_test(fe_model, ["nope"])
