"""Tests for prepare_coefficient_plot (themed coefficient plots)."""

from __future__ import annotations

import plotly.graph_objects as go
import pytest

import expdpy as ex
from expdpy import CoefficientPlotResult

_DF_COLS = {"model", "term", "estimate", "se", "ci_lower", "ci_upper"}


@pytest.fixture(scope="module")
def pooled(kuznets):
    return ex.prepare_regression_table(
        kuznets,
        dvs="gini_regional",
        idvs=["log_gdp_pc", "log_gdp_pc_sq", "log_gdp_pc_cu"],
    )


@pytest.fixture(scope="module")
def fe(kuznets):
    return ex.prepare_regression_table(
        kuznets,
        dvs="gini_regional",
        idvs=["log_gdp_pc", "log_gdp_pc_sq", "log_gdp_pc_cu"],
        feffects=["country", "year"],
        clusters=["country"],
    )


def test_accepts_regression_table_result(pooled):
    res = ex.prepare_coefficient_plot(pooled)
    assert isinstance(res, CoefficientPlotResult)
    assert isinstance(res.fig, go.Figure)
    assert set(res.df.columns) == _DF_COLS
    assert len(res.fig.data) == 1
    assert res.fig.data[0].showlegend is False


def test_accepts_single_model_and_list(pooled):
    one = ex.prepare_coefficient_plot(pooled.models[0])
    assert len(one.fig.data) == 1
    two = ex.prepare_coefficient_plot([pooled.models[0], pooled.models[0]])
    assert len(two.fig.data) == 2


def test_drop_intercept_default_and_override(pooled):
    default = ex.prepare_coefficient_plot(pooled)
    assert "Intercept" not in set(default.df["term"])
    kept = ex.prepare_coefficient_plot(pooled, drop_intercept=False)
    assert "Intercept" in set(kept.df["term"])


def test_keep_filters_terms(pooled):
    res = ex.prepare_coefficient_plot(pooled, keep=["log_gdp_pc_sq"])
    assert set(res.df["term"]) == {"log_gdp_pc_sq"}


def test_multi_model_labels_and_dodge(pooled, fe):
    res = ex.prepare_coefficient_plot(
        [pooled.models[0], fe.models[0]],
        model_labels=["Pooled OLS", "Two-way FE"],
        keep=["log_gdp_pc"],
    )
    assert [t.name for t in res.fig.data] == ["Pooled OLS", "Two-way FE"]
    assert all(t.showlegend for t in res.fig.data)
    # the two models are dodged onto slightly different categorical positions
    assert res.fig.data[0].y[0] != res.fig.data[1].y[0]


def test_model_labels_length_mismatch_raises(pooled):
    with pytest.raises(ValueError, match="model_labels"):
        ex.prepare_coefficient_plot(pooled, model_labels=["only one", "too many"])


def test_joint_bands_at_least_as_wide_as_pointwise(pooled):
    point = ex.prepare_coefficient_plot(pooled, drop_intercept=False)
    joint = ex.prepare_coefficient_plot(pooled, drop_intercept=False, joint=True)
    w_point = (point.df["ci_upper"] - point.df["ci_lower"]).mean()
    w_joint = (joint.df["ci_upper"] - joint.df["ci_lower"]).mean()
    assert w_joint >= w_point


def test_alpha_widens_intervals(pooled):
    ci95 = ex.prepare_coefficient_plot(pooled, alpha=0.05)
    ci90 = ex.prepare_coefficient_plot(pooled, alpha=0.10)
    w95 = (ci95.df["ci_upper"] - ci95.df["ci_lower"]).mean()
    w90 = (ci90.df["ci_upper"] - ci90.df["ci_lower"]).mean()
    assert w95 > w90  # 95% intervals are wider than 90% intervals


def test_orientation_switches_error_axis(pooled):
    horizontal = ex.prepare_coefficient_plot(pooled, horizontal=True)
    vertical = ex.prepare_coefficient_plot(pooled, horizontal=False)
    assert horizontal.fig.data[0].error_x.array is not None
    assert vertical.fig.data[0].error_y.array is not None


def test_zero_reference_line_present(pooled):
    res = ex.prepare_coefficient_plot(pooled)
    assert len(res.fig.layout.shapes) == 1  # the dashed zero line


def test_coef_labels_rename_ticks(pooled):
    res = ex.prepare_coefficient_plot(
        pooled, coef_labels={"log_gdp_pc": "log GDP per capita"}
    )
    assert "log GDP per capita" in list(res.fig.layout.yaxis.ticktext)
