"""Tests for prepare_fwl_plot (Frisch-Waugh-Lovell residualized scatter)."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import pytest
import statsmodels.api as sm

from expdpy import FWLPlotResult, prepare_fwl_plot, prepare_regression_table


def test_fwl_slope_equals_full_coef_with_controls_and_fe(sample_df):
    """FWL theorem: residual-scatter slope == the focal coefficient in the full model."""
    res = prepare_fwl_plot(
        sample_df,
        dv="x2",
        var="x1",
        controls=["x3"],
        feffects=["firm"],
        clusters=["firm"],
    )
    tab = prepare_regression_table(
        sample_df, dvs="x2", idvs=["x1", "x3"], feffects=["firm"], clusters=["firm"]
    )
    coef = float(tab.models[0].coef()["x1"])
    se = float(tab.models[0].se()["x1"])
    assert isinstance(res, FWLPlotResult)
    assert res.slope == pytest.approx(coef, rel=1e-8)  # point estimates match exactly
    assert res.se == pytest.approx(se, rel=1e-8)  # clustered SE matches the table


def test_fwl_is_conditional_on_fixed_effects(sample_df):
    """The scatter is conditional on the fixed effects, not just the controls.

    With fixed effects the slope must equal the within (group-demeaned) OLS slope, and it
    must differ from the unconditional slope.
    """
    sub = sample_df[["x1", "x2", "firm"]].dropna()
    xd = sub["x1"] - sub.groupby("firm")["x1"].transform("mean")
    yd = sub["x2"] - sub.groupby("firm")["x2"].transform("mean")
    within = float(sm.OLS(yd, sm.add_constant(xd)).fit().params.iloc[1])

    fe = prepare_fwl_plot(sample_df, dv="x2", var="x1", feffects=["firm"])
    assert fe.slope == pytest.approx(within, rel=1e-6)
    assert np.isfinite(fe.r2_within)
    assert fe.fig.layout.xaxis.title.text == "Residualized x1"

    no_fe = prepare_fwl_plot(sample_df, dv="x2", var="x1")
    assert fe.slope != pytest.approx(no_fe.slope, rel=1e-3)  # FE actually changes it


def test_fwl_no_controls_matches_simple_ols(sample_df):
    """With no controls and no FE the slope is the simple OLS slope (raw scatter)."""
    res = prepare_fwl_plot(sample_df, dv="x2", var="x1")
    sm_fit = sm.OLS(sample_df["x2"], sm.add_constant(sample_df[["x1"]])).fit()
    assert res.slope == pytest.approx(sm_fit.params["x1"], rel=1e-8)
    assert res.fig.layout.xaxis.title.text == "x1"  # not "Residualized x1"
    assert res.fig.layout.yaxis.title.text == "x2"
    assert np.isnan(res.r2_within)


def test_fwl_figure_structure_and_band(sample_df):
    """The figure has band + fit + points traces, an annotation, and an ordered band."""
    res = prepare_fwl_plot(sample_df, dv="x2", var="x1", controls=["x3"])
    assert isinstance(res.fig, go.Figure)
    assert [t.name for t in res.fig.data] == ["ci", "fit", "points"]
    assert len(res.fig.layout.annotations) == 1
    assert "Slope" in res.fig.layout.annotations[0].text
    assert set(res.df.columns) == {"x_resid", "y_resid", "fit", "lwr", "upr"}
    assert (res.df["lwr"] <= res.df["fit"] + 1e-9).all()
    assert (res.df["fit"] <= res.df["upr"] + 1e-9).all()
    assert res.df["x_resid"].is_monotonic_increasing


def test_fwl_n_sample_reduces_points_not_slope(sample_df):
    """Subsampling shrinks the scatter trace but leaves the fit (slope, df) unchanged."""
    full = prepare_fwl_plot(
        sample_df, dv="x2", var="x1", controls=["x3"], n_sample=None
    )
    sub = prepare_fwl_plot(sample_df, dv="x2", var="x1", controls=["x3"], n_sample=20)
    assert len(full.fig.data[2].x) == full.n_obs
    assert len(sub.fig.data[2].x) == 20
    assert sub.slope == pytest.approx(full.slope, rel=1e-12)
    assert len(sub.df) == len(full.df) == full.n_obs


def test_fwl_focal_var_in_controls_raises(sample_df):
    with pytest.raises(ValueError, match="must not also be a control"):
        prepare_fwl_plot(sample_df, dv="x2", var="x1", controls=["x1", "x3"])


def test_fwl_missing_column_raises(sample_df):
    with pytest.raises(KeyError, match="nope"):
        prepare_fwl_plot(sample_df, dv="x2", var="nope")
