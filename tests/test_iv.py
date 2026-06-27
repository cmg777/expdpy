"""Tests for instrumental-variables / 2SLS (cross-section + panel).

The cross-section anchor is the canonical Acemoglu-Johnson-Robinson (2001) base-sample
result (the instrumented expropriation-risk slope is ~0.94); the panel anchor is the
regional-conflict first-stage F, which reproduces the published 24-40 range.
"""

from __future__ import annotations

import math

import pytest

import expdpy as ex
from expdpy._types import IVRegressionResult
from expdpy.data import load_colonial_origins, load_regional_conflict


@pytest.fixture(scope="module")
def ajr():
    return load_colonial_origins()


@pytest.fixture(scope="module")
def ajr_base(ajr):
    return ajr[ajr["base_sample"] == 1].copy()


@pytest.fixture(scope="module")
def conflict():
    return ex.set_panel(load_regional_conflict(), entity="region_id", time="year")


def _coef(result: IVRegressionResult, term: str) -> float:
    return float(result.df.set_index("term").loc[term, "Estimate"])


# --- cross-section IV (AJR) ---------------------------------------------------
def test_ajr_base_sample_just_identified(ajr_base):
    """The famous AJR Table 4 Col 1: instrumented expropriation-risk slope ~ 0.94."""
    res = ex.analyze_iv_regression(
        ajr_base,
        dv="log_gdp_pc_1995",
        endog="expropriation_risk",
        instruments="log_settler_mortality",
    )
    assert isinstance(res, IVRegressionResult)
    assert _coef(res, "expropriation_risk") == pytest.approx(0.9443, abs=2e-3)
    assert int(res.model._N) == 64
    # Just-identified: a single instrument, finite first-stage F.
    assert res.first_stage_f > 10
    assert math.isfinite(res.first_stage_p)


def test_iv_overidentified_with_control(ajr):
    """Two instruments + an exogenous control fit and over-identify the model."""
    res = ex.analyze_iv_regression(
        ajr,
        dv="log_gdp_pc_1995",
        endog="expropriation_risk",
        instruments=["log_settler_mortality", "european_settlement_1900"],
        exog=["latitude"],
    )
    assert {"expropriation_risk", "latitude"}.issubset(set(res.df["term"]))
    assert math.isfinite(res.first_stage_f)


def test_under_identified_raises(ajr_base):
    with pytest.raises(ValueError, match="under-identified"):
        ex.analyze_iv_regression(
            ajr_base,
            dv="log_gdp_pc_1995",
            endog=["expropriation_risk", "latitude"],
            instruments="log_settler_mortality",
        )


def test_missing_instrument_raises(ajr_base):
    with pytest.raises(ValueError, match="instrument"):
        ex.analyze_iv_regression(
            ajr_base, dv="log_gdp_pc_1995", endog="expropriation_risk", instruments=[]
        )


def test_non_numeric_dv_raises(ajr_base):
    df = ajr_base.assign(label=ajr_base["country"])
    with pytest.raises(NotImplementedError):
        ex.analyze_iv_regression(
            df,
            dv="label",
            endog="expropriation_risk",
            instruments="log_settler_mortality",
        )


@pytest.mark.parametrize("fmt", ["df", "md"])
def test_format_variants(ajr_base, fmt):
    res = ex.analyze_iv_regression(
        ajr_base,
        dv="log_gdp_pc_1995",
        endog="expropriation_risk",
        instruments="log_settler_mortality",
        format=fmt,
    )
    assert res.etable is not None


def test_interpret_and_explain(ajr_base):
    res = ex.analyze_iv_regression(
        ajr_base,
        dv="log_gdp_pc_1995",
        endog="expropriation_risk",
        instruments="log_settler_mortality",
    )
    text = res.interpret()
    assert "expropriation_risk" in text
    assert "first-stage F" in text
    # Associations, never causal language.
    for bad in ("causes", "caused by", "effect of", "causal effect of"):
        assert bad not in text
    assert res.explain().topic == "instrumental_variables"
    assert res.glance().loc[0, "N"] == 64


# --- panel IV (regional conflict) ---------------------------------------------
def test_panel_iv_first_stage_matches_published(conflict):
    """Two-way FE panel IV reproduces the published first-stage F (~25) and a negative slope."""
    res = ex.analyze_panel_iv_regression(
        conflict,
        dv="conflict",
        endog="log_lights_lag1",
        instruments=["rain_lag2", "drought_lag2"],
    )
    assert int(res.model._N) == 96591
    assert _coef(res, "log_lights_lag1") < 0  # more activity, less conflict
    # Published both-instruments first-stage F is 25.32.
    assert res.first_stage_f == pytest.approx(25.3, abs=1.0)
    assert bool(res.model._has_fixef)


def test_panel_iv_resolves_panel_and_oneway(conflict):
    """Panel ids resolve from set_panel; one-way (entity-only) FE also fits."""
    res = ex.analyze_panel_iv_regression(
        conflict,
        dv="conflict",
        endog="log_lights_lag1",
        instruments="rain_lag2",
        twoway=False,
        cluster_entity=False,
    )
    assert isinstance(res, IVRegressionResult)
    assert math.isfinite(res.first_stage_f)
