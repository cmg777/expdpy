"""Tests for prepare_cre_table (Correlated Random Effects / the Mundlak device)."""

from __future__ import annotations

import pytest

import expdpy as ex
from expdpy import CRETableResult


def test_cre_recovers_within_estimate(kuznets):
    cre = ex.prepare_cre_table(
        kuznets, dv="gini_regional", idvs=["log_gdp_pc"], entity="country", time="year"
    )
    assert isinstance(cre, CRETableResult)
    cre_b = float(cre.df.query("term == 'log_gdp_pc'")["Estimate"].iloc[0])
    fe_b = float(
        ex.prepare_regression_table(
            kuznets, dvs="gini_regional", idvs=["log_gdp_pc"], feffects=["country"]
        )
        .models[0]
        .coef()["log_gdp_pc"]
    )
    assert cre_b == pytest.approx(fe_b, rel=1e-4)  # Mundlak equivalence


def test_cre_mean_terms_present(kuznets):
    cre = ex.prepare_cre_table(
        kuznets,
        dv="gini_regional",
        idvs=["log_gdp_pc", "log_gdp_pc_sq"],
        entity="country",
        time="year",
    )
    assert {"log_gdp_pc_mean", "log_gdp_pc_sq_mean"} <= set(cre.df["term"])
    assert cre.models[0]._cre_mundlak_df == 2


def test_cre_mundlak_test_matches_hausman_unadjusted(kuznets):
    # The regression-form Hausman test (a Wald test on the mean terms) matches the classic
    # Hausman test when both use the model's own (unadjusted) covariance.
    cre = ex.prepare_cre_table(
        kuznets,
        dv="gini_regional",
        idvs=["log_gdp_pc", "log_gdp_pc_sq"],
        entity="country",
        time="year",
        cov_type="unadjusted",
    )
    haus = ex.prepare_hausman_test(
        kuznets,
        dv="gini_regional",
        idvs=["log_gdp_pc", "log_gdp_pc_sq"],
        entity="country",
        time="year",
    )
    # Asymptotically equivalent, not identical in finite samples: agree to the same order of
    # magnitude (both strongly reject), unlike the clustered Wald test (p ≈ 0.24 here).
    assert cre.models[0]._cre_mundlak_p == pytest.approx(haus.p_value, rel=0.5)
    assert cre.models[0]._cre_mundlak_p < 0.01 and haus.p_value < 0.01


def test_cre_interpret_and_explain(kuznets):
    cre = ex.prepare_cre_table(
        kuznets, dv="gini_regional", idvs=["log_gdp_pc"], entity="country", time="year"
    )
    text = cre.interpret()
    assert "Mundlak" in text
    assert "effect of" not in text.lower()  # associational, no causal language
    assert cre.explain().topic == "correlated_random_effects"
    assert ex.explain("mundlak").topic == "correlated_random_effects"


def test_cre_requires_idvs(kuznets):
    with pytest.raises(ValueError, match="independent variable"):
        ex.prepare_cre_table(
            kuznets, dv="gini_regional", idvs=[], entity="country", time="year"
        )
