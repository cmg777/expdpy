"""Characterization + unit tests for the shared estimation engine.

The characterization tests pin the *exact* coefficients and standard errors that
``analyze_regression_table`` produced before the engine was extracted into
``expdpy._estimation``. They are the safety net for the refactor: the public function
must keep returning byte-identical numbers (same dropna order, category casting, dedupe,
``_SSC`` and ``{"CRV1": ...}`` vcov path).

The builder tests exercise the pure ``_formula`` / ``_vcov`` helpers directly.
"""

from __future__ import annotations

import pytest

from expdpy import analyze_regression_table
from expdpy._estimation import (
    ModelSpec,
    VCovSpec,
    build_formula,
    build_vcov,
)

# Golden values captured from the pre-refactor implementation (pyfixest 0.60.0).
_KUZNETS_POOLED = {
    "coef": {
        "Intercept": -18.4903952884,
        "log_gdp_pc": 6.3854377822,
        "log_gdp_pc_sq": -0.7109331919,
        "log_gdp_pc_cu": 0.0259187557,
    },
    "se": {
        "Intercept": 0.4019981171,
        "log_gdp_pc": 0.1340495323,
        "log_gdp_pc_sq": 0.0146800629,
        "log_gdp_pc_cu": 0.0005284815,
    },
    "N": 880,
}
_KUZNETS_TWFE_CLUSTER = {
    "coef": {
        "log_gdp_pc": 6.4113204009,
        "log_gdp_pc_sq": -0.7147720135,
        "log_gdp_pc_cu": 0.0260812981,
    },
    "se": {
        "log_gdp_pc": 0.2096102749,
        "log_gdp_pc_sq": 0.022799823,
        "log_gdp_pc_cu": 0.0008141253,
    },
    "N": 880,
    "r2_within": 0.5205724631,
}
_SAMPLE_FE_CLUSTER = {
    "coef": {"x1": 0.6282911842, "x3": -0.0418463079},
    "se": {"x1": 0.105828557, "x3": 0.0326781904},
    "N": 160,
    "r2_within": 0.2386191082,
}


def _assert_model_matches(model, golden) -> None:
    coef = model.coef()
    se = model.se()
    for term, value in golden["coef"].items():
        assert float(coef[term]) == pytest.approx(value, rel=1e-8, abs=1e-10)
    for term, value in golden["se"].items():
        assert float(se[term]) == pytest.approx(value, rel=1e-8, abs=1e-10)
    assert int(model._N) == golden["N"]
    if "r2_within" in golden:
        assert float(model._r2_within) == pytest.approx(golden["r2_within"], rel=1e-8)


def test_characterize_kuznets_pooled(kuznets):
    res = analyze_regression_table(
        kuznets,
        dvs="gini_regional",
        idvs=["log_gdp_pc", "log_gdp_pc_sq", "log_gdp_pc_cu"],
    )
    _assert_model_matches(res.models[0], _KUZNETS_POOLED)


def test_characterize_kuznets_twfe_cluster(kuznets):
    res = analyze_regression_table(
        kuznets,
        dvs="gini_regional",
        idvs=["log_gdp_pc", "log_gdp_pc_sq", "log_gdp_pc_cu"],
        feffects=["country", "year"],
        clusters=["country"],
    )
    _assert_model_matches(res.models[0], _KUZNETS_TWFE_CLUSTER)


def test_characterize_sample_fe_cluster(sample_df):
    res = analyze_regression_table(
        sample_df, dvs="x2", idvs=["x1", "x3"], feffects=["firm"], clusters=["firm"]
    )
    _assert_model_matches(res.models[0], _SAMPLE_FE_CLUSTER)


def test_characterize_byvar_levels(sample_df):
    res = analyze_regression_table(sample_df, dvs="x2", idvs=["x1"], byvar="grp")
    assert len(res.models) == 1 + sample_df["grp"].nunique()
    assert "Full Sample" in set(res.df["byvalue"])


# --- pure builders -------------------------------------------------------------------


def test_build_formula_ols_and_fe():
    spec = ModelSpec(dv=("y",), idvs=("x1", "x2"), feffects=("firm", "year"))
    assert build_formula(spec) == "y ~ x1 + x2 | firm + year"


def test_build_formula_intercept_only_and_multi_lhs():
    assert build_formula(ModelSpec(dv=("y",), idvs=())) == "y ~ 1"
    assert build_formula(ModelSpec(dv=("y1", "y2"), idvs=("x",))) == "y1 + y2 ~ x"


def test_build_formula_stepwise():
    spec = ModelSpec(dv=("y",), idvs=("x1", "x2", "x3"), stepwise="csw")
    assert build_formula(spec) == "y ~ csw(x1, x2, x3)"


def test_build_vcov_variants():
    assert build_vcov(VCovSpec(kind="iid")) == ("iid", None)
    assert build_vcov(VCovSpec(kind="HC1")) == ("HC1", None)
    assert build_vcov(VCovSpec(kind="CRV1", cluster=("firm", "year"))) == (
        {"CRV1": "firm + year"},
        None,
    )
    vcov, kw = build_vcov(VCovSpec(kind="NW", time_id="t", panel_id="id", lag=2))
    assert vcov == "NW"
    assert kw == {"time_id": "t", "panel_id": "id", "lag": 2}


def test_build_vcov_errors():
    with pytest.raises(ValueError, match="cluster"):
        build_vcov(VCovSpec(kind="CRV1"))
    with pytest.raises(ValueError, match="time_id"):
        build_vcov(VCovSpec(kind="DK", time_id="t"))
