"""Tests for prepare_estimation (IV/2SLS, GLM, VCOV options, model comparison)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import expdpy as ex
from expdpy import EstimationResult

_FORBIDDEN = ("causes", "caused by", "effect of")


def _no_causal(text: str) -> bool:
    low = text.lower()
    return not any(b in low for b in _FORBIDDEN)


@pytest.fixture(scope="module")
def iv_df() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    n = 800
    z1, z2 = rng.normal(size=n), rng.normal(size=n)
    d = 0.8 * z1 + 0.4 * z2 + rng.normal(size=n)  # endogenous
    y = 2.0 * d + rng.normal(size=n)  # true effect of d == 2.0
    return pd.DataFrame(
        {"y": y, "d": d, "z1": z1, "z2": z2, "firm": rng.integers(0, 50, n)}
    )


@pytest.fixture(scope="module")
def counts(kuznets) -> pd.DataFrame:
    return kuznets.assign(
        gini_count=(kuznets["gini_regional"] * 100).round().astype(int),
        high=(kuznets["gini_regional"] > kuznets["gini_regional"].median()).astype(int),
    )


# --- estimators ----------------------------------------------------------------------


def test_iv_recovers_effect(iv_df):
    res = ex.prepare_estimation(
        iv_df,
        dv="y",
        model="iv",
        endog=["d"],
        instruments=["z1", "z2"],
        cluster=["firm"],
    )
    assert isinstance(res, EstimationResult)
    assert res.model_kind == "iv"
    est = float(res.df.query("term == 'd'")["Estimate"].iloc[0])
    assert est == pytest.approx(2.0, abs=0.15)


def test_iv_requires_endog_and_instruments(iv_df):
    with pytest.raises(ValueError, match="endog"):
        ex.prepare_estimation(iv_df, dv="y", idvs=["z1"], model="iv")


def test_stepwise_csw_makes_nested_models(kuznets):
    res = ex.prepare_estimation(
        kuznets,
        dv="gini_regional",
        idvs=["log_gdp_pc", "log_gdp_pc_sq", "log_gdp_pc_cu"],
        stepwise="csw",
    )
    assert len(res.models) == 3
    assert len(res.fit_stats) == 3


def test_multi_lhs(kuznets):
    res = ex.prepare_estimation(
        kuznets, dv=["gini_regional", "log_gdp_pc"], idvs=["population"]
    )
    assert len(res.models) == 2


def test_vcov_hetero_differs_from_iid(kuznets):
    iid = ex.prepare_estimation(kuznets, dv="gini_regional", idvs=["log_gdp_pc"])
    hc = ex.prepare_estimation(
        kuznets, dv="gini_regional", idvs=["log_gdp_pc"], vcov="hetero"
    )
    se_iid = float(iid.df.query("term=='log_gdp_pc'")["Std. Error"].iloc[0])
    se_hc = float(hc.df.query("term=='log_gdp_pc'")["Std. Error"].iloc[0])
    assert se_iid != pytest.approx(se_hc, rel=1e-6)


def test_cluster_shortcut_selects_crv1(kuznets):
    res = ex.prepare_estimation(
        kuznets, dv="gini_regional", idvs=["log_gdp_pc"], cluster=["country"]
    )
    assert res.models[0]._vcov_type_detail == "CRV1"


def test_newey_west_and_driscoll_kraay_run(kuznets):
    for v in ("NW", "DK"):
        res = ex.prepare_estimation(
            kuznets,
            dv="gini_regional",
            idvs=["log_gdp_pc"],
            vcov=v,
            time_id="year",
            panel_id="country",
        )
        se = float(res.df.query("term=='log_gdp_pc'")["Std. Error"].iloc[0])
        assert np.isfinite(se) and se > 0


def test_poisson_and_logit_fit_stats(counts):
    pois = ex.prepare_estimation(
        counts,
        dv="gini_count",
        idvs=["log_gdp_pc"],
        model="poisson",
        feffects=["country"],
    )
    assert "pseudo_r2" in pois.fit_stats.columns
    logit = ex.prepare_estimation(counts, dv="high", idvs=["log_gdp_pc"], model="logit")
    assert logit.model_kind == "logit"


def test_few_clusters_note(kuznets):
    res = ex.prepare_estimation(
        kuznets, dv="gini_regional", idvs=["log_gdp_pc"], cluster=["continent"]
    )
    assert any("clusters" in n for n in res.notes)


def test_missing_column_raises(kuznets):
    with pytest.raises(KeyError, match="nope"):
        ex.prepare_estimation(kuznets, dv="gini_regional", idvs=["nope"])


# --- interpretation / explanation ----------------------------------------------------


def test_interpret_iv_mentions_late_and_is_associational(iv_df):
    res = ex.prepare_estimation(
        iv_df, dv="y", model="iv", endog=["d"], instruments=["z1", "z2"]
    )
    text = res.interpret()
    assert "local average treatment effect" in text.lower()
    assert _no_causal(text)
    assert res.explain().topic == "iv"


def test_interpret_poisson_is_percent_change(counts):
    res = ex.prepare_estimation(
        counts, dv="gini_count", idvs=["log_gdp_pc"], model="poisson"
    )
    text = res.interpret()
    assert "%" in text
    assert res.explain().topic == "glm"
    assert _no_causal(text)


def test_interpret_logit_is_odds_ratio(counts):
    res = ex.prepare_estimation(counts, dv="high", idvs=["log_gdp_pc"], model="logit")
    assert "odds" in res.interpret().lower()


def test_glance_returns_fit_stats(kuznets):
    res = ex.prepare_estimation(kuznets, dv="gini_regional", idvs=["log_gdp_pc"])
    assert res.glance() is res.fit_stats
    assert res.tidy() is res.df
