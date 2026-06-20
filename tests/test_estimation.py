"""Tests for prepare_estimation (OLS, VCOV options, stepwise/multi-outcome comparison)."""

from __future__ import annotations

import numpy as np
import pytest

import expdpy as ex
from expdpy import EstimationResult

_FORBIDDEN = ("causes", "caused by", "effect of")


def _no_causal(text: str) -> bool:
    low = text.lower()
    return not any(b in low for b in _FORBIDDEN)


# --- estimators ----------------------------------------------------------------------


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


def test_few_clusters_note(kuznets):
    res = ex.prepare_estimation(
        kuznets, dv="gini_regional", idvs=["log_gdp_pc"], cluster=["continent"]
    )
    assert any("clusters" in n for n in res.notes)


def test_missing_column_raises(kuznets):
    with pytest.raises(KeyError, match="nope"):
        ex.prepare_estimation(kuznets, dv="gini_regional", idvs=["nope"])


# --- interpretation / explanation ----------------------------------------------------


def test_interpret_ols_is_associational(kuznets):
    res = ex.prepare_estimation(kuznets, dv="gini_regional", idvs=["log_gdp_pc"])
    text = res.interpret()
    assert isinstance(res, EstimationResult)
    assert "OLS" in text
    assert _no_causal(text)
    assert res.explain().topic == "ols"


def test_glance_returns_fit_stats(kuznets):
    res = ex.prepare_estimation(kuznets, dv="gini_regional", idvs=["log_gdp_pc"])
    assert res.glance() is res.fit_stats
    assert res.tidy() is res.df
