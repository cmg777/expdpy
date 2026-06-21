"""Tests for the linearmodels-backed panel estimators and the Hausman test.

The happy-path tests are skipped when the optional ``linearmodels`` package is absent; the
import-guard test runs regardless (it monkeypatches the import to fail).
"""

from __future__ import annotations

import builtins
import importlib.util

import numpy as np
import pytest

import expdpy as ex

pytestmark = pytest.mark.panel

_HAS_LM = importlib.util.find_spec("linearmodels") is not None
_needs_lm = pytest.mark.skipif(not _HAS_LM, reason="requires the linearmodels extra")


@_needs_lm
def test_panel_table_four_models(kuznets):
    res = ex.analyze_panel_table(
        kuznets, dv="gini_regional", idvs=["log_gdp_pc"], entity="country", time="year"
    )
    assert len(res.models) == 4
    assert set(res.df["model"]) == {1, 2, 3, 4}
    assert {"term", "Estimate", "Std. Error", "Pr(>|t|)"} <= set(res.df.columns)


@_needs_lm
def test_fe_matches_pyfixest(kuznets):
    panel = ex.analyze_panel_table(
        kuznets,
        dv="gini_regional",
        idvs=["log_gdp_pc"],
        entity="country",
        time="year",
        models=("fe",),
    )
    lm_fe = float(panel.df.query("term == 'log_gdp_pc'")["Estimate"].iloc[0])
    pf_fe = float(
        ex.analyze_regression_table(
            kuznets,
            dvs="gini_regional",
            idvs=["log_gdp_pc"],
            feffects=["country"],
            clusters=["country"],
        )
        .models[0]
        .coef()["log_gdp_pc"]
    )
    assert lm_fe == pytest.approx(pf_fe, rel=1e-5)


@_needs_lm
def test_panel_table_interpret_and_glance(kuznets):
    res = ex.analyze_panel_table(
        kuznets, dv="gini_regional", idvs=["log_gdp_pc"], entity="country", time="year"
    )
    assert "gini_regional" in res.interpret()
    glance = res.glance()
    assert len(glance) == 4
    assert bool(glance.query("model == 3")["has_fe"].iloc[0])  # the FE model


@_needs_lm
def test_panel_table_format_df(kuznets):
    res = ex.analyze_panel_table(
        kuznets,
        dv="gini_regional",
        idvs=["log_gdp_pc"],
        entity="country",
        time="year",
        format="df",
    )
    assert res.etable is res.df


@_needs_lm
def test_hausman_runs(kuznets):
    res = ex.analyze_hausman_test(
        kuznets,
        dv="gini_regional",
        idvs=["log_gdp_pc", "log_gdp_pc_sq"],
        entity="country",
        time="year",
    )
    assert np.isfinite(res.statistic)
    assert res.df_test == 2
    assert 0.0 <= res.p_value <= 1.0
    assert list(res.fe_coefs["term"]) == ["log_gdp_pc", "log_gdp_pc_sq"]
    assert "Hausman" in res.interpret()
    assert res.explain().topic == "hausman"


@_needs_lm
def test_panel_table_missing_column_raises(kuznets):
    with pytest.raises(KeyError, match="nope"):
        ex.analyze_panel_table(
            kuznets, dv="gini_regional", idvs=["nope"], entity="country", time="year"
        )


def test_requires_linearmodels_message(monkeypatch):
    """The friendly install message fires when linearmodels cannot be imported."""
    import expdpy.panel_models as pm

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("linearmodels"):
            raise ImportError("simulated missing linearmodels")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match=r"expdpy\[panel\]"):
        pm._require_linearmodels()
