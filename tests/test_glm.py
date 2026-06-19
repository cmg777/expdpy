"""Tests for the nonlinear fixed-effects models (Poisson, logit, probit).

These exercise the GLM paths of prepare_estimation: coefficient recovery, the probit family,
and the separation / convergence advisory notes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import expdpy as ex


@pytest.fixture(scope="module")
def glm_df() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    n = 1500
    x = rng.normal(size=n)
    firm = rng.integers(0, 25, n)
    count = rng.poisson(np.exp(0.5 + 0.4 * x))  # true Poisson slope == 0.4
    latent = 0.8 * x + rng.normal(size=n)
    return pd.DataFrame(
        {"x": x, "firm": firm, "count": count, "bin": (latent > 0).astype(int)}
    )


def test_poisson_recovers_slope_and_pseudo_r2(glm_df):
    res = ex.prepare_estimation(glm_df, dv="count", idvs=["x"], model="poisson")
    est = float(res.df.query("term == 'x'")["Estimate"].iloc[0])
    assert est == pytest.approx(0.4, abs=0.1)
    assert np.isfinite(res.fit_stats["pseudo_r2"].iloc[0])  # Poisson reports pseudo-R²


def test_probit_runs_and_has_positive_slope(glm_df):
    res = ex.prepare_estimation(glm_df, dv="bin", idvs=["x"], model="probit")
    assert res.model_kind == "probit"
    assert float(res.df.query("term == 'x'")["Estimate"].iloc[0]) > 0
    assert "latent index" in res.interpret()


def test_logit_with_fe(glm_df):
    res = ex.prepare_estimation(
        glm_df, dv="bin", idvs=["x"], model="logit", feffects=["firm"]
    )
    assert "pseudo_r2" in res.fit_stats.columns  # GLM fit-stats schema
    assert int(res.fit_stats["N"].iloc[0]) > 0
    assert bool(res.fit_stats["has_fe"].iloc[0])


def test_separation_note(glm_df):
    # A Poisson fixed-effect level with all-zero counts is perfectly separated and dropped.
    extra = pd.DataFrame({"x": [0.1] * 30, "firm": 999, "count": 0, "bin": 0})
    df = pd.concat([glm_df, extra], ignore_index=True)
    res = ex.prepare_estimation(
        df, dv="count", idvs=["x"], model="poisson", feffects=["firm"]
    )
    assert any("separation" in n for n in res.notes)


def test_regression_table_still_ols_only(glm_df):
    # prepare_regression_table stays OLS-only; binary/count outcomes go through
    # prepare_estimation instead.
    str_df = glm_df.assign(label=glm_df["bin"].map({0: "no", 1: "yes"}))
    with pytest.raises(NotImplementedError):
        ex.prepare_regression_table(str_df, dvs="label", idvs=["x"])
