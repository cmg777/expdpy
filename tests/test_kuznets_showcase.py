"""The kuznets showcase: the N-shaped cubic survives country + year fixed effects.

`kuznets` is a country-year panel, so its lead illustrations control for two-way (country +
year) fixed effects. These tests guard that the synthetic data still identifies the cubic
Kuznets curve *within* country under that specification (so the README / docs claim of a
"positive, significant cubic term" stays true), and that the FWL plot is consistent with it.
"""

from __future__ import annotations

import pytest

from expdpy import prepare_fwl_plot, prepare_regression_table
from expdpy.data import load_kuznets

IDVS = ["log_gdp_pc", "log_gdp_pc_sq", "log_gdp_pc_cu"]


def test_cubic_survives_two_way_fixed_effects():
    """With country + year FE (clustered by country) the cubic term stays positive & sig."""
    df = load_kuznets()
    res = prepare_regression_table(
        df,
        dvs="gini_regional",
        idvs=IDVS,
        feffects=["country", "year"],
        clusters=["country"],
    )
    m = res.models[0]
    assert float(m.coef()["log_gdp_pc_cu"]) > 0  # positive cubic term -> the "N"
    assert (
        float(m.tidy().loc["log_gdp_pc_cu", "Pr(>|t|)"]) < 0.01
    )  # within-country sig.


def test_fwl_slope_matches_two_way_fe_coefficient():
    """FWL slope on the focal regressor equals its coefficient in the two-way FE model."""
    df = load_kuznets()
    tab = prepare_regression_table(
        df,
        dvs="gini_regional",
        idvs=IDVS,
        feffects=["country", "year"],
        clusters=["country"],
    ).models[0]
    fwl = prepare_fwl_plot(
        df,
        dv="gini_regional",
        var="log_gdp_pc",
        controls=["log_gdp_pc_sq", "log_gdp_pc_cu"],
        feffects=["country", "year"],
        clusters=["country"],
    )
    assert fwl.slope == pytest.approx(float(tab.coef()["log_gdp_pc"]), rel=1e-7)
    assert fwl.se == pytest.approx(float(tab.se()["log_gdp_pc"]), rel=1e-7)
