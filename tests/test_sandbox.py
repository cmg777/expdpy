"""Tests for the concept sandboxes.

Each demonstration has a deterministic ``summary`` whose key fact we assert (the bias has
the predicted sign, the within estimator recovers the truth, clustering inflates the SE).
"""

from __future__ import annotations

import plotly.graph_objects as go

import expdpy as ex
from expdpy import SandboxResult


def test_ovb_short_is_biased_long_is_not():
    res = ex.learn_omitted_variable_bias(beta_x=1.0, beta_z=1.0, corr_xz=0.6, seed=0)
    assert isinstance(res, SandboxResult)
    assert isinstance(res.fig, go.Figure)
    s = res.summary
    # positive confounder effect * positive correlation -> upward bias
    assert s["short_coef"] > s["true_beta_x"]
    assert abs(s["long_coef"] - s["true_beta_x"]) < 0.1
    assert res.topic == "omitted_variable_bias"
    assert "bias" in res.interpret().lower()
    assert res.explain().topic == "omitted_variable_bias"


def test_ovb_bias_grows_with_correlation():
    low = ex.learn_omitted_variable_bias(corr_xz=0.2, seed=1).summary["bias"]
    high = ex.learn_omitted_variable_bias(corr_xz=0.8, seed=1).summary["bias"]
    assert abs(high) > abs(low)


def test_pooled_vs_fe_recovers_truth():
    res = ex.learn_pooled_vs_fixed_effects(beta=1.0, unit_effect_corr=0.8, seed=0)
    s = res.summary
    assert s["pooled_coef"] > s["fe_coef"]  # pooled biased upward
    assert abs(s["fe_coef"] - s["true_beta"]) < 0.15  # FE recovers the truth
    assert res.topic == "fixed_effects"
    assert res.explain().topic == "fixed_effects"


def test_clustering_inflates_se_not_estimate():
    res = ex.learn_clustering_se(n_clusters=40, cluster_size=30, icc=0.4, seed=0)
    s = res.summary
    assert s["clustered_se"] >= s["iid_se"]  # clustering widens the SE
    assert s["se_ratio"] > 1.0
    assert res.topic == "clustered_se"
    # the point estimate is identical under both SE choices
    assert res.df["coefficient"].nunique() == 1
    assert "standard error" in res.interpret().lower()


def test_clustering_se_grows_with_icc():
    low = ex.learn_clustering_se(icc=0.1, seed=2).summary["se_ratio"]
    high = ex.learn_clustering_se(icc=0.5, seed=2).summary["se_ratio"]
    assert high > low


def test_first_differences_matches_within_at_two_periods():
    res = ex.learn_first_differences(n_units=150, n_periods=2, beta=2.0, seed=0)
    assert isinstance(res, SandboxResult)
    assert isinstance(res.fig, go.Figure)
    s = res.summary
    assert s["fd_within_gap"] < 1e-6  # FD == within at T=2
    assert abs(s["fd_coef"] - s["true_beta"]) < 0.25  # both recover beta
    assert abs(s["pooled_coef"] - s["true_beta"]) > abs(s["fd_coef"] - s["true_beta"])
    assert res.topic == "first_differences"
    assert "differenc" in res.interpret().lower()
    assert res.explain().topic == "first_differences"


def test_within_equals_lsdv_for_many_periods():
    res = ex.learn_within_vs_lsdv(n_units=30, n_periods=6, beta=2.0, seed=1)
    s = res.summary
    assert s["within_lsdv_gap"] < 1e-6  # within == LSDV for any T
    assert abs(s["within_coef"] - s["true_beta"]) < 0.25
    assert res.topic == "within_transformation"
    assert res.explain().topic == "within_transformation"
    assert "lsdv" in res.interpret().lower() or "dummy" in res.interpret().lower()
