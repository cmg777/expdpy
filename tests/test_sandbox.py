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


def test_beta_convergence_conditional_recovers_truth():
    res = ex.learn_beta_convergence(rho=0.9, gamma=0.6, corr=0.7, seed=0)
    assert isinstance(res, SandboxResult)
    assert isinstance(res.fig, go.Figure)
    s = res.summary
    # Conditional recovers the true convergence slope; unconditional is biased away from it.
    assert abs(s["conditional_coef"] - s["true_beta"]) < 5e-3
    assert abs(s["unconditional_coef"] - s["true_beta"]) > abs(
        s["conditional_coef"] - s["true_beta"]
    )
    # Recovered speed of convergence matches the AR(1) truth (-ln rho).
    assert abs(s["conditional_speed"] - s["true_speed"]) < 1e-2
    assert res.topic == "beta_convergence"
    assert "convergence" in res.interpret().lower()
    assert res.explain().topic == "beta_convergence"


def test_beta_convergence_bias_grows_with_loading():
    # A larger loading on the omitted determinant biases the unconditional slope more.
    low = ex.learn_beta_convergence(gamma=0.2, seed=3).summary
    high = ex.learn_beta_convergence(gamma=1.0, seed=3).summary
    bias_low = abs(low["unconditional_coef"] - low["true_beta"])
    bias_high = abs(high["unconditional_coef"] - high["true_beta"])
    assert bias_high > bias_low


def test_sigma_convergence_recovers_true_dispersion_rate():
    res = ex.learn_sigma_convergence(rho=0.93, seed=0)
    assert isinstance(res, SandboxResult)
    assert isinstance(res.fig, go.Figure)
    s = res.summary
    # Each dispersion measure narrows at the known log-rate ln(rho) (exact, noiseless DGP).
    assert abs(s["true_slope"] - s["std_slope"]) < 1e-9
    assert abs(s["true_slope"] - s["gini_slope"]) < 1e-9
    assert abs(s["true_slope"] - s["cv_slope"]) < 1e-9
    assert s["std_slope"] < 0  # convergence
    assert res.topic == "sigma_convergence"
    assert "convergence" in res.interpret().lower()
    assert res.explain().topic == "sigma_convergence"


def test_sigma_convergence_rate_tracks_rho():
    # A smaller rho (faster contraction) gives a more negative true log-dispersion rate.
    fast = ex.learn_sigma_convergence(rho=0.80, seed=1).summary
    slow = ex.learn_sigma_convergence(rho=0.97, seed=1).summary
    assert fast["std_slope"] < slow["std_slope"] < 0
