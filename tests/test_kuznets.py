"""Tests for :func:`expdpy.analyze_kuznets_waves` (the extended Kuznets curve).

The verification backbone is a **planted polynomial** panel whose true coefficients are known.
Because the wave is a within-unit relationship, the **within** (two-way fixed-effects) and
**pooled** estimators recover the planted ``b_1 .. b_degree`` (entity/year effects are drawn
independently of the development variable), while a separate cross-sectional DGP pins the
**between** estimator. The pure ``_turning_points`` helper is checked against a closed-form
cubic whose extrema are known exactly. R-parity lives in ``test_kuznets_vs_r.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

import expdpy as ex
from expdpy import KuznetsWavesResult
from expdpy.kuznets import _eval_poly, _poly_terms, _turning_points

pytestmark = pytest.mark.panel

_BETAS = (0.5, -0.3, 0.05, 0.04)  # planted (b1, b2, b3, b4)


def _wave_panel(
    betas: tuple[float, ...] = _BETAS,
    *,
    n_units: int = 80,
    n_years: int = 15,
    between_sd: float = 1.0,
    within_sd: float = 1.0,
    eff_sd: float = 0.4,
    noise: float = 0.03,
    seed: int = 0,
) -> pd.DataFrame:
    """Panel ``y = sum_k b_k g^k + a_i + d_t + e`` with ``g`` varying within and between units.

    The unit effect ``a_i`` and year effect ``d_t`` are independent of ``g``, so the planted
    within-unit wave is recovered by the within (two-way FE) and pooled estimators.
    """
    rng = np.random.default_rng(seed)
    a = rng.normal(0.0, eff_sd, n_units)
    d = rng.normal(0.0, eff_sd, n_years)
    xbar = rng.normal(0.0, between_sd, n_units)
    rows = []
    for i in range(n_units):
        for t in range(n_years):
            g = float(xbar[i] + rng.normal(0.0, within_sd))
            y = (
                sum(float(b) * g ** (k + 1) for k, b in enumerate(betas))
                + float(a[i])
                + float(d[t])
                + float(rng.normal(0.0, noise))
            )
            rows.append((f"u{i:03d}", t, g, y))
    return pd.DataFrame(rows, columns=["unit", "year", "g", "y"])


def _between_wave_panel(
    betas: tuple[float, ...] = _BETAS,
    *,
    n_units: int = 90,
    n_years: int = 10,
    spread: float = 1.6,
    noise: float = 0.01,
    seed: int = 0,
) -> pd.DataFrame:
    """Cross-sectional DGP: ``y_i = sum_k b_k xbar_i^k`` with negligible within variation.

    The development index is essentially constant within a unit, so each unit's mean development
    equals ``xbar_i`` and the between estimator (a polynomial in the entity means) recovers the
    planted coefficients.
    """
    rng = np.random.default_rng(seed)
    xbar = rng.uniform(-spread, spread, n_units)
    rows = []
    for i in range(n_units):
        base = sum(float(b) * xbar[i] ** (k + 1) for k, b in enumerate(betas))
        for t in range(n_years):
            g = float(xbar[i] + rng.normal(0.0, 0.01))
            y = base + float(rng.normal(0.0, noise))
            rows.append((f"u{i:03d}", t, g, y))
    return pd.DataFrame(rows, columns=["unit", "year", "g", "y"])


def _betas_of(model, degree: int = 4) -> np.ndarray:
    """Pull the polynomial coefficients ``b_1 .. b_degree`` from a fitted pyfixest model."""
    coef = model.coef()
    return np.array([float(coef[t]) for t in _poly_terms("g", degree)])


# --- 1. Pure-helper / closed-form unit tests ---------------------------------------------
def test_turning_points_known_cubic():
    # f'(g) = (g-2)(g-4) = g^2 - 6g + 8  ->  f(g) = g^3/3 - 3 g^2 + 8 g
    tps = _turning_points([8.0, -3.0, 1.0 / 3.0], 0.0, 10.0)
    assert [tp["g"] for tp in tps] == pytest.approx([2.0, 4.0], abs=1e-9)
    assert [tp["kind"] for tp in tps] == ["peak", "trough"]


def test_turning_points_domain_guards():
    assert _turning_points([1.0], 0.0, 10.0) == []  # linear: no turning point
    assert _turning_points([0.0, 0.0, 0.0, 0.0], -5.0, 5.0) == []  # all-zero polynomial
    assert (
        _turning_points([8.0, -3.0, 1.0 / 3.0], 5.0, 10.0) == []
    )  # extrema out of range


def test_poly_helpers():
    assert _poly_terms("g", 4) == ["g", "g_p2", "g_p3", "g_p4"]
    assert _poly_terms("log_gdp_pc", 2) == ["log_gdp_pc", "log_gdp_pc_p2"]
    got = _eval_poly([2.0, 3.0], np.array([1.0, 2.0]))  # 2 g + 3 g^2
    assert got == pytest.approx([5.0, 16.0])


# --- 2. Mathematical-validity: recovery of the planted polynomial ------------------------
def test_within_recovers_planted_wave():
    df = _wave_panel(seed=1)
    res = ex.analyze_kuznets_waves(df, "y", "g", entity="unit", time="year", degree=4)
    within = _betas_of(res.models["within"][-1])
    np.testing.assert_allclose(within, _BETAS, atol=0.03)


def test_pooled_recovers_planted_wave():
    df = _wave_panel(seed=2)
    res = ex.analyze_kuznets_waves(df, "y", "g", entity="unit", time="year", degree=4)
    pooled = _betas_of(res.models["pooled"][-1])
    np.testing.assert_allclose(pooled, _BETAS, atol=0.05)


def test_between_recovers_cross_sectional_wave():
    df = _between_wave_panel(seed=3)
    res = ex.analyze_kuznets_waves(df, "y", "g", entity="unit", time="year", degree=4)
    between = _betas_of(res.models["between"][-1])
    np.testing.assert_allclose(between, _BETAS, atol=0.02)


def test_degree_two_recovers_inverted_u_peak():
    # A pure quadratic y = 2 + 4 g - 0.5 g^2 peaks at g = 4 (cross-sectional).
    rng = np.random.default_rng(7)
    xbar = rng.uniform(0.0, 8.0, 100)
    rows = []
    for i in range(100):
        base = 2.0 + 4.0 * xbar[i] - 0.5 * xbar[i] ** 2
        for t in range(8):
            rows.append((f"u{i:03d}", t, float(xbar[i] + rng.normal(0, 0.01)), base))
    df = pd.DataFrame(rows, columns=["unit", "year", "g", "y"])
    res = ex.analyze_kuznets_waves(df, "y", "g", entity="unit", time="year", degree=2)
    row = res.summary.set_index("estimator").loc["between"]
    assert int(row["n_turning_points"]) == 1
    assert float(row["peak_g"]) == pytest.approx(4.0, abs=0.1)


# --- 3. Result-surface / expected-use ----------------------------------------------------
def test_result_surface_figures_tables_and_frames():
    df = _wave_panel(seed=4)
    res = ex.analyze_kuznets_waves(
        df, "y", "g", controls=None, entity="unit", time="year", degree=4
    )
    assert isinstance(res, KuznetsWavesResult)
    for fig in (res.fig, res.fig_between, res.fig_within):
        assert isinstance(fig, go.Figure)
        markers = next(t for t in fig.data if t.mode == "markers")
        assert markers.customdata is not None and len(markers.customdata) > 0
    for gt in (res.gt_pooled, res.gt_between, res.gt_within):
        assert gt is not None
    assert list(res.summary["estimator"]) == ["pooled", "between", "within"]
    assert set(res.models) == {"pooled", "between", "within"}
    assert all(len(v) == res.degree for v in res.models.values())
    # tidy stacks all three estimators across the nested specifications
    assert set(res.df["estimator"]) == {"pooled", "between", "within"}
    assert sorted(res.df["spec"].unique()) == [1, 2, 3, 4]
    assert res.tidy() is res.df
    assert res.glance() is res.summary


def test_controls_partial_out_and_annotation_present():
    df = _wave_panel(seed=5)
    df["z"] = np.random.default_rng(0).normal(size=len(df))  # an inert covariate
    res = ex.analyze_kuznets_waves(
        df, "y", "g", controls=["z"], entity="unit", time="year", degree=4
    )
    assert res.controls == ("z",)
    # the within figure annotation reports curvature, N and the within R²
    ann = " ".join(a.text or "" for a in res.fig_within.layout.annotations)
    assert "N =" in ann and "within R2" in ann


def test_interpret_is_associational_and_explain_topic():
    df = _wave_panel(seed=6)
    res = ex.analyze_kuznets_waves(df, "y", "g", entity="unit", time="year")
    txt = res.interpret()
    assert "causes" not in txt.lower()
    assert "effect of" not in txt.lower()
    assert "association" in txt.lower()  # the shared closing note
    assert "Kuznets" in txt or "kuznets" in txt.lower()
    assert res.explain().topic == "kuznets_waves"


def test_set_panel_supplies_ids():
    df = ex.set_panel(_wave_panel(seed=9), entity="unit", time="year")
    res = ex.analyze_kuznets_waves(df, "y", "g")  # ids resolved from attrs
    assert res.entity == "unit" and res.time == "year" and res.n_obs == len(df)


# --- 4. Edge cases: assert the right exception type/message ------------------------------
def test_missing_time_raises():
    df = _wave_panel(seed=10)
    with pytest.raises(ValueError, match="time id is required"):
        ex.analyze_kuznets_waves(df, "y", "g", entity="unit")


def test_non_numeric_outcome_raises():
    df = _wave_panel(seed=11)
    with pytest.raises(TypeError):
        ex.analyze_kuznets_waves(df, "unit", "g", entity="unit", time="year")


def test_development_as_control_raises():
    df = _wave_panel(seed=12)
    with pytest.raises(ValueError, match="must not also be a control"):
        ex.analyze_kuznets_waves(
            df, "y", "g", controls=["g"], entity="unit", time="year"
        )


@pytest.mark.parametrize("bad_degree", [1, 7])
def test_degree_out_of_range_raises(bad_degree):
    df = _wave_panel(seed=13)
    with pytest.raises(ValueError, match="degree must be in"):
        ex.analyze_kuznets_waves(
            df, "y", "g", entity="unit", time="year", degree=bad_degree
        )


def test_zero_variance_development_raises():
    df = _wave_panel(seed=14)
    df["g"] = 3.0  # kill all variation in the regressor
    with pytest.raises(ValueError, match="not identified"):
        ex.analyze_kuznets_waves(df, "y", "g", entity="unit", time="year")


def test_too_few_entities_for_between_raises():
    # 3 units x 20 years: enough rows for a quartic, but too few entities for the between fit.
    rng = np.random.default_rng(15)
    rows = [
        (f"u{i}", t, float(rng.normal()), float(rng.normal()))
        for i in range(3)
        for t in range(20)
    ]
    df = pd.DataFrame(rows, columns=["unit", "year", "g", "y"])
    with pytest.raises(ValueError, match="too few entities"):
        ex.analyze_kuznets_waves(df, "y", "g", entity="unit", time="year", degree=4)
