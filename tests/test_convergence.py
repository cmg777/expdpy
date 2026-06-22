"""Tests for :func:`expdpy.analyze_beta_convergence`.

The verification backbone is an **AR(1) in logs** whose convergence parameters are known in
closed form. With ``x_{t+1} = a + rho*x_t + eps`` on annual data, regressing the annualized
growth ``(x_T - x_0)/T`` on the initial level ``x_0`` has slope ``beta = (rho**T - 1)/T`` and
the structural speed of convergence is exactly ``lambda = -ln(rho)`` (independent of the
horizon), with half-life ``ln 2 / lambda``. A fixed steady-state determinant ``z_i`` that is
correlated with ``x_0`` biases the *unconditional* slope; conditioning on it (FWL) recovers the
truth. These let us assert recovery against hand-computed values.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

import expdpy as ex
from expdpy import (
    BetaConvergenceResult,
    ConvergenceClubsResult,
    SigmaConvergenceResult,
)
from expdpy.convergence import (
    _andrews_lrv,
    _gini,
    _half_life,
    _hp_trend,
    _log_t_test,
    _speed_of_convergence,
    _sround,
)

pytestmark = pytest.mark.panel


def _ar1_panel(
    *,
    n_units: int = 120,
    n_years: int = 21,
    rho: float | Sequence[float] = 0.9,
    gamma: float = 0.0,
    corr: float = 0.6,
    noise: float = 0.005,
    seed: int = 0,
) -> pd.DataFrame:
    """Annual AR(1) panel ``x_{t+1} = a + rho_t*x_t + gamma*z_i + eps`` in logs.

    ``rho`` may be a scalar or a per-step sequence of length ``n_years - 1`` (for time-varying
    persistence). ``z_i`` is a fixed unit trait correlated (``corr``) with the initial level.
    """
    rng = np.random.default_rng(seed)
    steps = (
        [float(rho)] * (n_years - 1) if np.isscalar(rho) else [float(r) for r in rho]  # type: ignore[union-attr]
    )
    assert len(steps) == n_years - 1
    z = rng.normal(size=n_units)
    x0 = 10.0 + 2.0 * (
        corr * z + math.sqrt(max(0.0, 1.0 - corr**2)) * rng.normal(size=n_units)
    )
    rows = []
    for i in range(n_units):
        x = float(x0[i])
        for t in range(n_years):
            rows.append((f"C{i:03d}", t, x, float(z[i])))
            if t < n_years - 1:
                r = steps[t]
                a = (1.0 - r) * 10.0  # steady-state level ~ 10
                x = a + r * x + gamma * float(z[i]) + rng.normal(0.0, noise)
    return pd.DataFrame(rows, columns=["country", "year", "x", "z"])


def _closed_form(rho: float, horizon: float) -> tuple[float, float, float]:
    beta = (rho**horizon - 1.0) / horizon
    speed = -math.log(rho)
    return beta, speed, math.log(2.0) / speed


# --- closed-form helpers ------------------------------------------------------
def test_speed_and_half_life_formulas():
    # AR(1) with rho -> slope b = (rho^T - 1)/T maps back to speed = -ln(rho).
    rho, T = 0.9, 20.0
    b = (rho**T - 1.0) / T
    assert _speed_of_convergence(b, T) == pytest.approx(-math.log(rho), abs=1e-12)
    assert _half_life(_speed_of_convergence(b, T)) == pytest.approx(
        math.log(2.0) / -math.log(rho), abs=1e-12
    )
    # Divergence (b >= 0) has no finite positive half-life.
    assert math.isnan(_half_life(_speed_of_convergence(0.01, 20.0)))
    # Undefined when 1 + b*T <= 0.
    assert math.isnan(_speed_of_convergence(-1.0, 20.0))


# --- unconditional recovery ---------------------------------------------------
def test_unconditional_recovers_ar1_parameters():
    rho = 0.9
    df = _ar1_panel(rho=rho, seed=1)
    res = ex.analyze_beta_convergence(df, "x", entity="country", time="year")
    b_true, sp_true, hl_true = _closed_form(rho, 20.0)
    assert isinstance(res, BetaConvergenceResult)
    assert res.horizon == 20.0
    assert res.n_obs == 120
    assert res.beta == pytest.approx(b_true, abs=2e-3)
    assert res.beta < 0  # convergence
    assert res.speed == pytest.approx(sp_true, abs=5e-3)
    assert res.half_life == pytest.approx(hl_true, abs=0.3)


def test_start_end_override_changes_horizon_and_slope():
    rho = 0.9
    df = _ar1_panel(rho=rho, seed=2)
    res = ex.analyze_beta_convergence(
        df, "x", entity="country", time="year", start=0, end=10
    )
    assert res.horizon == 10.0
    b_true, _, _ = _closed_form(rho, 10.0)
    assert res.beta == pytest.approx(b_true, abs=3e-3)


def test_vcov_changes_se_not_point_estimate():
    df = _ar1_panel(rho=0.9, seed=3)
    a = ex.analyze_beta_convergence(df, "x", entity="country", time="year", vcov="iid")
    b = ex.analyze_beta_convergence(
        df, "x", entity="country", time="year", vcov="hetero"
    )
    assert a.beta == pytest.approx(b.beta, abs=1e-12)  # same point estimate
    assert a.se != pytest.approx(b.se, abs=1e-12)  # different SEs


# --- conditional recovery (FWL) ----------------------------------------------
def test_conditional_recovers_true_slope_unconditional_biased():
    rho = 0.9
    df = _ar1_panel(rho=rho, gamma=0.6, corr=0.7, seed=4)
    res = ex.analyze_beta_convergence(
        df, "x", controls=["z"], entity="country", time="year"
    )
    b_true, _, _ = _closed_form(rho, 20.0)
    assert res.fig_conditional is not None
    assert len(res.models) == 2
    # Conditional recovers the truth; unconditional is biased further away.
    assert res.beta_cond == pytest.approx(b_true, abs=4e-3)
    assert abs(res.beta - b_true) > abs(res.beta_cond - b_true)
    # FWL identity: the conditional slope equals the full-model coefficient on the initial.
    assert res.beta_cond == pytest.approx(
        float(res.models[1].coef()["initial"]), abs=1e-9
    )


# --- rolling fixed-width windows ---------------------------------------------
def test_rolling_windows_match_closed_form():
    rho = 0.9
    df = _ar1_panel(rho=rho, seed=5)
    res = ex.analyze_beta_convergence(df, "x", entity="country", time="year", window=4)
    roll = res.rolling
    assert roll is not None and len(roll) == (21 - 1) - 4 + 1
    for _, r in roll.iterrows():
        w = r["horizon"]
        assert r["beta"] == pytest.approx((rho**w - 1.0) / w, abs=3e-3)
        assert r["speed"] == pytest.approx(-math.log(rho), abs=1e-2)
        assert r["ci_lower"] < r["beta"] < r["ci_upper"]


def test_rolling_tracks_time_varying_persistence():
    # Block A (fast convergence, rho=0.85) then block B (slow, rho=0.98).
    rho_a, rho_b, w = 0.85, 0.98, 4
    steps = [rho_a] * 12 + [rho_b] * 12  # 24 steps -> 25 years
    df = _ar1_panel(n_years=25, rho=steps, seed=6)
    res = ex.analyze_beta_convergence(df, "x", entity="country", time="year", window=w)
    roll = res.rolling.set_index("window_start")
    # A window fully inside block A recovers rho_a; one fully inside block B recovers rho_b.
    assert roll.loc[0.0, "speed"] == pytest.approx(-math.log(rho_a), abs=1.5e-2)
    assert roll.loc[16.0, "speed"] == pytest.approx(-math.log(rho_b), abs=1.5e-2)
    # Convergence was faster early than late.
    assert roll.loc[0.0, "speed"] > roll.loc[16.0, "speed"]


# --- result surface -----------------------------------------------------------
def test_result_surface_figs_hover_and_tables():
    df = _ar1_panel(rho=0.9, gamma=0.6, corr=0.7, seed=7)
    res = ex.analyze_beta_convergence(
        df, "x", controls=["z"], entity="country", time="year"
    )
    # Figures present.
    for fig in (res.fig, res.fig_conditional, res.fig_rolling):
        assert isinstance(fig, go.Figure)
    # Country names ride along on hover (on both the unconditional and conditional scatters).
    markers = next(t for t in res.fig.data if t.mode == "markers")
    assert markers.customdata is not None
    assert len(markers.customdata) == res.n_obs
    assert str(markers.customdata[0]).startswith("C")
    cond_markers = next(t for t in res.fig_conditional.data if t.mode == "markers")
    assert cond_markers.customdata is not None
    # Every scatter reports the core regression stats in its annotation box.
    uncond_ann = " ".join(a.text or "" for a in res.fig.layout.annotations)
    for token in ("β", "SE", "R²", "N"):
        assert token in uncond_ann
    # The (converging) conditional scatter also reports the speed and half-life.
    cond_ann = " ".join(a.text or "" for a in res.fig_conditional.layout.annotations)
    for token in ("β", "SE", "R²", "λ", "half-life"):
        assert token in cond_ann
    # Comparison table + tidy/glance.
    assert res.gt is not None
    assert list(res.summary["metric"]) == [
        "beta",
        "se",
        "r2",
        "n_obs",
        "speed",
        "half_life",
    ]
    assert "conditional" in res.summary.columns
    assert res.glance().shape[0] == 1
    assert set(res.df.columns) >= {"country", "initial", "final", "growth", "z__0"}


def test_interpret_is_associational_and_explain_topic():
    df = _ar1_panel(rho=0.9, seed=8)
    res = ex.analyze_beta_convergence(df, "x", entity="country", time="year")
    txt = res.interpret()
    assert "convergence" in txt.lower()
    # Interpretations describe associations, never causation.
    assert "causes" not in txt.lower()
    assert "effect of" not in txt.lower()
    assert "association" in txt.lower()  # the shared closing note
    assert res.explain().topic == "beta_convergence"


# --- edge cases ---------------------------------------------------------------
def test_too_few_periods_skips_rolling_with_note():
    df = _ar1_panel(n_years=2, rho=0.9, seed=9)  # 2 periods, window 5 impossible
    res = ex.analyze_beta_convergence(df, "x", entity="country", time="year", window=5)
    assert res.rolling is None
    assert res.fig_rolling is None
    assert any("rolling" in n for n in res.notes)


def test_missing_time_raises():
    df = _ar1_panel(rho=0.9, seed=10)
    with pytest.raises(ValueError):
        ex.analyze_beta_convergence(df, "x", entity="country")  # no time id


def test_non_numeric_var_raises():
    df = _ar1_panel(rho=0.9, seed=11)
    with pytest.raises(TypeError):
        ex.analyze_beta_convergence(df, "country", entity="country", time="year")


def test_zero_variance_initial_raises():
    # Every unit starts at the same initial level -> the convergence slope is not identified.
    df = _ar1_panel(rho=0.9, seed=13)
    t0 = df["year"].min()
    df.loc[df["year"] == t0, "x"] = 10.0
    with pytest.raises(ValueError, match="not identified"):
        ex.analyze_beta_convergence(df, "x", entity="country", time="year")


def test_unknown_control_raises():
    df = _ar1_panel(rho=0.9, seed=12)
    with pytest.raises(KeyError):
        ex.analyze_beta_convergence(
            df, "x", controls=["nope"], entity="country", time="year"
        )


# ============================ sigma convergence ==============================
# Verification backbone: a deterministic "geometric narrowing" panel in which every unit's
# value contracts toward a common mean mu at rate rho per period. Then mean_t = mu is constant
# and std_t = std_0*rho^t, Gini_t = Gini_0*rho^t, CV_t = CV_0*rho^t, so the OLS trend of the
# log dispersion on time equals ln(rho) EXACTLY for all three measures.


def _sigma_panel(
    *,
    n_units: int = 50,
    n_years: int = 11,
    rho: float = 0.9,
    noise: float = 0.0,
    lo: float = 1.0,
    hi: float = 20.0,
    seed: int = 0,
) -> pd.DataFrame:
    """Geometric-narrowing panel ``x_{i,t} = mu + (x_{i,0} - mu) * rho**t`` (``mu = mean x_0``)."""
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(lo, hi, size=n_units)
    mu = float(np.mean(x0))
    rows = []
    for i in range(n_units):
        for t in range(n_years):
            val = mu + (float(x0[i]) - mu) * (rho**t)
            if noise:
                val += float(rng.normal(0.0, noise))
            rows.append((f"U{i:03d}", t, val))
    return pd.DataFrame(rows, columns=["country", "year", "x"])


# --- Gini helper --------------------------------------------------------------
def test_gini_known_values_and_guards():
    # All equal -> 0; integers 1..n -> (n-1)/(3n); scale-invariant; guards return nan.
    assert _gini(np.array([5.0, 5.0, 5.0, 5.0])) == pytest.approx(0.0, abs=1e-12)
    n = 7
    assert _gini(np.arange(1.0, n + 1.0)) == pytest.approx((n - 1) / (3 * n), abs=1e-12)
    x = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0])
    assert _gini(x) == pytest.approx(_gini(2.0 * x), abs=1e-12)
    assert math.isnan(_gini(np.array([1.0])))  # fewer than two values
    assert math.isnan(_gini(np.array([-1.0, 2.0, 3.0])))  # negatives undefined


# --- mathematical validity: recovery of the known log-dispersion rate --------
def test_sigma_recovers_geometric_narrowing_rate():
    rho = 0.9
    df = _sigma_panel(rho=rho, n_units=60, n_years=15, seed=1)
    res = ex.analyze_sigma_convergence(df, "x", entity="country", time="year")
    target = math.log(rho)
    assert isinstance(res, SigmaConvergenceResult)
    assert res.std_slope == pytest.approx(target, abs=1e-9)
    assert res.gini_slope == pytest.approx(target, abs=1e-9)
    assert res.cv_slope == pytest.approx(target, abs=1e-9)
    assert res.std_slope < 0  # convergence
    assert res.std_r2 == pytest.approx(1.0, abs=1e-9)
    # A perfect fit gives SE ~ 0; the p-value is not asserted (it may be 0 or nan).
    assert bool(res.summary.set_index("measure").loc["std", "converging"])


def test_sigma_vcov_changes_se_not_slope():
    df = _sigma_panel(rho=0.9, noise=0.05, n_years=15, seed=2)
    a = ex.analyze_sigma_convergence(df, "x", entity="country", time="year", vcov="iid")
    b = ex.analyze_sigma_convergence(
        df, "x", entity="country", time="year", vcov="hetero"
    )
    assert a.std_slope == pytest.approx(b.std_slope, abs=1e-12)  # same point estimate
    assert a.std_se != pytest.approx(b.std_se, abs=1e-12)  # different SEs


def test_sigma_slope_orders_by_convergence_speed():
    # Faster contraction (smaller rho) => a more negative slope; ranking holds under noise.
    fast = ex.analyze_sigma_convergence(
        _sigma_panel(rho=0.80, noise=0.02, n_years=15, seed=3),
        "x",
        entity="country",
        time="year",
    )
    slow = ex.analyze_sigma_convergence(
        _sigma_panel(rho=0.97, noise=0.02, n_years=15, seed=3),
        "x",
        entity="country",
        time="year",
    )
    assert fast.std_slope < slow.std_slope < 0
    assert fast.std_slope == pytest.approx(math.log(0.80), abs=3e-2)
    assert slow.std_slope == pytest.approx(math.log(0.97), abs=3e-2)


# --- result surface -----------------------------------------------------------
def test_sigma_result_surface():
    df = _sigma_panel(rho=0.9, seed=4)
    res = ex.analyze_sigma_convergence(df, "x", entity="country", time="year")
    assert isinstance(res.fig, go.Figure)
    assert res.fig.layout.yaxis2 is not None  # dual axis (std left, Gini right)
    names = {t.name for t in res.fig.data}
    assert "Std. dev." in names and "Gini index" in names
    ann = " ".join(a.text or "" for a in res.fig.layout.annotations)
    assert "Std" in ann and "periods" in ann
    assert set(res.df.columns) == {"year", "n_units", "mean", "std", "gini", "cv"}
    assert len(res.df) == res.n_periods
    assert list(res.summary["measure"]) == ["std", "gini", "cv"]
    assert res.glance().shape[0] == 1
    assert res.gt is not None


def test_sigma_interpret_is_associational_and_explain_topic():
    df = _sigma_panel(rho=0.9, seed=5)
    res = ex.analyze_sigma_convergence(df, "x", entity="country", time="year")
    txt = res.interpret()
    assert "convergence" in txt.lower()
    assert "causes" not in txt.lower()
    assert "effect of" not in txt.lower()
    assert "association" in txt.lower()  # the shared closing note
    assert res.explain().topic == "sigma_convergence"


# --- edge cases ---------------------------------------------------------------
def test_sigma_unbalanced_panel_raises():
    df = _sigma_panel(rho=0.9, seed=6).iloc[5:]  # first unit loses its earliest periods
    with pytest.raises(ValueError, match="balanced"):
        ex.analyze_sigma_convergence(df, "x", entity="country", time="year")


def test_sigma_non_numeric_var_raises():
    df = _sigma_panel(rho=0.9, seed=7)
    with pytest.raises(TypeError):
        ex.analyze_sigma_convergence(df, "country", entity="country", time="year")


def test_sigma_missing_time_raises():
    df = _sigma_panel(rho=0.9, seed=8)
    with pytest.raises(ValueError):
        ex.analyze_sigma_convergence(df, "x", entity="country")  # no time id


def test_sigma_too_few_periods_raises():
    df = _sigma_panel(rho=0.9, n_years=2, seed=9)
    with pytest.raises(ValueError, match="periods"):
        ex.analyze_sigma_convergence(df, "x", entity="country", time="year")


def test_sigma_too_few_units_raises():
    df = _sigma_panel(rho=0.9, n_units=1, n_years=6, seed=10)
    with pytest.raises(ValueError, match="units"):
        ex.analyze_sigma_convergence(df, "x", entity="country", time="year")


def test_sigma_negative_values_degrade_gini_with_note():
    df = _sigma_panel(rho=0.9, seed=11)
    df.loc[df.index[:10], "x"] = (
        -df.loc[df.index[:10], "x"].abs() - 1.0
    )  # some negatives
    res = ex.analyze_sigma_convergence(df, "x", entity="country", time="year")
    assert math.isfinite(res.std_slope)  # std still computable
    assert math.isnan(res.gini_slope)  # Gini undefined on negatives
    assert any("Gini" in n for n in res.notes)


def test_sigma_cv_degrades_when_mean_crosses_zero():
    df = _sigma_panel(rho=0.9, seed=12)
    df["x"] = df["x"] - df.groupby("year")["x"].transform("mean")  # mean ~0 each period
    res = ex.analyze_sigma_convergence(df, "x", entity="country", time="year")
    assert math.isnan(res.cv_slope)
    assert any("coefficient of variation" in n for n in res.notes)


def test_sigma_cv_degrades_for_stably_negative_mean():
    # A uniformly-negative variable (no sign flip, not near zero) still has an undefined CV.
    df = _sigma_panel(rho=0.9, seed=13)
    df["x"] = -df["x"].abs() - 100.0
    res = ex.analyze_sigma_convergence(df, "x", entity="country", time="year")
    assert math.isnan(res.cv_slope)
    assert res.df["cv"].isna().all()  # no wrong-signed CV is surfaced
    assert any("coefficient of variation" in n for n in res.notes)
    assert math.isfinite(res.std_slope)  # the headline std trend is unaffected


# --- beta: duplicate-key data-quality guard ----------------------------------
def test_beta_duplicate_endpoint_rows_are_noted():
    # A duplicate (entity, time) row would otherwise make the slope depend on row order.
    df = _ar1_panel(rho=0.9, seed=14)
    t0 = df["year"].min()
    dup = df[(df["country"] == "C000") & (df["year"] == t0)].copy()
    dup["x"] = dup["x"] + 5.0
    dirty = pd.concat([dup, df], ignore_index=True)  # duplicate placed first
    res = ex.analyze_beta_convergence(dirty, "x", entity="country", time="year")
    assert any("duplicate" in n for n in res.notes)


def test_beta_nan_duplicate_does_not_evict_valid_row():
    # A NaN-valued duplicate ordered first must not evict the real observation (the dedup skips
    # NaN, like the sigma/clubs paths), so the unit is kept with its true value.
    df = _ar1_panel(rho=0.9, seed=15)
    t0 = df["year"].min()
    real = float(df[(df["country"] == "C000") & (df["year"] == t0)]["x"].iloc[0])
    nan_dup = df[(df["country"] == "C000") & (df["year"] == t0)].copy()
    nan_dup["x"] = np.nan
    dirty = pd.concat([nan_dup, df], ignore_index=True)  # NaN duplicate placed first
    res = ex.analyze_beta_convergence(dirty, "x", entity="country", time="year")
    assert res.n_obs == 120  # C000 is not silently dropped
    row = res.df[res.df["country"] == "C000"]
    assert float(row["initial"].iloc[0]) == pytest.approx(
        real
    )  # real value kept, not NaN


def test_beta_collinear_or_constant_control_is_skipped():
    # A control collinear with `initial` (or constant) is dropped by the estimator; the
    # conditional model must be skipped with a note, not reported as the unconditional slope.
    df = _ar1_panel(rho=0.9, seed=16)
    df["x_copy"] = df["x"]  # initial-year value is collinear with `initial`
    res = ex.analyze_beta_convergence(
        df, "x", controls=["x_copy"], entity="country", time="year"
    )
    assert res.fig_conditional is None
    assert math.isnan(res.beta_cond)
    assert "conditional" not in res.summary.columns
    assert any("collinear" in n for n in res.notes)

    df2 = _ar1_panel(rho=0.9, seed=17)
    df2["const"] = 5.0  # a constant control is collinear with the intercept
    res2 = ex.analyze_beta_convergence(
        df2, "x", controls=["const"], entity="country", time="year"
    )
    assert res2.fig_conditional is None and math.isnan(res2.beta_cond)
    assert any("collinear" in n or "constant" in n for n in res2.notes)


# ============================== club convergence ===============================
# Verification backbone: a panel with a *planted* club structure. Each unit in club k sits at a
# distinct long-run level levels[k] plus an idiosyncratic deviation that decays geometrically,
# so units within a club converge to a common path while the distinct levels keep the whole
# panel from converging. The Phillips-Sul algorithm should reject global convergence and recover
# the planted partition exactly.


def _club_panel(
    *,
    n_per_club: int = 12,
    levels: Sequence[float] = (10.0, 9.3, 8.6),
    n_years: int = 35,
    rho: float = 0.9,
    spread: float = 0.4,
    noise: float = 0.002,
    seed: int = 0,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Panel with known clubs; returns ``(df, {unit: true_club})``."""
    rng = np.random.default_rng(seed)
    rows: list[tuple[str, int, float]] = []
    truth: dict[str, int] = {}
    for k, mu in enumerate(levels, start=1):
        for j in range(n_per_club):
            uid = f"c{k}u{j:02d}"
            truth[uid] = k
            dev = float(rng.uniform(-spread, spread))
            for t in range(1, n_years + 1):
                rows.append(
                    (uid, t, mu + dev * rho ** (t - 1) + float(rng.normal(0, noise)))
                )
    return pd.DataFrame(rows, columns=["unit", "year", "x"]), truth


def _single_club_panel(
    *, n: int = 40, n_years: int = 35, rho: float = 0.9, seed: int = 0
) -> pd.DataFrame:
    """One converging group: all units approach a common level."""
    df, _ = _club_panel(
        n_per_club=n, levels=(10.0,), n_years=n_years, rho=rho, seed=seed
    )
    return df


def _recovery_accuracy(membership: pd.DataFrame, truth: dict[str, int]) -> float:
    """Best-match fraction of units placed in their planted club (label-invariant)."""
    detected = dict(zip(membership["entity"], membership["club"], strict=True))
    by_detected: dict[int, list[int]] = {}
    for uid, det in detected.items():
        by_detected.setdefault(int(det), []).append(truth[uid])
    correct = 0
    for det, trues in by_detected.items():
        if det == 0:
            continue
        modal = max(set(trues), key=trues.count)
        correct += sum(1 for tc in trues if tc == modal)
    return correct / len(truth)


# --- 1. Pure-helper unit tests -----------------------------------------------
def test_sround_rounds_half_away_from_zero():
    # Stata's round() rounds halves away from zero, unlike Python's banker's rounding.
    assert _sround(7.5) == 8
    assert _sround(2.5) == 3  # Python round(2.5) == 2
    assert _sround(-2.5) == -3
    assert _sround(2.4) == 2
    assert _sround(0.3 * 25) == 8  # round(7.5)


def test_andrews_lrv_recovers_white_noise_variance():
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 2.0, size=600)  # iid -> long-run variance == variance == 4
    assert _andrews_lrv(x) == pytest.approx(4.0, rel=0.2)
    assert math.isnan(_andrews_lrv(np.array([1.0, 2.0])))  # too short


def test_hp_trend_leaves_a_linear_series_unchanged():
    # The HP filter passes deterministic linear trends through untouched (a known fixed point).
    t = np.arange(40, dtype=float)
    mat = np.vstack([3.0 + 0.5 * t, -1.0 + 0.2 * t])
    trend = _hp_trend(mat, 400.0)
    assert np.allclose(trend, mat, atol=1e-6)


def test_log_t_test_matches_independent_ols_slope():
    rng = np.random.default_rng(3)
    mat = 10.0 + np.cumsum(rng.normal(0.0, 0.05, size=(6, 30)), axis=1)
    b, t = _log_t_test(mat, 0.3)
    big_t = mat.shape[1]
    xcm = mat.mean(0)
    h_var = np.mean((mat / xcm - 1.0) ** 2, axis=0)
    logt = np.log(np.arange(1, big_t + 1))
    with np.errstate(divide="ignore", invalid="ignore"):
        y = np.log(h_var[0] / h_var) - 2.0 * np.log(logt)
    start = _sround(0.3 * big_t)
    design = np.column_stack([logt[start:], np.ones(big_t - start)])
    b_ref = float(np.linalg.lstsq(design, y[start:], rcond=None)[0][0])
    assert b == pytest.approx(b_ref, abs=1e-9)
    assert math.isfinite(t)


def test_log_t_test_flags_convergence_and_divergence():
    # Vanishing deviations -> converges (t > -1.65); constant distinct levels -> diverges.
    conv = np.array(
        [
            [10.0 + (0.5 if i % 2 else -0.5) * 0.9**t for t in range(35)]
            for i in range(20)
        ]
    )
    assert _log_t_test(conv, 0.3)[1] > -1.65
    rng = np.random.default_rng(1)
    div = np.array([[8.0 if i < 10 else 12.0] * 35 for i in range(20)], dtype=float)
    div = div + rng.normal(0.0, 1e-3, div.shape)
    assert _log_t_test(div, 0.3)[1] <= -1.65


# --- 2. Mathematical-validity: recover the planted clubs ----------------------
def test_recovers_planted_clubs():
    df, truth = _club_panel(seed=1)
    res = ex.analyze_convergence_clubs(df, "x", entity="unit", time="year")
    assert isinstance(res, ConvergenceClubsResult)
    assert res.converged is False  # distinct levels => no global convergence
    assert res.global_tstat <= -1.65
    assert res.n_clubs == 3
    assert _recovery_accuracy(res.membership, truth) == pytest.approx(1.0)
    # every detected club passes the convergence test
    clubs = res.summary[res.summary["club"] != "Divergent"]
    assert bool(clubs["converging"].all())


def test_whole_panel_converges_to_single_club():
    res = ex.analyze_convergence_clubs(
        _single_club_panel(seed=2), "x", entity="unit", time="year"
    )
    assert res.converged is True
    assert res.global_tstat > -1.65
    assert res.n_clubs == 1
    assert res.n_divergent == 0
    assert (res.membership["club"] == 1).all()


def test_merging_does_not_increase_club_count():
    df, _ = _club_panel(seed=4)
    counts = {
        mode: ex.analyze_convergence_clubs(
            df, "x", entity="unit", time="year", merge=mode
        ).n_clubs
        for mode in ("none", "single", "iterative")
    }
    assert counts["iterative"] <= counts["single"] <= counts["none"]


def test_ps_method_variant_runs():
    df, truth = _club_panel(seed=5)
    res = ex.analyze_convergence_clubs(df, "x", entity="unit", time="year", method="ps")
    assert res.n_clubs >= 1
    assert _recovery_accuracy(res.membership, truth) == pytest.approx(1.0)


def test_fr_sort_window_matches_reference():
    # fr>0 sorts units by the mean of the last (1-fr) periods. The window start is
    # trunc((1-fr)*n_periods) (matching the Stata reference), not one period earlier.
    from expdpy.convergence import _sort_order

    mat = np.arange(40, dtype=float).reshape(4, 10)  # 4 units x 10 periods
    expected = mat[:, 7:].mean(axis=1)  # fr=0.3 -> last trunc(0.7*10)=7 -> periods 7..9
    assert list(_sort_order(mat, 0.3)) == list(np.argsort(-expected, kind="stable"))
    # and the full pipeline still recovers the planted clubs under fr>0.
    df, truth = _club_panel(seed=6)
    res = ex.analyze_convergence_clubs(df, "x", entity="unit", time="year", fr=0.3)
    assert _recovery_accuracy(res.membership, truth) == pytest.approx(1.0)


# --- 3. Result-surface / expected-use ----------------------------------------
def test_result_surface_figs_tables_and_interpret():
    df, _ = _club_panel(seed=6)
    res = ex.analyze_convergence_clubs(df, "x", entity="unit", time="year")
    for fig in (res.fig, res.fig_paths, res.fig_clubs):
        assert isinstance(fig, go.Figure)
    assert set(res.df.columns) == {"unit", "year", "value", "relative", "club"}
    assert res.gt is not None
    assert list(res.summary.columns) == [
        "club",
        "n_members",
        "beta",
        "tstat",
        "converging",
        "members",
    ]
    assert set(res.membership.columns) == {"entity", "club", "club_label"}
    assert res.glance().shape[0] == 1
    assert res.tidy() is res.summary
    txt = res.interpret()
    assert "club" in txt.lower()
    assert "causes" not in txt.lower() and "effect of" not in txt.lower()
    assert "association" in txt.lower()  # the shared closing note
    assert res.explain().topic == "convergence_clubs"


def test_filter_none_uses_variable_as_supplied():
    df, truth = _club_panel(seed=7)
    res = ex.analyze_convergence_clubs(df, "x", entity="unit", time="year", filter=None)
    assert math.isnan(res.hp_lambda)  # no HP filter applied
    assert _recovery_accuracy(res.membership, truth) == pytest.approx(1.0)


# --- 4. Edge cases -----------------------------------------------------------
def test_clubs_missing_time_raises():
    df, _ = _club_panel(seed=8)
    with pytest.raises(ValueError):
        ex.analyze_convergence_clubs(df, "x", entity="unit")  # no time id


def test_clubs_non_numeric_var_raises():
    df, _ = _club_panel(seed=9)
    with pytest.raises(TypeError):
        ex.analyze_convergence_clubs(df, "unit", entity="unit", time="year")


def test_unbalanced_panel_raises():
    df, _ = _club_panel(seed=10)
    dropped = df.drop(df.index[5])  # remove one (unit, year) cell -> a gap
    with pytest.raises(ValueError, match="not balanced"):
        ex.analyze_convergence_clubs(dropped, "x", entity="unit", time="year")


def test_too_few_periods_raises():
    df, _ = _club_panel(n_years=5, seed=11)  # 5 - round(0.3*5)=2 -> 3 < 4
    with pytest.raises(ValueError, match="too few periods"):
        ex.analyze_convergence_clubs(df, "x", entity="unit", time="year")


def test_bad_trim_fraction_raises():
    df, _ = _club_panel(seed=12)
    with pytest.raises(ValueError, match="trimming fraction"):
        ex.analyze_convergence_clubs(df, "x", entity="unit", time="year", r=1.5)


# --- 5. Real-data smoke ------------------------------------------------------
def test_real_data_productivity_smoke():
    from expdpy.data import load_productivity

    df = load_productivity()
    res = ex.analyze_convergence_clubs(df, "log_gdppc", entity="country", time="year")
    assert isinstance(res, ConvergenceClubsResult)
    assert res.n_units == 108 and res.n_periods == 25
    assert res.n_clubs >= 1
    assert isinstance(res.fig, go.Figure)
    # every reported club clears the convergence threshold
    clubs = res.summary[res.summary["club"] != "Divergent"]
    assert bool((clubs["tstat"] > -1.65).all())
