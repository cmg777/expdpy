"""Interactive teaching sandboxes that *generate* data to make a concept tangible.

Unlike the ``explore_*`` / ``analyze_*`` functions (which summarize *your* data), the
``learn_*`` functions simulate data from a known data-generating process so a learner can see
a concept
in action and turn the knobs: omitted-variable bias, pooled-OLS-vs-fixed-effects, and the
effect of clustering on standard errors. Each returns a :class:`~expdpy.SandboxResult` whose
``summary`` holds the scalar facts the demonstration turns on (so they are easy to test and
to read back).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import statsmodels.api as sm

from expdpy._theme import apply_default_layout, color_for
from expdpy._types import SandboxResult
from expdpy.convergence import (
    analyze_beta_convergence,
    analyze_convergence_clubs,
    analyze_sigma_convergence,
)
from expdpy.kuznets import analyze_kuznets_waves
from expdpy.regression import analyze_regression_table

__all__ = [
    "learn_beta_convergence",
    "learn_clustering_se",
    "learn_convergence_clubs",
    "learn_first_differences",
    "learn_kuznets_waves",
    "learn_omitted_variable_bias",
    "learn_pooled_vs_fixed_effects",
    "learn_sigma_convergence",
    "learn_within_vs_lsdv",
]


def learn_omitted_variable_bias(
    *,
    n: int = 2000,
    beta_x: float = 1.0,
    beta_z: float = 1.0,
    corr_xz: float = 0.6,
    seed: int = 0,
) -> SandboxResult:
    """Show how omitting a correlated confounder biases a regression coefficient.

    Simulates ``y = beta_x * x + beta_z * z + noise`` where ``x`` and the confounder ``z``
    are correlated (``corr_xz``), then compares the *short* regression (omitting ``z``) with
    the *long* regression (controlling for ``z``).

    Parameters
    ----------
    n
        Sample size.
    beta_x
        True effect of the focal regressor ``x``.
    beta_z
        True effect of the omitted confounder ``z``.
    corr_xz
        Correlation between ``x`` and ``z`` (drives the bias).
    seed
        Random seed.

    Returns
    -------
    SandboxResult
        ``df`` (short vs long vs true coefficient), ``fig``, ``summary`` and ``topic``.
    """
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    x = corr_xz * z + math.sqrt(max(0.0, 1.0 - corr_xz**2)) * rng.normal(size=n)
    y = beta_x * x + beta_z * z + rng.normal(size=n)
    data = pd.DataFrame({"x": x, "z": z, "y": y})

    short = sm.OLS(y, sm.add_constant(data[["x"]])).fit()
    long = sm.OLS(y, sm.add_constant(data[["x", "z"]])).fit()
    short_b = float(short.params.iloc[1])
    long_b = float(long.params.iloc[1])

    df = pd.DataFrame(
        {
            "model": ["short (omits z)", "long (controls for z)", "true value"],
            "x_coefficient": [short_b, long_b, float(beta_x)],
        }
    )
    summary = {
        "true_beta_x": float(beta_x),
        "short_coef": short_b,
        "long_coef": long_b,
        "bias": short_b - float(beta_x),
        "corr_xz": float(corr_xz),
    }

    fig = go.Figure(
        go.Bar(
            x=df["model"],
            y=df["x_coefficient"],
            marker={"color": [color_for(0), color_for(2), color_for(9)]},
        )
    )
    fig.add_hline(
        y=float(beta_x),
        line_dash="dash",
        line_color="rgba(0,0,0,0.5)",
        annotation_text="true effect",
    )
    apply_default_layout(
        fig, xaxis={"title": ""}, yaxis={"title": "Estimated coefficient on x"}
    )
    fig.update_layout(title="Omitted-variable bias")
    return SandboxResult(df=df, fig=fig, summary=summary, topic="omitted_variable_bias")


def learn_pooled_vs_fixed_effects(
    *,
    n_units: int = 50,
    n_periods: int = 10,
    beta: float = 1.0,
    unit_effect_corr: float = 0.8,
    seed: int = 0,
) -> SandboxResult:
    """Show how pooled OLS is biased by unit effects, and fixed effects fix it.

    Simulates a panel where the regressor ``x`` is correlated with each unit's fixed effect.
    Pooled OLS confounds within- and between-unit variation (biased); the within (fixed-
    effects) estimator recovers the true slope.

    Parameters
    ----------
    n_units, n_periods
        Panel dimensions.
    beta
        True within-unit slope.
    unit_effect_corr
        Correlation between ``x`` and the unit effect (drives the pooled bias).
    seed
        Random seed.

    Returns
    -------
    SandboxResult
        ``df`` (pooled vs FE vs true slope), ``fig``, ``summary`` and ``topic``.
    """
    rng = np.random.default_rng(seed)
    alpha = rng.normal(size=n_units)  # unit effects
    rows = []
    for i in range(n_units):
        x = unit_effect_corr * alpha[i] + math.sqrt(
            max(0.0, 1.0 - unit_effect_corr**2)
        ) * rng.normal(size=n_periods)
        y = beta * x + alpha[i] + rng.normal(0.0, 0.5, size=n_periods)
        for t in range(n_periods):
            rows.append((i, t, float(x[t]), float(y[t])))
    data = pd.DataFrame(rows, columns=["unit", "period", "x", "y"])

    pooled = analyze_regression_table(data, dvs="y", idvs=["x"]).models[0]
    fe = analyze_regression_table(data, dvs="y", idvs=["x"], feffects=["unit"]).models[
        0
    ]
    pooled_b = float(pooled.coef()["x"])
    fe_b = float(fe.coef()["x"])

    df = pd.DataFrame(
        {
            "model": ["pooled OLS", "fixed effects", "true value"],
            "slope": [pooled_b, fe_b, float(beta)],
        }
    )
    summary = {
        "true_beta": float(beta),
        "pooled_coef": pooled_b,
        "fe_coef": fe_b,
        "unit_effect_corr": float(unit_effect_corr),
    }

    fig = go.Figure()
    shown = min(8, n_units)
    for j in range(shown):
        d = data[data["unit"] == j]
        fig.add_trace(
            go.Scatter(
                x=d["x"],
                y=d["y"],
                mode="markers",
                marker={"color": color_for(j), "size": 6, "opacity": 0.7},
                name=f"unit {j}",
                showlegend=False,
            )
        )
    xs = np.array([data["x"].min(), data["x"].max()])
    intercept = float(pooled.coef()["Intercept"])
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=intercept + pooled_b * xs,
            mode="lines",
            line={"color": "black", "width": 2},
            name="pooled OLS fit",
        )
    )
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.02,
        y=0.98,
        xanchor="left",
        yanchor="top",
        showarrow=False,
        align="left",
        bgcolor="rgba(255,255,255,0.7)",
        bordercolor="rgba(0,0,0,0.2)",
        borderwidth=1,
        text=(
            f"pooled slope = {pooled_b:.2f}<br>"
            f"FE slope = {fe_b:.2f}<br>true = {beta:.2f}"
        ),
    )
    apply_default_layout(fig, xaxis={"title": "x"}, yaxis={"title": "y"})
    fig.update_layout(title="Pooled OLS vs fixed effects")
    return SandboxResult(df=df, fig=fig, summary=summary, topic="fixed_effects")


def learn_clustering_se(
    *,
    n_clusters: int = 40,
    cluster_size: int = 30,
    icc: float = 0.3,
    seed: int = 0,
) -> SandboxResult:
    """Show that clustering changes the standard error, not the point estimate.

    Simulates data with cluster-correlated regressor and errors (intra-cluster correlation
    ``icc``), then compares classical (iid) standard errors with cluster-robust ones for the
    *same* coefficient.

    Parameters
    ----------
    n_clusters, cluster_size
        Number of clusters and observations per cluster.
    icc
        Intra-cluster correlation of the errors (drives the standard-error inflation).
    seed
        Random seed.

    Returns
    -------
    SandboxResult
        ``df`` (iid vs clustered standard error), ``fig``, ``summary`` and ``topic``.
    """
    rng = np.random.default_rng(seed)
    beta = 1.0
    sd_cluster = math.sqrt(max(0.0, icc))
    sd_idio = math.sqrt(max(0.0, 1.0 - icc))
    rows = []
    for c in range(n_clusters):
        u_c = rng.normal(0.0, sd_cluster)  # cluster error component
        x_c = rng.normal()  # cluster-level regressor shift
        for _ in range(cluster_size):
            x = x_c + rng.normal()
            y = beta * x + u_c + rng.normal(0.0, sd_idio)
            rows.append((c, x, y))
    data = pd.DataFrame(rows, columns=["cluster", "x", "y"])

    iid = analyze_regression_table(data, dvs="y", idvs=["x"]).models[0]
    clustered = analyze_regression_table(
        data, dvs="y", idvs=["x"], clusters=["cluster"]
    ).models[0]
    coef = float(iid.coef()["x"])
    iid_se = float(iid.se()["x"])
    clustered_se = float(clustered.se()["x"])

    df = pd.DataFrame(
        {
            "standard_error": ["iid", "clustered"],
            "coefficient": [coef, coef],
            "se": [iid_se, clustered_se],
            "ci_lower": [coef - 1.96 * iid_se, coef - 1.96 * clustered_se],
            "ci_upper": [coef + 1.96 * iid_se, coef + 1.96 * clustered_se],
        }
    )
    summary = {
        "coef": coef,
        "iid_se": iid_se,
        "clustered_se": clustered_se,
        "se_ratio": clustered_se / iid_se if iid_se else float("nan"),
        "icc": float(icc),
    }

    fig = go.Figure()
    for j, (label, se) in enumerate([("iid", iid_se), ("clustered", clustered_se)]):
        fig.add_trace(
            go.Scatter(
                x=[coef],
                y=[label],
                mode="markers",
                error_x={
                    "type": "data",
                    "array": [1.96 * se],
                    "thickness": 1.5,
                    "color": color_for(j),
                },
                marker={"color": color_for(j), "size": 11},
                showlegend=False,
            )
        )
    fig.add_vline(x=0, line_dash="dash", line_color="rgba(0,0,0,0.4)")
    apply_default_layout(
        fig,
        xaxis={"title": "Coefficient on x (95% interval)"},
        yaxis={"title": "", "type": "category"},
    )
    fig.update_layout(title="Clustering and standard errors")
    return SandboxResult(df=df, fig=fig, summary=summary, topic="clustered_se")


def _panel_with_unit_effects(
    rng: np.random.Generator,
    *,
    n_units: int,
    n_periods: int,
    beta: float,
    unit_effect_corr: float,
    noise_sd: float,
) -> pd.DataFrame:
    """Simulate a balanced panel ``y = beta*x + alpha_i + e`` with x correlated with alpha."""
    alpha = rng.normal(size=n_units)  # unit fixed effects
    rows = []
    for i in range(n_units):
        x = unit_effect_corr * alpha[i] + math.sqrt(
            max(0.0, 1.0 - unit_effect_corr**2)
        ) * rng.normal(size=n_periods)
        y = beta * x + alpha[i] + rng.normal(0.0, noise_sd, size=n_periods)
        for t in range(n_periods):
            rows.append((i, t, float(x[t]), float(y[t])))
    return pd.DataFrame(rows, columns=["unit", "period", "x", "y"])


def learn_first_differences(
    *,
    n_units: int = 150,
    n_periods: int = 2,
    beta: float = 2.0,
    unit_effect_corr: float = 0.8,
    noise_sd: float = 0.5,
    seed: int = 0,
) -> SandboxResult:
    """Show that first differencing removes the unit effect — matching the within estimator.

    Simulates a balanced panel ``y_it = beta * x_it + alpha_i + e_it`` where the regressor
    ``x`` is correlated with each unit's fixed effect ``alpha_i`` (so pooled OLS is biased).
    Differencing (``Δy`` on ``Δx``) cancels ``alpha_i``; on a two-period panel the first-
    differences estimate equals the within (demeaning) estimate, and both recover ``beta``.

    Parameters
    ----------
    n_units, n_periods
        Panel dimensions. With ``n_periods=2`` first differences and the within estimator
        coincide exactly; for more periods they differ slightly in finite samples.
    beta
        True within-unit slope.
    unit_effect_corr
        Correlation between ``x`` and the unit effect (drives the pooled bias).
    noise_sd
        Idiosyncratic noise standard deviation.
    seed
        Random seed.

    Returns
    -------
    SandboxResult
        ``df`` (pooled vs first differences vs within vs true slope), ``fig``, ``summary``
        and ``topic``.
    """
    rng = np.random.default_rng(seed)
    data = _panel_with_unit_effects(
        rng,
        n_units=n_units,
        n_periods=n_periods,
        beta=beta,
        unit_effect_corr=unit_effect_corr,
        noise_sd=noise_sd,
    )

    pooled_b = float(
        analyze_regression_table(data, dvs="y", idvs=["x"]).models[0].coef()["x"]
    )
    within_b = float(
        analyze_regression_table(data, dvs="y", idvs=["x"], feffects=["unit"])
        .models[0]
        .coef()["x"]
    )
    d = data.sort_values(["unit", "period"])
    d = d.assign(
        dx=d.groupby("unit")["x"].diff(), dy=d.groupby("unit")["y"].diff()
    ).dropna(subset=["dx", "dy"])
    fd_b = float(sm.OLS(d["dy"].to_numpy(), d[["dx"]].to_numpy()).fit().params[0])

    df = pd.DataFrame(
        {
            "method": [
                "pooled OLS",
                "first differences",
                "within (demeaning)",
                "true value",
            ],
            "slope": [pooled_b, fd_b, within_b, float(beta)],
        }
    )
    summary = {
        "true_beta": float(beta),
        "pooled_coef": pooled_b,
        "fd_coef": fd_b,
        "within_coef": within_b,
        "fd_within_gap": abs(fd_b - within_b),
        "n_periods": float(n_periods),
    }

    fig = go.Figure(
        go.Bar(
            x=df["method"],
            y=df["slope"],
            marker={"color": [color_for(9), color_for(0), color_for(2), color_for(4)]},
        )
    )
    fig.add_hline(
        y=float(beta),
        line_dash="dash",
        line_color="rgba(0,0,0,0.5)",
        annotation_text="true slope",
    )
    apply_default_layout(
        fig, xaxis={"title": ""}, yaxis={"title": "Estimated slope on x"}
    )
    fig.update_layout(title="First differences vs the within estimator")
    return SandboxResult(df=df, fig=fig, summary=summary, topic="first_differences")


def learn_within_vs_lsdv(
    *,
    n_units: int = 30,
    n_periods: int = 6,
    beta: float = 2.0,
    unit_effect_corr: float = 0.8,
    noise_sd: float = 0.5,
    seed: int = 0,
) -> SandboxResult:
    """Show that within (demeaning) and least-squares dummy variables give the same slope.

    Simulates a panel with a unit fixed effect and recovers the slope two ways: the within
    transformation (demeaning, via absorbed fixed effects) and least-squares dummy variables
    (one dummy per unit in OLS). By the Frisch-Waugh-Lovell theorem the two slopes are
    identical for any number of periods — demeaning and unit dummies do the same job.

    Parameters
    ----------
    n_units, n_periods
        Panel dimensions (kept modest so LSDV's one-dummy-per-unit design stays cheap).
    beta
        True within-unit slope.
    unit_effect_corr
        Correlation between ``x`` and the unit effect.
    noise_sd
        Idiosyncratic noise standard deviation.
    seed
        Random seed.

    Returns
    -------
    SandboxResult
        ``df`` (within vs LSDV vs true slope), ``fig``, ``summary`` and ``topic``.
    """
    rng = np.random.default_rng(seed)
    data = _panel_with_unit_effects(
        rng,
        n_units=n_units,
        n_periods=n_periods,
        beta=beta,
        unit_effect_corr=unit_effect_corr,
        noise_sd=noise_sd,
    )

    within_b = float(
        analyze_regression_table(data, dvs="y", idvs=["x"], feffects=["unit"])
        .models[0]
        .coef()["x"]
    )
    dummies = pd.get_dummies(data["unit"], prefix="u", drop_first=True).astype(float)
    exog = sm.add_constant(
        pd.concat(
            [data[["x"]].reset_index(drop=True), dummies.reset_index(drop=True)],
            axis=1,
        )
    )
    lsdv_b = float(sm.OLS(data["y"].to_numpy(), exog).fit().params["x"])

    df = pd.DataFrame(
        {
            "method": ["within (demeaning)", "LSDV (unit dummies)", "true value"],
            "slope": [within_b, lsdv_b, float(beta)],
        }
    )
    summary = {
        "true_beta": float(beta),
        "within_coef": within_b,
        "lsdv_coef": lsdv_b,
        "within_lsdv_gap": abs(within_b - lsdv_b),
        "n_periods": float(n_periods),
    }

    fig = go.Figure(
        go.Bar(
            x=df["method"],
            y=df["slope"],
            marker={"color": [color_for(2), color_for(0), color_for(4)]},
        )
    )
    fig.add_hline(
        y=float(beta),
        line_dash="dash",
        line_color="rgba(0,0,0,0.5)",
        annotation_text="true slope",
    )
    apply_default_layout(
        fig, xaxis={"title": ""}, yaxis={"title": "Estimated slope on x"}
    )
    fig.update_layout(title="Within transformation vs LSDV")
    return SandboxResult(df=df, fig=fig, summary=summary, topic="within_transformation")


def learn_beta_convergence(
    *,
    n_units: int = 60,
    n_years: int = 21,
    rho: float = 0.9,
    gamma: float = 0.6,
    corr: float = 0.7,
    noise: float = 0.05,
    seed: int = 0,
) -> SandboxResult:
    """Show unconditional vs conditional β-convergence on a known-parameter panel.

    Simulates an AR(1) panel in logs ``x_{t+1} = a + rho*x_t + gamma*z_i + e`` where ``z_i`` is
    a fixed steady-state determinant correlated (``corr``) with each unit's initial level. The
    AR(1) persistence pins the truth exactly: over a horizon ``T = n_years - 1`` the slope of
    growth on the initial level is ``beta = (rho**T - 1) / T`` and the structural speed of
    convergence is ``lambda = -ln(rho)`` (half-life ``ln 2 / lambda``). Because ``z_i`` is
    omitted, the **unconditional** regression is biased — units look like they barely converge
    (or even diverge); conditioning on ``z_i`` recovers the true convergence slope. This is the
    classic distinction between absolute and conditional convergence, demonstrated with
    :func:`expdpy.analyze_beta_convergence`.

    Parameters
    ----------
    n_units, n_years
        Panel dimensions (units and annual periods). The horizon is ``T = n_years - 1``.
    rho
        AR(1) persistence in ``(0, 1)``; it sets the true speed ``-ln(rho)`` (closer to 1 means
        slower convergence).
    gamma
        Loading of the steady-state determinant ``z`` (drives the omitted-variable bias of the
        unconditional estimate).
    corr
        Correlation between ``z`` and the initial level (also drives the bias).
    noise
        Idiosyncratic shock standard deviation.
    seed
        Random seed.

    Returns
    -------
    SandboxResult
        ``df`` (unconditional vs conditional vs true slope), ``fig``, ``summary`` and ``topic``.
    """
    rng = np.random.default_rng(seed)
    horizon = float(n_years - 1)
    a = (1.0 - rho) * 10.0  # steady-state level ~ 10
    z = rng.normal(size=n_units)  # fixed steady-state determinant
    x0 = 10.0 + 2.0 * (
        corr * z + math.sqrt(max(0.0, 1.0 - corr**2)) * rng.normal(size=n_units)
    )
    rows = []
    for i in range(n_units):
        x = float(x0[i])
        for t in range(n_years):
            rows.append((f"unit {i:02d}", t, x, float(z[i])))
            x = a + rho * x + gamma * float(z[i]) + rng.normal(0.0, noise)
    panel = pd.DataFrame(rows, columns=["unit", "year", "x", "z"])

    res = analyze_beta_convergence(
        panel, "x", controls=["z"], entity="unit", time="year", rolling=False
    )
    true_beta = (rho**horizon - 1.0) / horizon
    true_speed = -math.log(rho)
    true_half_life = math.log(2.0) / true_speed

    df = pd.DataFrame(
        {
            "model": [
                "unconditional (omits z)",
                "conditional (controls z)",
                "true value",
            ],
            "beta": [res.beta, res.beta_cond, true_beta],
        }
    )
    summary = {
        "true_beta": float(true_beta),
        "unconditional_coef": float(res.beta),
        "conditional_coef": float(res.beta_cond),
        "true_speed": float(true_speed),
        "conditional_speed": float(res.speed_cond),
        "true_half_life": float(true_half_life),
        "conditional_half_life": float(res.half_life_cond),
        "rho": float(rho),
        "horizon": float(horizon),
    }

    fig = go.Figure(
        go.Bar(
            x=df["model"],
            y=df["beta"],
            marker={"color": [color_for(9), color_for(2), color_for(0)]},
        )
    )
    fig.add_hline(
        y=float(true_beta),
        line_dash="dash",
        line_color="rgba(0,0,0,0.5)",
        annotation_text="true convergence slope",
    )
    apply_default_layout(
        fig,
        xaxis={"title": ""},
        yaxis={"title": "Convergence slope β (growth on initial level)"},
    )
    fig.update_layout(title="Unconditional vs conditional β-convergence")
    return SandboxResult(df=df, fig=fig, summary=summary, topic="beta_convergence")


def learn_sigma_convergence(
    *,
    n_units: int = 60,
    n_years: int = 21,
    rho: float = 0.93,
    noise: float = 0.0,
    seed: int = 0,
) -> SandboxResult:
    """Show σ-convergence on a panel whose dispersion narrows at a known rate.

    Simulates a panel in which every unit's value contracts geometrically toward a common mean
    ``mu``: ``x_{i,t} = mu + (x_{i,0} - mu) * rho**t``. Because the deviations from ``mu`` shrink
    by a constant factor ``rho`` each period while the mean stays fixed, **every** dispersion
    measure — the standard deviation, the Gini index and the coefficient of variation — scales
    as ``rho**t``. The trend of its log on time therefore equals ``ln(rho)`` exactly, the true
    speed of σ-convergence. Running :func:`expdpy.analyze_sigma_convergence` on the panel should
    recover that slope for all three measures.

    Parameters
    ----------
    n_units, n_years
        Panel dimensions (units and annual periods). The horizon is ``T = n_years - 1``.
    rho
        Per-period contraction factor in ``(0, 1)``; the true log-dispersion trend is
        ``ln(rho)`` (closer to 1 means slower convergence).
    noise
        Standard deviation of an optional additive shock (``0`` gives exact recovery).
    seed
        Random seed.

    Returns
    -------
    SandboxResult
        ``df`` (recovered trend per measure vs the true ``ln(rho)``), ``fig``, ``summary`` and
        ``topic``.
    """
    rng = np.random.default_rng(seed)
    horizon = float(n_years - 1)
    x0 = rng.uniform(1.0, 20.0, size=n_units)
    mu = float(np.mean(x0))
    rows = []
    for i in range(n_units):
        for t in range(n_years):
            val = mu + (float(x0[i]) - mu) * (rho**t)
            if noise:
                val += float(rng.normal(0.0, noise))
            rows.append((f"unit {i:02d}", t, val))
    panel = pd.DataFrame(rows, columns=["unit", "year", "x"])

    res = analyze_sigma_convergence(panel, "x", entity="unit", time="year")
    true_slope = math.log(rho)

    df = pd.DataFrame(
        {
            "measure": ["std", "gini", "cv", "true (ln rho)"],
            "trend": [res.std_slope, res.gini_slope, res.cv_slope, true_slope],
        }
    )
    summary = {
        "rho": float(rho),
        "horizon": float(horizon),
        "n_units": float(n_units),
        "true_slope": float(true_slope),
        "std_slope": float(res.std_slope),
        "gini_slope": float(res.gini_slope),
        "cv_slope": float(res.cv_slope),
    }

    fig = go.Figure(
        go.Bar(
            x=df["measure"],
            y=df["trend"],
            marker={"color": [color_for(0), color_for(1), color_for(2), color_for(9)]},
        )
    )
    fig.add_hline(
        y=float(true_slope),
        line_dash="dash",
        line_color="rgba(0,0,0,0.5)",
        annotation_text="true log-dispersion trend",
    )
    apply_default_layout(
        fig,
        xaxis={"title": ""},
        yaxis={"title": "Log-dispersion trend per period"},
    )
    fig.update_layout(title="σ-convergence: recovered vs true dispersion trend")
    return SandboxResult(df=df, fig=fig, summary=summary, topic="sigma_convergence")


def learn_kuznets_waves(
    *,
    n_units: int = 80,
    n_years: int = 15,
    betas: tuple[float, ...] = (0.5, -0.3, 0.05, 0.04),
    between_sd: float = 1.0,
    within_sd: float = 0.9,
    unit_effect_sd: float = 0.5,
    noise: float = 0.05,
    seed: int = 0,
) -> SandboxResult:
    """Show the three Kuznets-waves estimators recovering a *planted* polynomial wave.

    Simulates a panel ``y_it = sum_k betas[k-1] * g_it^k + a_i + d_t + e_it`` where the
    development index ``g_it = xbar_i + w_it`` mixes a cross-unit component (``between_sd``) and a
    within-unit component (``within_sd``), and the unit effect ``a_i`` and year effect ``d_t`` are
    drawn independently of ``g`` (so they bias none of the estimators). Because the planted wave
    is a within-unit relationship, the **within** (two-way fixed-effects) and **pooled** estimators
    recover the top-order coefficient, while the **between** estimator — comparing unit averages —
    differs, since averaging a nonlinear function is not the function of the average. Demonstrated
    with :func:`expdpy.analyze_kuznets_waves`.

    Parameters
    ----------
    n_units, n_years
        Panel dimensions.
    betas
        The planted polynomial coefficients ``(b_1, ..., b_degree)`` on ``g, g^2, ...``; its
        length sets the degree (default the quartic ``(0.5, -0.3, 0.05, 0.04)``).
    between_sd, within_sd
        Standard deviations of the cross-unit and within-unit components of ``g``.
    unit_effect_sd
        Standard deviation of the unit and year effects in the outcome (independent of ``g``).
    noise
        Idiosyncratic shock standard deviation.
    seed
        Random seed.

    Returns
    -------
    SandboxResult
        ``df`` (the top-order coefficient recovered by each estimator vs the true value),
        ``fig`` (the within partial-residual wave), ``summary`` and ``topic``.
    """
    rng = np.random.default_rng(seed)
    degree = len(betas)
    a = rng.normal(0.0, unit_effect_sd, size=n_units)
    d = rng.normal(0.0, unit_effect_sd, size=n_years)
    xbar = rng.normal(0.0, between_sd, size=n_units)
    rows = []
    for i in range(n_units):
        for t in range(n_years):
            gv = float(xbar[i] + rng.normal(0.0, within_sd))
            poly = sum(float(b) * gv ** (k + 1) for k, b in enumerate(betas))
            y = poly + float(a[i]) + float(d[t]) + float(rng.normal(0.0, noise))
            rows.append((f"unit {i:02d}", t, gv, y))
    panel = pd.DataFrame(rows, columns=["unit", "year", "g", "y"])

    res = analyze_kuznets_waves(
        panel, "y", "g", entity="unit", time="year", degree=degree
    )
    by = {str(row["estimator"]): row for _, row in res.summary.iterrows()}
    true_top = float(betas[-1])

    df = pd.DataFrame(
        {
            "estimator": ["pooled", "between", "within", "true"],
            "top_coefficient": [
                float(by["pooled"]["top_estimate"]),
                float(by["between"]["top_estimate"]),
                float(by["within"]["top_estimate"]),
                true_top,
            ],
        }
    )
    summary = {
        "degree": float(degree),
        "true_top": true_top,
        "pooled_top": float(by["pooled"]["top_estimate"]),
        "between_top": float(by["between"]["top_estimate"]),
        "within_top": float(by["within"]["top_estimate"]),
        "n_units": float(n_units),
        "n_years": float(n_years),
    }
    return SandboxResult(
        df=df, fig=res.fig_within, summary=summary, topic="kuznets_waves"
    )


def learn_convergence_clubs(
    *,
    n_per_club: int = 15,
    levels: tuple[float, ...] = (10.0, 9.3, 8.6),
    n_years: int = 35,
    rho: float = 0.9,
    spread: float = 0.4,
    noise: float = 0.002,
    seed: int = 0,
) -> SandboxResult:
    """Show Phillips-Sul club clustering recovering a *planted* club structure.

    Builds a panel with ``len(levels)`` known convergence clubs: every unit in club ``k`` starts
    at its long-run level ``levels[k]`` plus an idiosyncratic deviation that decays
    geometrically (``deviation * rho**t``), so units within a club converge to a common path
    while the distinct club levels keep the panel from converging globally. Running
    :func:`expdpy.analyze_convergence_clubs` on it should reject whole-panel convergence and
    recover the planted clubs. Demonstrates that the clustering is data-driven, not imposed.

    Parameters
    ----------
    n_per_club
        Units per planted club.
    levels
        The distinct long-run (log) levels, one per club; well-separated levels give clean clubs.
    n_years
        Number of annual periods (the horizon over which deviations decay).
    rho
        Per-period decay of the within-club deviations in ``(0, 1)`` (smaller converges faster).
    spread
        Half-width of the initial within-club deviation (uniform).
    noise
        Idiosyncratic shock standard deviation (kept tiny so the clubs stay sharp).
    seed
        Random seed.

    Returns
    -------
    SandboxResult
        ``df`` (each unit's planted vs detected club), ``fig`` (the recovered within-club average
        paths), ``summary`` and ``topic``.
    """
    rng = np.random.default_rng(seed)
    rows = []
    true_club: dict[str, int] = {}
    for k, mu in enumerate(levels, start=1):
        for j in range(n_per_club):
            uid = f"c{k}u{j:02d}"
            true_club[uid] = k
            dev = float(rng.uniform(-spread, spread))
            for t in range(1, n_years + 1):
                val = mu + dev * (rho ** (t - 1)) + float(rng.normal(0.0, noise))
                rows.append((uid, t, val))
    panel = pd.DataFrame(rows, columns=["unit", "year", "x"])

    res = analyze_convergence_clubs(panel, "x", entity="unit", time="year")

    detected = dict(zip(res.membership["entity"], res.membership["club"], strict=True))
    # Best-match accuracy: each detected club is scored by its modal planted club.
    by_detected: dict[int, list[int]] = {}
    for uid, det in detected.items():
        by_detected.setdefault(int(det), []).append(true_club[uid])
    correct = 0
    for det, trues in by_detected.items():
        if det == 0:  # the divergent group has no planted truth
            continue
        modal = max(set(trues), key=trues.count)
        correct += sum(1 for tc in trues if tc == modal)
    accuracy = correct / len(true_club)

    df = pd.DataFrame(
        {
            "unit": list(true_club),
            "true_club": [true_club[u] for u in true_club],
            "detected_club": [int(detected[u]) for u in true_club],
        }
    )
    summary = {
        "true_clubs": float(len(levels)),
        "detected_clubs": float(res.n_clubs),
        "n_units": float(len(true_club)),
        "accuracy": float(accuracy),
        "global_tstat": float(res.global_tstat),
    }
    return SandboxResult(df=df, fig=res.fig, summary=summary, topic="convergence_clubs")
