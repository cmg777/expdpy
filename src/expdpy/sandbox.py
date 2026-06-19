"""Interactive teaching sandboxes that *generate* data to make a concept tangible.

Unlike the ``prepare_*`` functions (which summarize *your* data), the ``sandbox_*``
functions simulate data from a known data-generating process so a learner can see a concept
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
from expdpy.regression import prepare_regression_table

__all__ = [
    "sandbox_clustering_se",
    "sandbox_omitted_variable_bias",
    "sandbox_pooled_vs_fixed_effects",
]


def sandbox_omitted_variable_bias(
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


def sandbox_pooled_vs_fixed_effects(
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

    pooled = prepare_regression_table(data, dvs="y", idvs=["x"]).models[0]
    fe = prepare_regression_table(data, dvs="y", idvs=["x"], feffects=["unit"]).models[
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


def sandbox_clustering_se(
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

    iid = prepare_regression_table(data, dvs="y", idvs=["x"]).models[0]
    clustered = prepare_regression_table(
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
