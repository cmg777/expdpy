"""Frisch-Waugh-Lovell plot: residualized scatter of a focal regressor against the outcome.

Ported from the R package ``fwlplot`` (Kyle Butts). Both the outcome and the focal
regressor are residualized on the *other* regressors **and the fixed effects** (the latter
absorbed/demeaned via ``pyfixest``), and the two residual vectors are plotted against each
other. By the Frisch-Waugh-Lovell theorem the slope of that residual scatter equals the
focal coefficient in the full multivariate model — which is exactly what
:func:`expdpy.analyze_regression_table` reports.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pyfixest as pf
import statsmodels.api as sm
from pandas.api import types as pdt

from expdpy._labels import resolve_label
from expdpy._theme import apply_default_layout, color_for
from expdpy._types import FWLPlotResult
from expdpy._validation import ensure_dataframe
from expdpy.regression import _SSC, _as_list

__all__ = ["analyze_fwl_plot"]


def _residualize(data: pd.DataFrame, target: str, rhs: str, fe_part: str) -> np.ndarray:
    """Residuals of ``target`` on ``rhs`` (+ fixed effects ``fe_part``) via pyfixest.

    With fixed effects the residuals are demeaned within each FE group, so the returned
    values are conditional on both the controls and the fixed effects.
    """
    model = pf.feols(f"{target} ~ {rhs}{fe_part}", data=data, vcov="iid", ssc=_SSC)
    return np.asarray(model.resid(), dtype=float)


def analyze_fwl_plot(
    df: pd.DataFrame,
    dv: str,
    var: str,
    controls: Sequence[str] | str | None = None,
    feffects: Sequence[str] | str | None = None,
    clusters: Sequence[str] | str | None = None,
    *,
    n_sample: int | None = 1000,
    alpha: float = 0.5,
    seed: int = 0,
) -> FWLPlotResult:
    """Frisch-Waugh-Lovell scatter of ``dv`` against the focal regressor ``var``.

    Residualizes both ``dv`` and ``var`` on ``controls`` **and** ``feffects`` over a single
    complete-case sample, then plots the residuals with an OLS fit line and a 95% pointwise
    confidence band. Fixed effects are absorbed (group-demeaned) by ``pyfixest``, so the plot
    shows the relationship between ``var`` and ``dv`` net of *both* the other regressors and
    the fixed effects. By the Frisch-Waugh-Lovell theorem the slope of this residual
    regression equals the coefficient on ``var`` in the full model; the annotation states
    this and reports the full model's standard error (clustered when ``clusters`` is given,
    matching :func:`expdpy.analyze_regression_table`), the sample size, and the within-R².

    Parameters
    ----------
    df
        Data frame containing the variables.
    dv
        Dependent (outcome) variable name.
    var
        Focal regressor whose partial relationship with ``dv`` is plotted.
    controls
        Additional regressors to residualize out (entered linearly). May be ``None``.
    feffects
        Fixed-effects variable name(s) absorbed during residualization. May be ``None``.
    clusters
        Cluster variable name(s) for the standard error reported in the annotation. Does not
        affect the point estimate or the plotted confidence band.
    n_sample
        Number of points drawn in the scatter (default 1000). The fit line and band are
        always computed on all complete-case rows. ``None`` plots every point.
    alpha
        Marker opacity for the scatter points (default 0.5).
    seed
        Seed for the point-subsampling RNG (default 0), for reproducible figures.

    Returns
    -------
    FWLPlotResult
        ``df`` (residual frame), ``fig`` (Plotly figure) and the scalar statistics
        ``slope``, ``se``, ``intercept``, ``n_obs`` and ``r2_within``.

    Examples
    --------
    Basic — partial relationship of the outcome and a single regressor:

    ```python
    import expdpy as ex
    from expdpy.data import load_kuznets

    df = load_kuznets()
    ex.analyze_fwl_plot(df, dv="gini_regional", var="log_gdp_pc").fig
    ```

    Advanced — residualize on the other cubic terms and two-way fixed effects, cluster
    the reported standard error by country, then read the FWL statistics back:

    ```python
    result = ex.analyze_fwl_plot(
        df,
        dv="gini_regional",
        var="log_gdp_pc",
        controls=["log_gdp_pc_sq", "log_gdp_pc_cu"],
        feffects=["country", "year"],
        clusters=["country"],
    )
    result.fig
    result.slope, result.se, result.r2_within
    ```
    """
    df = ensure_dataframe(df)
    controls = _as_list(controls)
    fe = _as_list(feffects)
    cl = _as_list(clusters)

    if var in controls:
        raise ValueError(f"focal variable '{var}' must not also be a control")
    needed = [dv, var, *controls, *fe, *cl]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"columns not found in df: {missing}")
    if not pdt.is_numeric_dtype(df[var]):
        raise TypeError(f"focal variable '{var}' must be numeric")
    if not (pdt.is_numeric_dtype(df[dv]) or pdt.is_bool_dtype(df[dv])):
        raise NotImplementedError(
            f"dependent variable '{dv}' is non-numeric (OLS only)"
        )

    var_label = resolve_label(df, var)
    dv_label = resolve_label(df, dv)
    used = list(dict.fromkeys(needed))
    data = df[used].dropna().copy()
    for f in fe:
        data[f] = data[f].astype("category")
    if len(data) < len(controls) + len(fe) + 3:
        raise ValueError("too few complete-case rows to fit the FWL regression")

    has_controls = len(controls) > 0
    has_fe = len(fe) > 0
    residualized = has_controls or has_fe
    fe_part = (" | " + " + ".join(fe)) if has_fe else ""

    # Full model: source of the reported slope (== focal coefficient) and clustered SE,
    # so the annotation is identical to analyze_regression_table.
    full_rhs = " + ".join([var, *controls])
    vcov: object = {"CRV1": " + ".join(cl)} if cl else "iid"
    full = pf.feols(f"{dv} ~ {full_rhs}{fe_part}", data=data, vcov=vcov, ssc=_SSC)
    slope = float(full.coef()[var])
    se = float(full.se()[var])
    n_obs = int(getattr(full, "_N", len(data)))
    r2_within = float(getattr(full, "_r2_within", np.nan))

    # Residualize y and x on the SAME controls + fixed effects (the FWL step).
    if residualized:
        rhs = " + ".join(controls) if has_controls else "1"
        y_resid = _residualize(data, dv, rhs, fe_part)
        x_resid = _residualize(data, var, rhs, fe_part)
        x_label, y_label = f"Residualized {var_label}", f"Residualized {dv_label}"
    else:
        y_resid = data[dv].to_numpy(dtype=float)
        x_resid = data[var].to_numpy(dtype=float)
        x_label, y_label = var_label, dv_label

    if len(x_resid) != len(y_resid):  # pragma: no cover - aligned by construction
        raise RuntimeError("residual vectors misaligned; cannot build FWL plot")
    if np.ptp(x_resid) <= 1e-10 * (1.0 + float(np.abs(x_resid).max())):
        raise ValueError(
            f"residualized '{var}' has (near) zero variance; the controls/fixed effects "
            "absorb all of its variation, so an FWL slope is not identified"
        )

    # OLS fit + 95% confidence band on the residual scatter (mirrors R add_prediction()).
    order = np.argsort(x_resid)
    xs = x_resid[order]
    ys = y_resid[order]
    ols = sm.OLS(ys, sm.add_constant(xs, has_constant="add")).fit()
    intercept = float(ols.params[0])
    pred = ols.get_prediction(sm.add_constant(xs, has_constant="add"))
    fit = pred.predicted_mean
    ci = pred.conf_int(alpha=0.05)

    resid_df = pd.DataFrame(
        {"x_resid": xs, "y_resid": ys, "fit": fit, "lwr": ci[:, 0], "upr": ci[:, 1]}
    )

    # Subsample plotted points only; the fit/band always use every row.
    plot_idx = np.arange(len(xs))
    if n_sample is not None and len(xs) > n_sample:
        rng = np.random.default_rng(seed)
        plot_idx = np.sort(rng.choice(len(xs), size=n_sample, replace=False))

    fig = _build_fig(
        resid_df, plot_idx, x_label, y_label, alpha, slope, se, n_obs, r2_within
    )
    return FWLPlotResult(
        df=resid_df,
        fig=fig,
        slope=slope,
        se=se,
        intercept=intercept,
        n_obs=n_obs,
        r2_within=r2_within,
    )


def _build_fig(
    resid_df: pd.DataFrame,
    plot_idx: np.ndarray,
    x_label: str,
    y_label: str,
    alpha: float,
    slope: float,
    se: float,
    n_obs: int,
    r2_within: float,
) -> go.Figure:
    """Assemble the FWL Plotly figure (confidence band, fit line, points, annotation)."""
    xs = resid_df["x_resid"].to_numpy()
    fig = go.Figure()
    # Confidence band, drawn first so it sits behind the line and points.
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([xs, xs[::-1]]),
            y=np.concatenate(
                [resid_df["upr"].to_numpy(), resid_df["lwr"].to_numpy()[::-1]]
            ),
            fill="toself",
            fillcolor="rgba(0,0,0,0.12)",
            line={"color": "rgba(0,0,0,0)"},
            hoverinfo="skip",
            showlegend=False,
            name="ci",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=resid_df["fit"].to_numpy(),
            mode="lines",
            line={"color": color_for(2), "width": 2},
            name="fit",
        )
    )
    sub = resid_df.iloc[plot_idx]
    fig.add_trace(
        go.Scatter(
            x=sub["x_resid"].to_numpy(),
            y=sub["y_resid"].to_numpy(),
            mode="markers",
            marker={"opacity": alpha, "color": color_for(0)},
            name="points",
        )
    )

    lines = [
        f"Slope = {slope:.4g} (= full-model coef on focal var)",
        f"SE = {se:.4g}",
        f"N = {n_obs:,}",
    ]
    if np.isfinite(r2_within):
        lines.append(f"Within R² = {r2_within:.3f}")
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.02,
        y=0.98,
        xanchor="left",
        yanchor="top",
        showarrow=False,
        align="left",
        bordercolor="rgba(0,0,0,0.2)",
        borderwidth=1,
        bgcolor="rgba(255,255,255,0.7)",
        text="<br>".join(lines),
    )
    apply_default_layout(fig, xaxis={"title": x_label}, yaxis={"title": y_label})
    return fig
