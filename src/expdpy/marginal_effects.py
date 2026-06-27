"""Marginal-effects plot for an interaction model (delta-method confidence band).

When a model contains an interaction ``focal * moderator``, the slope of ``focal`` is not a
single number — it is ``b_focal + b_interaction * moderator``, a line across the moderator's
range. :func:`analyze_marginal_effects_plot` fits that model via pyfixest, traces the marginal
effect with a delta-method confidence band, and reports the average marginal effect — the
panel-data analogue of Stata's ``margins`` for interactions.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pandas.api import types as pdt
from scipy.stats import norm

from expdpy._estimation import ModelSpec, VCovSpec, as_list, fit_model
from expdpy._labels import resolve_label
from expdpy._theme import apply_default_layout, color_for
from expdpy._types import MarginalEffectsResult
from expdpy._validation import drop_missing, ensure_dataframe

__all__ = ["analyze_marginal_effects_plot"]

_BAND_FILL = "rgba(78,121,167,0.2)"  # color_for(0) at 20% opacity


def _interaction_name(names: list[str], focal: str, moderator: str) -> str:
    """Return the interaction coefficient name (``focal:moderator`` either ordering)."""
    for cand in (f"{focal}:{moderator}", f"{moderator}:{focal}"):
        if cand in names:
            return cand
    raise ValueError(
        f"no interaction term between '{focal}' and '{moderator}' was estimated "
        "(it may be collinear with the controls or fixed effects)"
    )


def _build_fig(
    out: pd.DataFrame,
    df: pd.DataFrame,
    focal: str,
    moderator: str,
    alpha: float,
    title: str | None,
) -> go.Figure:
    """Assemble the marginal-effect line with its delta-method CI band and a zero line."""
    flabel, mlabel = resolve_label(df, focal), resolve_label(df, moderator)
    x = out[moderator]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=out["ci_upper"],
            mode="lines",
            line={"width": 0},
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=out["ci_lower"],
            mode="lines",
            line={"width": 0},
            fill="tonexty",
            fillcolor=_BAND_FILL,
            showlegend=False,
            hoverinfo="skip",
            name=f"{(1 - alpha) * 100:.0f}% CI",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=out["me"],
            mode="lines",
            line={"color": color_for(0), "width": 2.5},
            name=f"Marginal effect of {flabel}",
        )
    )
    fig.add_hline(y=0, line_dash="dash", line_color="rgba(0,0,0,0.4)")
    apply_default_layout(
        fig,
        xaxis={"title": mlabel},
        yaxis={"title": f"Marginal effect of {flabel}"},
        showlegend=False,
    )
    fig.update_layout(title=title or f"Marginal effect of {flabel} across {mlabel}")
    return fig


def analyze_marginal_effects_plot(
    df: pd.DataFrame,
    dv: str,
    focal: str,
    moderator: str,
    *,
    controls: Sequence[str] | None = None,
    feffects: Sequence[str] | None = None,
    clusters: Sequence[str] | None = None,
    at: Sequence[float] | None = None,
    n_points: int = 50,
    alpha: float = 0.05,
    title: str | None = None,
) -> MarginalEffectsResult:
    """Plot the marginal effect of ``focal`` across ``moderator`` for an interaction model.

    Fits ``dv ~ focal * moderator (+ controls) (| fixed effects)`` via pyfixest, then traces
    the marginal effect of ``focal`` — ``b_focal + b_interaction * moderator`` — across the
    moderator's observed range, with a delta-method confidence band. Also reports the **average
    marginal effect** (evaluated at the sample-mean moderator).

    Parameters
    ----------
    df
        Data frame containing the data.
    dv
        Dependent (outcome) variable name.
    focal
        The focal regressor whose marginal effect is traced.
    moderator
        The moderating variable it is interacted with.
    controls
        Additional exogenous control variable names.
    feffects
        Fixed-effects variable names absorbed by pyfixest.
    clusters
        Cluster variable name(s) for cluster-robust standard errors (and hence the band).
    at
        Explicit moderator values at which to evaluate the marginal effect. Defaults to an
        evenly spaced grid over the moderator's observed range.
    n_points
        Number of grid points when ``at`` is not given (default 50).
    alpha
        Significance level for the confidence band (default ``0.05`` → 95%).
    title
        Optional figure title.

    Returns
    -------
    MarginalEffectsResult
        ``df`` (the grid: ``<moderator>``, ``me``, ``se``, ``ci_lower``, ``ci_upper``),
        ``fig`` (the Plotly figure), ``focal`` / ``moderator``, and ``ame`` / ``ame_se`` (the
        average marginal effect and its standard error).

    Raises
    ------
    NotImplementedError
        If the dependent variable is non-numeric.
    ValueError
        If no interaction term between ``focal`` and ``moderator`` is estimated.

    Examples
    --------
    Does the gradient of inequality in income depend on trade openness? Trace the marginal
    effect of log GDP per capita across the range of the trade share:

    ```python
    import expdpy as ex
    from expdpy.data import load_kuznets, load_kuznets_data_def

    df = ex.set_labels(load_kuznets(), load_kuznets_data_def(), set_panel=True)
    result = ex.analyze_marginal_effects_plot(
        df, dv="gini_regional", focal="log_gdp_pc", moderator="trade_share"
    )
    result.fig
    result.ame
    print(result.interpret())
    ```
    """
    df = ensure_dataframe(df)
    controls_l = as_list(controls)
    fe_l = as_list(feffects)
    cl_l = as_list(clusters)

    if not (pdt.is_numeric_dtype(df[dv]) or pdt.is_bool_dtype(df[dv])):
        raise NotImplementedError(
            f"dependent variable '{dv}' is non-numeric; marginal effects are OLS-only."
        )

    used = list(dict.fromkeys([dv, focal, moderator, *controls_l, *fe_l, *cl_l]))
    data = drop_missing(df[used], used, func="analyze_marginal_effects_plot").copy()
    for fe in fe_l:
        data[fe] = data[fe].astype("category")

    vcov = VCovSpec(kind="CRV1", cluster=tuple(cl_l)) if cl_l else VCovSpec()
    spec = ModelSpec(
        dv=(dv,),
        idvs=(f"{focal}*{moderator}", *controls_l),
        feffects=tuple(fe_l),
        vcov=vcov,
    )
    model = fit_model(data, spec)

    names = [str(n) for n in model._coefnames]
    if focal not in names:
        raise ValueError(f"focal regressor '{focal}' was not estimated (collinear?)")
    inter = _interaction_name(names, focal, moderator)
    coef = {str(k): float(v) for k, v in model.coef().items()}
    vmat = np.asarray(model._vcov, dtype=float)
    i_f, i_int = names.index(focal), names.index(inter)
    b_f, b_int = coef[focal], coef[inter]
    v_ff, v_ii, v_fi = vmat[i_f, i_f], vmat[i_int, i_int], vmat[i_f, i_int]

    if at is not None:
        grid = np.asarray(list(at), dtype=float)
    else:
        lo, hi = float(data[moderator].min()), float(data[moderator].max())
        grid = np.linspace(lo, hi, int(n_points))

    me = b_f + b_int * grid
    var = v_ff + grid**2 * v_ii + 2.0 * grid * v_fi
    se = np.sqrt(np.clip(var, 0.0, None))
    zcrit = float(norm.ppf(1 - alpha / 2))
    out = pd.DataFrame(
        {
            moderator: grid,
            "me": me,
            "se": se,
            "ci_lower": me - zcrit * se,
            "ci_upper": me + zcrit * se,
        }
    )

    zbar = float(data[moderator].mean())
    ame = b_f + b_int * zbar
    ame_se = float(np.sqrt(max(v_ff + zbar**2 * v_ii + 2.0 * zbar * v_fi, 0.0)))

    fig = _build_fig(out, df, focal, moderator, alpha, title)
    return MarginalEffectsResult(
        df=out,
        fig=fig,
        focal=focal,
        moderator=moderator,
        ame=ame,
        ame_se=ame_se,
        model=model,
    )
