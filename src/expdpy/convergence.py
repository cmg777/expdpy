"""Convergence analysis for panel data: β-convergence and σ-convergence.

:func:`analyze_beta_convergence` automates the standard cross-country β-convergence workflow
from a single variable: the
**unconditional** growth-vs-initial-level regression (canonically the growth of GDP per capita
on its initial log level), the **conditional** version that partials out steady-state
controls via the Frisch-Waugh-Lovell theorem, the implied **speed of convergence** λ and
**half-life**, and a **rolling** fixed-width-window view of how the convergence slope evolves.

The variable is used **as supplied** (no automatic log): for the canonical case pass *log*
GDP per capita, so the annualized growth ``(y_end - y_start) / T`` is the log-difference and
the x-axis is the initial log level. The same machinery applies to any variable — income,
schooling, health — by passing it on whatever scale the analysis calls for.

The growth rate is built over a **common window**: by default the earliest and latest year in
the panel (override with ``start`` / ``end``), keeping the units observed at both endpoints so
every unit shares the same horizon ``T``. Controls enter as their **initial-year** values.

:func:`analyze_sigma_convergence` takes the complementary **σ-convergence** view: it tracks the
*cross-sectional dispersion* of a variable over time — the standard deviation, the Gini index
and the coefficient of variation — and tests whether that dispersion shrinks via an OLS trend
of the **log dispersion** on time. A negative trend is σ-convergence (the distribution is
narrowing). The variable is used **as supplied** here too; the panel must be **balanced** so
the dispersion is comparable across periods.

:func:`analyze_convergence_clubs` runs the **Phillips-Sul (2007/2009) log(t) club-convergence**
workflow end to end: it smooths each unit's series with the **Hodrick-Prescott filter**
(``lambda = 400`` for annual data), forms the **relative transition path**
``h_it = x_it / mean_i(x_it)``, runs the **log(t) regression test** for the whole panel and — if
global convergence is rejected — applies the data-driven **clustering algorithm** to split the
units into convergence **clubs**, then **merges** adjacent clubs that jointly converge. It is a
faithful port of the Stata ``psecta`` package (Du 2017): the log(t) statistic uses the
Phillips-Sul scalar-long-run-variance HAC (Andrews 1991 quadratic-spectral kernel with an
AR(1) automatic bandwidth), which standard OLS engines do not provide, so that one statistic is
computed in NumPy here. The variable is used **as supplied** (pass *log* GDP per capita / log
labor productivity); the panel must be **balanced** (the HP filter needs a gap-free series).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pyfixest as pf
import statsmodels.api as sm
from pandas.api import types as pdt
from plotly.subplots import make_subplots

from expdpy._labels import resolve_label
from expdpy._panel import resolve_panel
from expdpy._theme import apply_default_layout, color_for
from expdpy._types import (
    BetaConvergenceResult,
    ConvergenceClubsResult,
    SigmaConvergenceResult,
)
from expdpy._validation import ensure_dataframe
from expdpy.fwl import _residualize
from expdpy.regression import _SSC, _as_list

__all__ = [
    "analyze_beta_convergence",
    "analyze_sigma_convergence",
    "analyze_convergence_clubs",
]

_METRIC_KEYS = ("beta", "se", "r2", "n_obs", "speed", "half_life")
_METRIC_LABELS = (
    "β (initial level)",
    "Std. error",
    "R²",
    "N",
    "Speed of convergence (λ)",
    "Half-life",
)


def _speed_of_convergence(beta: float, horizon: float) -> float:
    """Return the structural speed lambda = -ln(1 + b*T) / T implied by slope ``beta``.

    Positive lambda means convergence. Returns ``nan`` when ``1 + b*T <= 0`` (the mapping is
    undefined) or the horizon is non-positive.
    """
    arg = 1.0 + beta * horizon
    if horizon <= 0.0 or not math.isfinite(arg) or arg <= 0.0:
        return float("nan")
    return -math.log(arg) / horizon


def _half_life(speed: float) -> float:
    """Return the half-life ln 2 / lambda (periods to close half a gap); ``nan`` if lambda<=0."""
    if not math.isfinite(speed) or speed <= 0.0:
        return float("nan")
    return math.log(2.0) / speed


def _cross_section(
    df: pd.DataFrame,
    var: str,
    controls: list[str],
    entity: str,
    time: str,
    start: float | None,
    end: float | None,
) -> tuple[pd.DataFrame, float, float, float]:
    """Build the one-row-per-unit growth cross-section over a common window.

    Returns ``(cs, horizon, t0, t1)`` where ``cs`` has columns ``entity``, ``initial``,
    ``final``, ``growth`` and one ``<control>__0`` per control (its initial-year value).
    """
    cols = list(dict.fromkeys([entity, time, var, *controls]))
    sub = df[cols].copy()
    sub[time] = pd.to_numeric(sub[time], errors="coerce")
    years = sub[time].dropna()
    if years.empty:
        raise ValueError(f"time column {time!r} has no numeric values")
    t0 = float(start) if start is not None else float(years.min())
    t1 = float(end) if end is not None else float(years.max())
    if t1 <= t0:
        raise ValueError(f"end ({t1}) must be after start ({t0})")
    horizon = t1 - t0

    init = sub[sub[time] == t0].groupby(entity, observed=True).first()
    fin = sub[sub[time] == t1].groupby(entity, observed=True).first()
    if init.empty or fin.empty:
        raise ValueError(
            f"no units with data at both start ({t0:g}) and end ({t1:g}); "
            "check the years present in the panel or pass start=/end="
        )

    cs = pd.DataFrame(index=init.index)
    cs["initial"] = init[var]
    cs["final"] = fin[var].reindex(cs.index)
    for c in controls:
        cs[f"{c}__0"] = init[c]
    cs = cs.dropna(subset=["initial", "final"])
    cs["growth"] = (cs["final"] - cs["initial"]) / horizon
    cs = cs.reset_index()
    cs[entity] = cs[entity].astype(str)
    return cs, horizon, t0, t1


def _fit(
    data: pd.DataFrame, controls0: list[str], vcov: str
) -> tuple[Any, float, float, float, int]:
    """Fit ``growth ~ initial [+ controls0]`` and return ``(model, β, se, r2, n)``."""
    rhs = " + ".join(["initial", *controls0]) if controls0 else "initial"
    model = pf.feols(f"growth ~ {rhs}", data=data, vcov=vcov, ssc=_SSC)
    beta = float(model.coef()["initial"])
    se = float(model.se()["initial"])
    r2 = float(getattr(model, "_r2", np.nan))
    n = int(getattr(model, "_N", len(data)))
    return model, beta, se, r2, n


def _scatter(
    xv: np.ndarray,
    yv: np.ndarray,
    entities: np.ndarray,
    x_label: str,
    y_label: str,
    stats: tuple[float, float, float, int, float, float],
    title: str | None,
) -> go.Figure:
    """Interactive growth-vs-initial scatter: OLS fit + 95% band, unit hover, stat box."""
    beta, se, r2, n, speed, hl = stats
    finite = np.isfinite(xv) & np.isfinite(yv)
    xv, yv, entities = xv[finite], yv[finite], entities[finite]
    order = np.argsort(xv)
    xs, ys, ents = xv[order], yv[order], entities[order]

    fig = go.Figure()
    if xs.size >= 3 and np.ptp(xs) > 0:
        design = sm.add_constant(xs, has_constant="add")
        pred = sm.OLS(ys, design).fit().get_prediction(design)
        fit = pred.predicted_mean
        ci = pred.conf_int(alpha=0.05)
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([xs, xs[::-1]]),
                y=np.concatenate([ci[:, 1], ci[:, 0][::-1]]),
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
                y=fit,
                mode="lines",
                line={"color": color_for(2), "width": 2},
                name="fit",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="markers",
            marker={
                "color": color_for(0),
                "size": 8,
                "opacity": 0.75,
                "line": {"color": "white", "width": 0.5},
            },
            customdata=ents,
            hovertemplate=(
                "%{customdata}<br>initial=%{x:.4g}<br>growth=%{y:.4g}<extra></extra>"
            ),
            name="units",
            showlegend=False,
        )
    )

    lines = [f"β = {beta:.4g}", f"SE = {se:.4g}", f"R² = {r2:.3f}", f"N = {n:,}"]
    if math.isfinite(speed):
        lines.append(f"speed λ = {speed:.4g}")
    if math.isfinite(hl):
        lines.append(f"half-life = {hl:.4g}")
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
    if title:
        fig.update_layout(title=title)
    return fig


def _rolling(
    df: pd.DataFrame,
    var: str,
    entity: str,
    time: str,
    window: int | None,
    vcov: str,
    min_obs: int,
) -> tuple[pd.DataFrame | None, int]:
    """Estimate β-convergence on every fixed-width window of ``window`` periods.

    Returns ``(rolling_frame, window)`` or ``(None, window)`` when the panel is too short or
    no window has at least ``min_obs`` units.
    """
    years = sorted(pd.to_numeric(df[time], errors="coerce").dropna().unique())
    m = len(years)
    win = int(window) if window is not None else max(2, m // 2)
    if win < 1 or m < win + 1:
        return None, win

    rows: list[dict[str, float]] = []
    for i in range(0, m - win):
        t0, t1 = float(years[i]), float(years[i + win])
        cs, horizon, _, _ = _cross_section(df, var, [], entity, time, t0, t1)
        data = cs.dropna(subset=["initial", "growth"])
        if len(data) < min_obs or np.ptp(data["initial"].to_numpy()) <= 0:
            continue
        model = pf.feols("growth ~ initial", data=data, vcov=vcov, ssc=_SSC)
        beta = float(model.coef()["initial"])
        se = float(model.se()["initial"])
        sp = _speed_of_convergence(beta, horizon)
        rows.append(
            {
                "window_start": t0,
                "window_end": t1,
                "horizon": horizon,
                "beta": beta,
                "se": se,
                "ci_lower": beta - 1.96 * se,
                "ci_upper": beta + 1.96 * se,
                "speed": sp,
                "half_life": _half_life(sp),
                "n": float(len(data)),
            }
        )
    if not rows:
        return None, win
    return pd.DataFrame(rows), win


def _rolling_fig(roll: pd.DataFrame, title: str | None) -> go.Figure:
    """Plot the rolling convergence slope β (with 95% band) against the window start year."""
    xs = roll["window_start"].to_numpy(dtype=float)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([xs, xs[::-1]]),
            y=np.concatenate(
                [roll["ci_upper"].to_numpy(), roll["ci_lower"].to_numpy()[::-1]]
            ),
            fill="toself",
            fillcolor="rgba(78,121,167,0.18)",
            line={"color": "rgba(0,0,0,0)"},
            hoverinfo="skip",
            showlegend=False,
            name="ci",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=roll["beta"].to_numpy(),
            mode="lines+markers",
            line={"color": color_for(0), "width": 2},
            marker={"color": color_for(0), "size": 7},
            customdata=np.stack(
                [
                    roll["window_end"].to_numpy(),
                    roll["speed"].to_numpy(),
                    roll["half_life"].to_numpy(),
                ],
                axis=1,
            ),
            hovertemplate=(
                "window %{x:.0f}→%{customdata[0]:.0f}<br>β = %{y:.4g}<br>"
                "speed λ = %{customdata[1]:.4g}, half-life = %{customdata[2]:.4g}<extra></extra>"
            ),
            name="β",
            showlegend=False,
        )
    )
    fig.add_hline(y=0, line_dash="dot", line_color="rgba(0,0,0,0.4)")
    apply_default_layout(
        fig,
        xaxis={"title": "Window start year"},
        yaxis={"title": "Convergence slope β"},
    )
    if title:
        fig.update_layout(title=title)
    return fig


def _metric_values(
    beta: float, se: float, r2: float, n: int, speed: float, hl: float
) -> list[float]:
    """Order the six comparison metrics to match ``_METRIC_KEYS`` / ``_METRIC_LABELS``."""
    return [beta, se, r2, float(n), speed, hl]


def _fmt_metric(label: str, value: float) -> str:
    """Format one comparison-table cell (integer N, 3dp R², else 4 significant digits)."""
    if not math.isfinite(value):
        return "—"
    if label == "N":
        return f"{round(value):,}"
    if label == "R²":
        return f"{value:.3f}"
    return f"{value:.4g}"


def _summary_and_gt(
    uncond: list[float],
    cond: list[float] | None,
    var_label: str,
    horizon: float,
) -> tuple[pd.DataFrame, Any]:
    """Build the numeric ``summary`` frame and its Great-Tables comparison rendering."""
    from great_tables import GT

    summary = pd.DataFrame({"metric": list(_METRIC_KEYS), "unconditional": uncond})
    disp = pd.DataFrame(
        {
            "Metric": list(_METRIC_LABELS),
            "Unconditional": [
                _fmt_metric(lbl, v)
                for lbl, v in zip(_METRIC_LABELS, uncond, strict=True)
            ],
        }
    )
    if cond is not None:
        summary["conditional"] = cond
        disp["Conditional"] = [
            _fmt_metric(lbl, v) for lbl, v in zip(_METRIC_LABELS, cond, strict=True)
        ]

    gt = (
        GT(disp, rowname_col="Metric")
        .tab_header(
            title=f"β-convergence: {var_label}",
            subtitle=f"growth over a {horizon:g}-period horizon vs. initial level",
        )
        .tab_source_note(
            "β < 0 indicates convergence. Speed λ = -ln(1 + β·T)/T per period; "
            "half-life = ln 2 / λ. Conditional partials out the initial-year controls (FWL)."
        )
    )
    return summary, gt


def analyze_beta_convergence(
    df: pd.DataFrame,
    var: str,
    controls: Sequence[str] | str | None = None,
    *,
    entity: str | None = None,
    time: str | None = None,
    start: float | None = None,
    end: float | None = None,
    rolling: bool = True,
    window: int | None = None,
    min_obs: int = 10,
    vcov: Literal["hetero", "iid"] = "hetero",
    title: str | None = None,
) -> BetaConvergenceResult:
    """Unconditional and conditional β-convergence for a panel variable.

    Regresses each unit's annualized growth ``(y_end - y_start) / T`` on its initial level
    ``y_start`` over a common window. A **negative** slope β is convergence (units that start
    lower grow faster). With ``controls`` the conditional slope is read off a Frisch-Waugh-
    Lovell partial regression that holds the controls' initial-year values fixed. The slope
    maps to a speed ``λ = -ln(1 + β·T)/T`` and half-life ``ln 2 / λ``.

    The variable is used **as supplied** — no log is taken — so for the canonical income case
    pass *log* GDP per capita (then growth is the annualized log-difference and the x-axis is
    the initial log level).

    Parameters
    ----------
    df
        Panel data frame.
    var
        Numeric variable to analyse (e.g. ``"log_gdp_pc"``).
    controls
        Optional control name(s). Each enters the conditional model as its **initial-year**
        value. ``None`` runs the unconditional analysis only.
    entity, time
        Panel identifiers. Default to those declared via :func:`expdpy.set_panel`.
    start, end
        First and last year used to form the growth rate. Default to the earliest and latest
        year in the panel; only units observed at **both** endpoints are kept.
    rolling
        When ``True`` (default), also estimate the convergence slope on every fixed-width
        window of ``window`` periods and return the time path.
    window
        Width of the rolling window in **periods** (sorted distinct years). Default
        ``max(2, n_periods // 2)``.
    min_obs
        Minimum number of units required in the cross-section and in each rolling window.
    vcov
        Standard-error type for the reported coefficients: ``"hetero"`` (HC1, the default) or
        ``"iid"``. Does not affect the point estimate or the plotted band.
    title
        Title for the unconditional scatter (the others get descriptive titles).

    Returns
    -------
    BetaConvergenceResult
        The cross-section ``df``; the unconditional scatter ``fig``; the conditional FWL
        scatter ``fig_conditional`` and rolling plot ``fig_rolling`` (each ``None`` when not
        applicable); the comparison table ``gt`` / ``summary``; the ``rolling`` frame; the
        fitted ``models``; and the scalar fit statistics (``beta``/``speed``/``half_life`` and
        their ``*_cond`` counterparts).

    Examples
    --------
    Unconditional convergence of (log) GDP per capita across countries:

    ```python
    import numpy as np
    import expdpy as ex
    from expdpy.data import load_gapminder

    df = load_gapminder()
    df["log_gdppc"] = np.log(df["gdpPercap"])
    res = ex.analyze_beta_convergence(df, "log_gdppc", entity="country", time="year")
    res.fig
    res.gt
    res.speed, res.half_life
    ```

    Conditional convergence, controlling for initial life expectancy:

    ```python
    ex.analyze_beta_convergence(
        df, "log_gdppc", controls=["lifeExp"], entity="country", time="year"
    ).fig_conditional
    ```
    """
    df = ensure_dataframe(df)
    controls = _as_list(controls)
    entity, time = resolve_panel(
        df, entity, time, require_entity=True, require_time=True
    )
    assert entity is not None and time is not None  # guaranteed by require_* above

    missing = [c for c in [var, *controls] if c not in df.columns]
    if missing:
        raise KeyError(f"columns not found in df: {missing}")
    if not pdt.is_numeric_dtype(df[var]):
        raise TypeError(f"var {var!r} must be numeric")
    for c in controls:
        if not pdt.is_numeric_dtype(df[c]):
            raise TypeError(f"control {c!r} must be numeric")

    var_label = resolve_label(df, var)
    notes: list[str] = []

    n_dup = int(df.duplicated([entity, time]).sum())
    if n_dup:
        # Reduce with a NaN-skipping groupby (as the sigma/clubs paths do) so a NaN-valued
        # duplicate ordered first cannot evict an otherwise-valid observation.
        df = df.groupby([entity, time], observed=True, as_index=False).first()
        notes.append(
            f"found duplicate (entity, time) rows; kept the first non-missing of each "
            f"({n_dup} dropped)"
        )

    cs, horizon, _t0, _t1 = _cross_section(df, var, controls, entity, time, start, end)
    uncond = cs.dropna(subset=["initial", "growth"])
    if len(uncond) < max(3, min_obs):
        raise ValueError(
            f"too few units with data at both endpoints ({len(uncond)}); need >= "
            f"{max(3, min_obs)}. Try a different start/end or lower min_obs."
        )
    init_vals = uncond["initial"].to_numpy(dtype=float)
    if np.ptp(init_vals) <= 1e-10 * (1.0 + float(np.abs(init_vals).max())):
        raise ValueError(
            f"initial {var!r} has (near) zero variance across units; a convergence slope is "
            "not identified"
        )

    model_u, beta, se, r2, n = _fit(uncond, [], vcov)
    speed = _speed_of_convergence(beta, horizon)
    hl = _half_life(speed)
    fig = _scatter(
        uncond["initial"].to_numpy(dtype=float),
        uncond["growth"].to_numpy(dtype=float),
        uncond[entity].to_numpy(),
        f"Initial {var_label}",
        f"Growth of {var_label}",
        (beta, se, r2, n, speed, hl),
        title or f"Unconditional β-convergence: {var_label}",
    )
    models: list[Any] = [model_u]

    fig_conditional: go.Figure | None = None
    beta_c = se_c = r2_c = speed_c = hl_c = float("nan")
    n_c = 0
    if controls:
        ctrl0 = [f"{c}__0" for c in controls]
        cond = cs.dropna(subset=["initial", "growth", *ctrl0])
        if len(cond) < max(len(controls) + 3, min_obs):
            notes.append(
                "too few complete-case units for the conditional model; it was skipped"
            )
        else:
            model_c, beta_c, se_c, r2_c, n_c = _fit(cond, ctrl0, vcov)
            dropped = [
                c
                for c, t in zip(controls, ctrl0, strict=True)
                if t not in model_c.coef().index
            ]
            if dropped:
                # pyfixest silently drops a constant or collinear regressor; reporting the
                # surviving (unconditional) slope as "conditional" would be misleading.
                beta_c = se_c = r2_c = float("nan")
                n_c = 0
                notes.append(
                    f"conditional control(s) {dropped} are collinear with the initial level "
                    "or constant and were dropped by the estimator; the conditional model "
                    "was skipped"
                )
            else:
                speed_c = _speed_of_convergence(beta_c, horizon)
                hl_c = _half_life(speed_c)
                models.append(model_c)
                rhs = " + ".join(ctrl0)
                x_res = _residualize(cond, "initial", rhs, "")
                y_res = _residualize(cond, "growth", rhs, "")
                fig_conditional = _scatter(
                    x_res,
                    y_res,
                    cond[entity].to_numpy(),
                    f"Residualized initial {var_label}",
                    f"Residualized growth of {var_label}",
                    (beta_c, se_c, r2_c, n_c, speed_c, hl_c),
                    f"Conditional β-convergence (controls: {', '.join(controls)})",
                )

    rolling_df: pd.DataFrame | None = None
    fig_rolling: go.Figure | None = None
    if rolling:
        rolling_df, _win = _rolling(df, var, entity, time, window, vcov, min_obs)
        if rolling_df is None:
            notes.append("panel too short for a rolling window; rolling was skipped")
        else:
            fig_rolling = _rolling_fig(
                rolling_df, f"Rolling β-convergence: {var_label}"
            )

    has_cond = bool(controls) and n_c > 0
    summary, gt = _summary_and_gt(
        _metric_values(beta, se, r2, n, speed, hl),
        _metric_values(beta_c, se_c, r2_c, n_c, speed_c, hl_c) if has_cond else None,
        var_label,
        horizon,
    )

    return BetaConvergenceResult(
        df=cs,
        fig=fig,
        fig_conditional=fig_conditional,
        fig_rolling=fig_rolling,
        gt=gt,
        summary=summary,
        rolling=rolling_df,
        models=models,
        var=var,
        controls=tuple(controls),
        horizon=horizon,
        beta=beta,
        se=se,
        r2=r2,
        n_obs=n,
        speed=speed,
        half_life=hl,
        beta_cond=beta_c,
        se_cond=se_c,
        r2_cond=r2_c,
        n_obs_cond=n_c,
        speed_cond=speed_c,
        half_life_cond=hl_c,
        notes=tuple(notes),
    )


# ============================== sigma convergence ==============================
# σ-convergence: does the cross-sectional *dispersion* of a variable shrink over time?

_SIGMA_MEASURES = ("std", "gini", "cv")
_SIGMA_LABELS = {
    "std": "Standard deviation",
    "gini": "Gini index",
    "cv": "Coefficient of variation",
}


def _gini(x: np.ndarray) -> float:
    """Return the Gini coefficient of ``x`` (relative mean absolute difference / 2).

    Uses the sorted-order identity ``G = 2*sum(i*x_(i)) / (n*sum x) - (n + 1)/n`` on the values
    sorted ascending, which equals the mean absolute difference over ``2*mean``. Returns
    ``nan`` for fewer than two finite values, a non-positive sum, or any negative value (the
    index is only defined on non-negative data).
    """
    v = np.asarray(x, dtype=float)
    v = v[np.isfinite(v)]
    n = v.size
    if n < 2 or bool(np.any(v < 0.0)):
        return float("nan")
    total = float(v.sum())
    if total <= 0.0:
        return float("nan")
    v = np.sort(v)
    idx = np.arange(1, n + 1, dtype=float)
    return float(2.0 * float(np.sum(idx * v)) / (n * total) - (n + 1.0) / n)


def _balance_offenders(
    work: pd.DataFrame, entity: str, time: str
) -> tuple[int, int, int, int]:
    """Return ``(n_units, n_periods, units_missing, periods_missing)`` describing balance.

    A panel is **balanced** when every unit is observed in every period — i.e. both
    ``units_missing`` and ``periods_missing`` are zero. ``units_missing`` counts units absent
    from at least one period; ``periods_missing`` counts periods missing at least one unit.
    """
    n_periods = int(work[time].nunique())
    n_units = int(work[entity].nunique())
    per_unit = work.groupby(entity, observed=True)[time].nunique()
    per_period = work.groupby(time, observed=True)[entity].nunique()
    units_missing = int((per_unit < n_periods).sum())
    periods_missing = int((per_period < n_units).sum())
    return n_units, n_periods, units_missing, periods_missing


def _period_table(work: pd.DataFrame, var: str, time: str) -> pd.DataFrame:
    """One row per period: ``time``, ``n_units``, ``mean``, ``std`` (ddof=1), ``gini``, ``cv``."""
    rows: list[dict[str, float]] = []
    for t in sorted(work[time].unique()):
        v = work.loc[work[time] == t, var].to_numpy(dtype=float)
        mean = float(np.mean(v))
        std = float(np.std(v, ddof=1)) if v.size > 1 else float("nan")
        rows.append(
            {
                time: float(t),
                "n_units": int(v.size),
                "mean": mean,
                "std": std,
                "gini": _gini(v),
                "cv": std / mean if mean != 0.0 else float("nan"),
            }
        )
    return pd.DataFrame(rows).sort_values(time).reset_index(drop=True)


def _dispersion_trend(
    tab: pd.DataFrame, measure: str, time: str, vcov: str
) -> tuple[Any | None, float, float, float, float, int]:
    """Fit ``log(<measure>) ~ time`` over periods with a positive, finite dispersion.

    Returns ``(model, slope, se, pvalue, r2, n_used)``. ``model`` is ``None`` and the scalars
    are ``nan`` when fewer than three usable periods remain (``log`` needs a positive
    dispersion, so zero-dispersion periods are dropped). The slope is the average proportional
    change in the dispersion per period; a **negative** slope is σ-convergence.
    """
    disp = tab[measure].to_numpy(dtype=float)
    tv = tab[time].to_numpy(dtype=float)
    ok = np.isfinite(disp) & (disp > 0.0)
    n_used = int(ok.sum())
    if n_used < 3:
        return None, float("nan"), float("nan"), float("nan"), float("nan"), n_used
    data = pd.DataFrame({"log_disp": np.log(disp[ok]), "t": tv[ok]})
    model = pf.feols("log_disp ~ t", data=data, vcov=vcov, ssc=_SSC)
    slope = float(model.coef()["t"])
    se = float(model.se()["t"])
    try:
        pval = float(model.pvalue()["t"])
    except Exception:  # pragma: no cover - pyfixest always provides p-values
        pval = float("nan")
    r2 = float(getattr(model, "_r2", np.nan))
    return model, slope, se, pval, r2, n_used


def _sigma_annotation(
    trends: dict[str, tuple[Any | None, float, float, float, float, int]],
    n_periods: int,
) -> list[str]:
    """Build the annotation-box lines reporting each measure's log-dispersion trend."""

    def line(measure: str, label: str) -> str:
        slope, pval = trends[measure][1], trends[measure][3]
        if not math.isfinite(slope):
            return f"{label}: not estimated"
        verdict = "converging" if slope < 0 else "diverging"
        ptxt = f", p = {pval:.2g}" if math.isfinite(pval) else ""
        return f"{label} trend = {slope:.3g}/period ({verdict}{ptxt})"

    return [line("std", "Std"), line("gini", "Gini"), f"periods = {n_periods}"]


def _dual_axis_fig(
    tab: pd.DataFrame,
    trends: dict[str, tuple[Any | None, float, float, float, float, int]],
    time: str,
    time_label: str,
    var_label: str,
    title: str | None,
) -> go.Figure:
    """Build the dual-axis figure: std (left axis) and Gini (right axis) over time."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    tv = tab[time].to_numpy(dtype=float)
    std_color, gini_color = color_for(0), color_for(1)

    fig.add_trace(
        go.Scatter(
            x=tv,
            y=tab["std"].to_numpy(dtype=float),
            mode="lines+markers",
            name="Std. dev.",
            line={"color": std_color, "width": 2},
            marker={"color": std_color, "size": 7},
            hovertemplate=f"{time_label} = %{{x}}<br>std = %{{y:.4g}}<extra></extra>",
        ),
        secondary_y=False,
    )
    gini_v = tab["gini"].to_numpy(dtype=float)
    has_gini = bool(np.any(np.isfinite(gini_v)))
    if has_gini:
        fig.add_trace(
            go.Scatter(
                x=tv,
                y=gini_v,
                mode="lines+markers",
                name="Gini index",
                line={"color": gini_color, "width": 2},
                marker={"color": gini_color, "size": 7},
                hovertemplate=f"{time_label} = %{{x}}<br>Gini = %{{y:.4g}}<extra></extra>",
            ),
            secondary_y=True,
        )

    # Dashed exp(log-trend) overlays so the fitted convergence path is visible.
    for measure, color, sec in (("std", std_color, False), ("gini", gini_color, True)):
        model = trends[measure][0]
        if model is None or (measure == "gini" and not has_gini):
            continue
        b0 = float(model.coef()["Intercept"])
        b1 = float(trends[measure][1])
        fig.add_trace(
            go.Scatter(
                x=tv,
                y=np.exp(b0 + b1 * tv),
                mode="lines",
                line={"color": color, "width": 1, "dash": "dash"},
                hoverinfo="skip",
                showlegend=False,
                name=f"{measure} trend",
            ),
            secondary_y=sec,
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
        bordercolor="rgba(0,0,0,0.2)",
        borderwidth=1,
        bgcolor="rgba(255,255,255,0.7)",
        text="<br>".join(_sigma_annotation(trends, len(tab))),
    )
    apply_default_layout(fig, xaxis={"title": time_label})
    fig.update_yaxes(
        title_text=f"Std. dev. of {var_label}", color=std_color, secondary_y=False
    )
    fig.update_yaxes(
        title_text=f"Gini of {var_label}", color=gini_color, secondary_y=True
    )
    if title:
        fig.update_layout(title=title)
    return fig


def _sigma_summary_and_gt(
    trends: dict[str, tuple[Any | None, float, float, float, float, int]],
    var_label: str,
    n_periods: int,
    n_units: int,
) -> tuple[pd.DataFrame, Any]:
    """Build the numeric ``summary`` frame and its Great-Tables trend rendering."""
    from great_tables import GT

    def converging(slope: float) -> bool:
        return bool(slope < 0) if math.isfinite(slope) else False

    summary = pd.DataFrame(
        {
            "measure": list(_SIGMA_MEASURES),
            "slope": [trends[m][1] for m in _SIGMA_MEASURES],
            "se": [trends[m][2] for m in _SIGMA_MEASURES],
            "pvalue": [trends[m][3] for m in _SIGMA_MEASURES],
            "r2": [trends[m][4] for m in _SIGMA_MEASURES],
            "n_periods_used": [trends[m][5] for m in _SIGMA_MEASURES],
            "converging": [converging(trends[m][1]) for m in _SIGMA_MEASURES],
        }
    )

    def fmt(value: float, *, dp: bool = False) -> str:
        if not math.isfinite(value):
            return "—"
        return f"{value:.3f}" if dp else f"{value:.4g}"

    disp = pd.DataFrame(
        {
            "Measure": [_SIGMA_LABELS[m] for m in _SIGMA_MEASURES],
            "Trend (per period)": [fmt(trends[m][1]) for m in _SIGMA_MEASURES],
            "Std. error": [fmt(trends[m][2]) for m in _SIGMA_MEASURES],
            "p-value": [fmt(trends[m][3], dp=True) for m in _SIGMA_MEASURES],
            "σ-convergence": [
                "—"
                if not math.isfinite(trends[m][1])
                else ("yes" if trends[m][1] < 0 else "no")
                for m in _SIGMA_MEASURES
            ],
        }
    )
    gt = (
        GT(disp, rowname_col="Measure")
        .tab_header(
            title=f"σ-convergence: {var_label}",
            subtitle=f"trend of log dispersion over {n_periods} periods, {n_units} units",
        )
        .tab_source_note(
            "A negative trend in log dispersion is σ-convergence (the cross-sectional "
            "distribution is narrowing). Trend = OLS slope of ln(dispersion) on time."
        )
    )
    return summary, gt


def analyze_sigma_convergence(
    df: pd.DataFrame,
    var: str,
    *,
    entity: str | None = None,
    time: str | None = None,
    start: float | None = None,
    end: float | None = None,
    min_periods: int = 3,
    vcov: Literal["hetero", "iid"] = "hetero",
    title: str | None = None,
) -> SigmaConvergenceResult:
    r"""σ-convergence: track and test the cross-sectional dispersion of a panel variable.

    For each period the function measures how spread out ``var`` is **across units** — the
    standard deviation, the Gini index and the coefficient of variation — and then asks whether
    that dispersion shrinks over time by regressing the **log dispersion** on time. A
    **negative** trend slope is σ-convergence: the cross-sectional distribution is narrowing
    (units are becoming more alike). This is the distributional complement to β-convergence
    (which compares growth to the initial level); see :func:`analyze_beta_convergence`.

    The variable is used **as supplied** — no log or other transform is taken — so pass it on
    whatever scale the analysis calls for. The panel must be **balanced** (every unit present
    in every period) so the dispersion is comparable across periods.

    Parameters
    ----------
    df
        Panel data frame.
    var
        Numeric variable whose cross-sectional dispersion is tracked (e.g. ``"lifeExp"``).
        Used as supplied. The Gini index additionally requires non-negative values.
    entity, time
        Panel identifiers. Default to those declared via :func:`expdpy.set_panel`.
    start, end
        Optional first and last period to include. Default to the full range; the retained
        window must still be balanced.
    min_periods
        Minimum number of periods required to estimate a dispersion trend (at least 3).
    vcov
        Standard-error type for the trend coefficients: ``"hetero"`` (HC1, the default) or
        ``"iid"``. Does not change the point estimate.
    title
        Title for the dual-axis figure.

    Returns
    -------
    SigmaConvergenceResult
        The per-period dispersion table ``df`` (``time``, ``n_units``, ``mean``, ``std``,
        ``gini``, ``cv``); the dual-axis ``fig`` (std on the left axis, Gini on the right);
        the trend table ``gt`` / ``summary`` (one row per measure: slope, SE, p-value, R²);
        the fitted trend ``models``; the panel dimensions ``n_periods`` / ``n_units``; and the
        headline trend scalars (``std_slope`` and its ``std_se`` / ``std_pvalue`` / ``std_r2``,
        plus the ``gini_*`` and ``cv_*`` counterparts). ``notes`` records any degraded measure
        (e.g. a Gini set to ``NaN`` because the variable has negative values).

    Notes
    -----
    For a measure of dispersion ``D_t`` computed cross-sectionally at each time ``t``, the
    trend is the OLS slope ``b`` in

    .. math:: \ln D_t = a + b \, t + \varepsilon_t,

    so ``b`` is the average proportional change in dispersion per period and ``b < 0`` is
    σ-convergence. The standard deviation uses ``ddof = 1``; the Gini index is the relative
    mean absolute difference over twice the mean; the coefficient of variation is the standard
    deviation over the mean. If every unit's value contracts geometrically toward a common
    mean at rate ``rho`` per period, each dispersion measure scales as ``rho**t`` and the trend
    recovers ``b = ln(rho)`` exactly. See Barro & Sala-i-Martin, *Economic Growth*, ch. 11.

    Examples
    --------
    Cross-country convergence of life expectancy (a bounded, non-negative level):

    ```python
    import expdpy as ex
    from expdpy.data import load_gapminder

    res = ex.analyze_sigma_convergence(
        load_gapminder(), "lifeExp", entity="country", time="year"
    )
    res.fig          # std (left axis) and Gini (right axis) over time
    res.gt           # the trend table
    res.std_slope    # < 0 indicates σ-convergence
    ```
    """
    df = ensure_dataframe(df)
    entity, time = resolve_panel(
        df, entity, time, require_entity=True, require_time=True
    )
    assert entity is not None and time is not None  # guaranteed by require_* above

    if var not in df.columns:
        raise KeyError(f"column not found in df: {var!r}")
    if not pdt.is_numeric_dtype(df[var]):
        raise TypeError(f"var {var!r} must be numeric")

    var_label = resolve_label(df, var)
    time_label = resolve_label(df, time)
    notes: list[str] = []

    work = df[[entity, time, var]].copy()
    work[time] = pd.to_numeric(work[time], errors="coerce")
    work = work.dropna(subset=[time, var])
    if work.empty:
        raise ValueError(
            f"no rows with both a numeric {time!r} and a non-missing {var!r}"
        )
    if start is not None:
        work = work[work[time] >= float(start)]
    if end is not None:
        work = work[work[time] <= float(end)]

    before = len(work)
    work = work.groupby([time, entity], observed=True, as_index=False).first()
    if len(work) < before:
        notes.append(
            f"found duplicate (entity, time) rows; kept the first of each "
            f"({before - len(work)} dropped)"
        )

    n_units, n_periods, units_missing, periods_missing = _balance_offenders(
        work, entity, time
    )
    if units_missing or periods_missing:
        raise ValueError(
            f"panel is not balanced: {units_missing} of {n_units} units are missing in some "
            f"period and {periods_missing} of {n_periods} periods are missing some units. "
            "σ-convergence compares dispersion across a fixed set of units; restrict to a "
            "balanced window with start=/end= or drop the offending units."
        )
    if n_units < 2:
        raise ValueError(
            f"need >= 2 units to measure cross-sectional dispersion; got {n_units}"
        )
    if n_periods < max(3, min_periods):
        raise ValueError(
            f"need >= {max(3, min_periods)} periods to estimate a dispersion trend; "
            f"got {n_periods}"
        )

    tab = _period_table(work, var, time)

    # Degradation guards: Gini needs non-negative values; CV needs a stable, non-zero mean.
    if bool((work[var] < 0.0).any()):
        tab["gini"] = float("nan")
        notes.append(
            f"{var!r} has negative values; the Gini index is undefined and was set to NaN"
        )
    means = tab["mean"].to_numpy(dtype=float)
    max_abs = float(np.max(np.abs(means)))
    # CV = std / mean is only meaningful for a strictly positive mean; degrade it whenever any
    # period mean is negative or near zero (a sign flip is subsumed by the negative case).
    if bool(np.min(means) <= 1e-10 * (1.0 + max_abs)):
        tab["cv"] = float("nan")
        notes.append(
            "a period mean is not strictly positive (negative or near zero); the "
            "coefficient of variation is undefined and was set to NaN"
        )

    trends = {m: _dispersion_trend(tab, m, time, vcov) for m in _SIGMA_MEASURES}
    for m in _SIGMA_MEASURES:
        if trends[m][0] is None:
            notes.append(
                f"fewer than 3 periods with a positive {m}; its trend was not estimated"
            )

    fig = _dual_axis_fig(
        tab,
        trends,
        time,
        time_label,
        var_label,
        title or f"σ-convergence: {var_label}",
    )
    summary, gt = _sigma_summary_and_gt(trends, var_label, n_periods, n_units)
    models = [trends[m][0] for m in _SIGMA_MEASURES if trends[m][0] is not None]

    return SigmaConvergenceResult(
        df=tab,
        fig=fig,
        gt=gt,
        summary=summary,
        models=models,
        var=var,
        entity=entity,
        time=time,
        n_periods=n_periods,
        n_units=n_units,
        std_slope=trends["std"][1],
        std_se=trends["std"][2],
        std_pvalue=trends["std"][3],
        std_r2=trends["std"][4],
        gini_slope=trends["gini"][1],
        gini_se=trends["gini"][2],
        gini_pvalue=trends["gini"][3],
        gini_r2=trends["gini"][4],
        cv_slope=trends["cv"][1],
        cv_se=trends["cv"][2],
        cv_pvalue=trends["cv"][3],
        cv_r2=trends["cv"][4],
        notes=tuple(notes),
    )


# ============================== club convergence ===============================
# Phillips-Sul (2007/2009) log(t) test + data-driven club clustering, ported from the
# Stata `psecta` package (Du 2017) and its Mata source `lpsecta.do`. The numbers below
# mirror that source line-for-line so the clubs match the reference implementation.

_TCRIT = -1.65  # one-sided 5% critical value for the log(t) convergence test


def _sround(x: float) -> int:
    """Round half **away from zero**, matching Stata's ``round()`` (not banker's rounding).

    The log(t) trimming uses ``round(r*T)``; Python's built-in ``round`` rounds halves to even,
    which would silently shift the discarded fraction by one period at the half-way point.
    """
    return math.floor(x + 0.5) if x >= 0.0 else math.ceil(x - 0.5)


def _andrews_lrv(x: np.ndarray) -> float:
    """Long-run variance of a 1-D series via the Andrews (1991) quadratic-spectral HAC.

    A verbatim port of the Mata ``_andrs`` (itself a translation of Donggyu Sul's GAUSS code)
    used inside the Phillips-Sul log(t) test. The bandwidth is the AR(1)-based automatic choice
    ``band = 1.3221 (a2 * m)^(1/5)`` with ``a2 = 4 b1^2 / (1 - b1)^4``; the autocovariances run
    over the first ``m - 1`` terms and the variance is normalised by ``m - 1`` (both exactly as
    in the reference). Returns ``nan`` for a series too short or with no first-order variation.
    """
    v = np.asarray(x, dtype=float).ravel()
    m = v.size
    if m < 3:
        return float("nan")
    x1, y1 = v[:-1], v[1:]
    denom = float(np.dot(x1, x1))
    if denom <= 0.0:
        return float("nan")
    b1 = float(np.dot(x1, y1) / denom)  # AR(1) coefficient
    if b1 == 1.0:
        return float("nan")
    a2 = 4.0 * b1**2 / (1.0 - b1) ** 4
    band = 1.3221 * (a2 * m) ** 0.2
    if not math.isfinite(band) or band <= 0.0:
        return float("nan")
    t = m - 1
    j = np.arange(1, m, dtype=float)  # 1 .. m-1
    jb = j / band
    jband = jb * (1.2 * math.pi)
    # Quadratic-spectral kernel weights.
    kern = (np.sin(jband) / jband - np.cos(jband)) / ((jb * math.pi) ** 2 * 12.0) * 25.0
    lam = 0.0
    for i in range(1, t):  # i = 1 .. t-1
        c = float(np.dot(v[: t - i], v[i:t]))
        lam += 2.0 * c * kern[i - 1] / t
    sigm = float(np.dot(v, v)) / t
    return sigm + lam


def _log_t_test(mat: np.ndarray, r: float) -> tuple[float, float]:
    """Phillips-Sul log(t) convergence test on a units-by-time matrix.

    Forms the relative transition ``h_it = x_it / mean_i(x_it)`` and the cross-sectional
    variance ``H_t = mean_i (h_it - 1)^2``, then runs the regression

    ``log(H_1 / H_t) - 2 log(log t) = a + b log t + e``,    ``t = [rT] .. T``

    discarding the first ``round(r*T)`` periods. ``b = 2*alpha`` so a one-sided ``t_b > -1.65``
    fails to reject convergence. The standard error is the Phillips-Sul scalar-long-run-variance
    HAC ``V = (X'X)^{-1} * omega`` with ``omega`` from :func:`_andrews_lrv` on the demeaned
    residuals. Returns ``(b, t_b)``; either is ``nan`` when the test is not estimable. Port of
    the Mata ``_reglogt``.
    """
    m = np.asarray(mat, dtype=float)
    n_units, big_t = m.shape
    if n_units < 1 or big_t < 2:
        return float("nan"), float("nan")
    xcm = m.mean(axis=0)  # cross-sectional mean per period
    with np.errstate(divide="ignore", invalid="ignore"):
        h = m / xcm
        h_var = np.mean((h - 1.0) ** 2, axis=0)  # H_t
        logt = np.log(np.arange(1, big_t + 1, dtype=float))
        y = np.log(h_var[0] / h_var) - 2.0 * np.log(logt)
    start = _sround(r * big_t)  # discard the first `start` periods (0-based slice)
    if big_t - start < 4:
        return float("nan"), float("nan")
    design = np.column_stack([logt[start:], np.ones(big_t - start)])
    ys = y[start:]
    if not (np.all(np.isfinite(ys)) and np.all(np.isfinite(design))):
        return float("nan"), float("nan")
    xtx = design.T @ design
    try:
        b = np.linalg.solve(xtx, design.T @ ys)
        xtx_inv = np.linalg.inv(xtx)
    except np.linalg.LinAlgError:  # pragma: no cover - defensive
        return float("nan"), float("nan")
    resid = ys - design @ b
    resid = resid - resid.mean()
    omega = _andrews_lrv(resid)
    var0 = float(xtx_inv[0, 0]) * omega
    if not math.isfinite(var0) or var0 <= 0.0:
        return float(b[0]), float("nan")
    return float(b[0]), float(b[0] / math.sqrt(var0))


def _hp_trend(mat: np.ndarray, lamb: float) -> np.ndarray:
    """Return the Hodrick-Prescott **trend** of each row of ``mat`` (one unit per row).

    Mirrors the Stata ``pfilter ..., method(hp)`` step: the filter is applied to each unit's
    time series independently and the trend (not the cycle) is kept. Requires a gap-free series.
    """
    from statsmodels.tsa.filters.hp_filter import hpfilter

    out = np.empty_like(mat, dtype=float)
    for i in range(mat.shape[0]):
        _, trend = hpfilter(mat[i], lamb=lamb)
        out[i] = np.asarray(trend, dtype=float)
    return out


def _relative_transition(mat: np.ndarray) -> np.ndarray:
    """Return ``h_it = x_it / mean_i(x_it)`` (cross-sectional mean is 1 in every period)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return mat / mat.mean(axis=0)


def _sort_order(sub: np.ndarray, fr: float) -> np.ndarray:
    """Return indices sorting units by the cross-section criterion, **descending** (Step 1).

    ``fr == 0`` sorts by the last-period value; ``fr > 0`` sorts by the mean of the last
    ``(1 - fr)`` fraction of periods (the high-volatility option of the reference).
    """
    big_t = sub.shape[1]
    if fr <= 0.0:
        key = sub[:, -1]
    else:
        # Mata `_findclub` averages observation columns (trunc((1-fr)*(T-1))+2)..T of a matrix
        # whose first column is the id; here `sub` has no id column, so big_t == Mata's (T-1)
        # and the faithful 0-based start period is trunc((1-fr)*big_t).
        p_start = math.trunc((1.0 - fr) * big_t)
        key = sub[:, p_start:].mean(axis=1)
    return np.argsort(-key, kind="stable")


def _find_one_club(
    sub: np.ndarray,
    ids: np.ndarray,
    r: float,
    tcrit: float,
    cr: float,
    incr: float,
    max_cr: float,
    fr: float,
    adjust: bool,
) -> list[int]:
    """Find the highest-ranked convergence club in a subgroup (Steps 1-3 of Phillips-Sul).

    ``sub`` is the units-by-time matrix of the subgroup and ``ids`` the units' positional ids in
    the full panel. Returns the member ids (possibly empty when no club exists). Port of the
    Mata ``_findclub``: cross-section sort, core-group formation by maximum t-statistic, then a
    sieve of the complement — either the original PS-2007 ``cr``-increment rule or the
    Schnurbus et al. (2016) ``adjust`` refinement.
    """
    n_units = sub.shape[0]
    if n_units < 2:
        return []
    order = _sort_order(sub, fr)
    s = sub[order]
    sid = ids[order]

    # Step 2.1 - first successive pair (k, k+1) whose log(t) t-stat exceeds the threshold.
    tt = -100.0
    core_start = 0
    found = False
    while core_start < n_units - 1:
        _, tt = _log_t_test(s[core_start : core_start + 2], r)
        if math.isfinite(tt) and tt > tcrit:
            found = True
            break
        if not math.isfinite(tt):  # `.` in Mata stops the search (treated as failure)
            break
        core_start += 1
    if not found:
        return []

    # Step 2.2 - extend the core upward, keeping the prefix with the maximum t-statistic.
    ts_by_end: dict[int, float] = {}
    end = core_start + 1
    last_tt = tt
    while end <= n_units - 1 and last_tt > tcrit:
        _, last_tt = _log_t_test(s[core_start : end + 1], r)
        if not math.isfinite(last_tt):
            break
        ts_by_end[end] = last_tt
        end += 1
    core_end = (
        max(ts_by_end, key=lambda e: ts_by_end[e]) if ts_by_end else core_start + 1
    )
    core_pos = list(range(core_start, core_end + 1))
    core_set = set(core_pos)

    # Step 3.1/3.2 - sieve the complement, adding each unit whose core+unit t-stat exceeds cr.
    complement = [p for p in range(n_units) if p not in core_set]

    def club_tstat(positions: list[int]) -> float:
        _, t = _log_t_test(s[np.array(positions)], r)
        return t

    club_pos = list(core_pos)
    for p in complement:
        t = club_tstat([*core_pos, p])
        if math.isfinite(t) and t > cr:
            club_pos.append(p)

    # Step 3.3 - if the assembled club fails the joint test, refine it.
    club_t = club_tstat(club_pos)
    only_core = len(club_pos) == len(core_pos)
    if math.isfinite(club_t) and club_t <= tcrit and not only_core:
        if not adjust:  # original PS-2007: raise cr until the club converges
            cur_cr = cr
            while (not math.isfinite(club_t) or club_t <= tcrit) and cur_cr < max_cr:
                cur_cr += incr
                club_pos = list(core_pos)
                for p in complement:
                    t = club_tstat([*core_pos, p])
                    if math.isfinite(t) and t > cur_cr:
                        club_pos.append(p)
                club_t = club_tstat(club_pos)
            if not math.isfinite(club_t) or club_t <= tcrit:
                club_pos = list(core_pos)
        else:  # Schnurbus et al. (2016): add the best candidate one at a time
            candidates = [p for p in club_pos if p not in core_set]
            club_pos = list(core_pos)
            remaining = list(candidates)
            while remaining:
                # Score each candidate by the t-stat of the *growing* club plus that
                # candidate (not the core alone), so the stopping rule sees the club degrade
                # as members accumulate; stop before an addition would push it below tcrit.
                scored = [(club_tstat([*club_pos, p]), p) for p in remaining]
                best_t, best_p = max(scored, key=lambda it: it[0])
                if not math.isfinite(best_t) or best_t <= tcrit:
                    break
                club_pos.append(best_p)
                remaining = [p for p in candidates if p not in set(club_pos)]
            # Never return a club that fails its own joint test (as the PS-2007 branch does).
            final_t = club_tstat(club_pos)
            if not math.isfinite(final_t) or final_t <= tcrit:
                club_pos = list(core_pos)

    return [int(sid[p]) for p in club_pos]


def _get_clusters(
    mat: np.ndarray,
    r: float,
    tcrit: float,
    cr: float,
    incr: float,
    max_cr: float,
    fr: float,
    adjust: bool,
) -> dict[int, int]:
    """Recursively partition the panel into convergence clubs (Phillips-Sul Step 4).

    Returns a ``{unit_id: club}`` mapping with clubs numbered ``1..K`` from the highest-ranked
    group down; units left unassigned form the (divergent) residual group. Called only after
    the whole-panel log(t) test has rejected global convergence. Port of the Mata ``_getcluster``.
    """
    remaining = list(range(mat.shape[0]))
    club_of: dict[int, int] = {}
    club = 0
    while True:
        sub_ids = np.array(remaining)
        members = _find_one_club(
            mat[sub_ids], sub_ids, r, tcrit, cr, incr, max_cr, fr, adjust
        )
        if not members:
            break  # the remaining units do not form a further club (divergent)
        club += 1
        for cid in members:
            club_of[cid] = club
        member_set = set(members)
        remaining = [i for i in remaining if i not in member_set]
        if len(remaining) < 2:
            break
        # Does the whole remainder converge as a single final club?
        _, tt = _log_t_test(mat[np.array(remaining)], r)
        if math.isfinite(tt) and tt > tcrit:
            club += 1
            for cid in remaining:
                club_of[cid] = club
            break
    return club_of


def _merge_once(
    mat: np.ndarray, club_of: dict[int, int], r: float, tcrit: float
) -> tuple[dict[int, int], bool]:
    """One adjacent-club merging pass (Phillips-Sul 2009). Port of the Stata ``icheckmerge``.

    Walks the clubs in rank order, absorbing club ``k+1`` into the running merged block when
    their joint log(t) test converges, else starting a new block. Returns the relabelled
    ``{unit_id: club}`` and whether any merge happened.
    """
    members_by_club: dict[int, list[int]] = {}
    for cid, c in club_of.items():
        members_by_club.setdefault(c, []).append(cid)
    clubs = sorted(members_by_club)
    n_clubs = len(clubs)
    new_label = {clubs[0]: 1}
    running = list(members_by_club[clubs[0]])
    j = 1
    for k in range(1, n_clubs):
        cand = members_by_club[clubs[k]]
        _, tt = _log_t_test(mat[np.array(running + cand)], r)
        if math.isfinite(tt) and tt > tcrit:
            new_label[clubs[k]] = j
            running = running + cand
        else:
            j += 1
            new_label[clubs[k]] = j
            running = list(cand)
    return {cid: new_label[c] for cid, c in club_of.items()}, j < n_clubs


def _merge_clubs(
    mat: np.ndarray, club_of: dict[int, int], r: float, tcrit: float, mode: str
) -> dict[int, int]:
    """Merge adjacent clubs per ``mode`` (``"iterative"`` / ``"single"`` / ``"none"``)."""
    if mode == "none" or len(set(club_of.values())) < 2:
        return club_of
    if mode == "single":
        return _merge_once(mat, club_of, r, tcrit)[0]
    cur = club_of
    for _ in range(len(set(club_of.values()))):  # each pass drops >=1 club if it merges
        cur, merged = _merge_once(mat, cur, r, tcrit)
        if not merged:
            break
    return cur


def _club_color(club: int) -> str:
    """Palette color for a club label (1-based); the divergent group (0) renders grey."""
    return "#9AA0A6" if club == 0 else color_for(club - 1)


def _club_name(club: int) -> str:
    """Human label for a club number (0 is the non-converging 'Divergent' group)."""
    return "Divergent" if club == 0 else f"Club {club}"


def _clubs_long_frame(
    entities: np.ndarray,
    times: np.ndarray,
    trend: np.ndarray,
    relative: np.ndarray,
    club_of: dict[int, int],
    entity: str,
    time: str,
) -> pd.DataFrame:
    """Tidy long frame: one row per (unit, period) with ``value`` (trend), ``relative``, ``club``."""
    n_units, n_t = trend.shape
    rows = {
        entity: np.repeat(entities, n_t),
        time: np.tile(times, n_units),
        "value": trend.reshape(-1),
        "relative": relative.reshape(-1),
        "club": np.repeat([club_of.get(i, 0) for i in range(n_units)], n_t),
    }
    return pd.DataFrame(rows)


def _clubs_avg_fig(
    long: pd.DataFrame,
    entity: str,
    time: str,
    time_label: str,
    var_label: str,
    title: str | None,
) -> go.Figure:
    """Within-club **average** relative-transition paths (the headline figure)."""
    fig = go.Figure()
    for club in sorted(long["club"].unique()):
        sub = long[long["club"] == club]
        avg = sub.groupby(time, observed=True)["relative"].mean().sort_index()
        n_members = sub[entity].nunique()
        fig.add_trace(
            go.Scatter(
                x=avg.index.to_numpy(dtype=float),
                y=avg.to_numpy(dtype=float),
                mode="lines+markers",
                name=f"{_club_name(int(club))} (n={n_members})",
                line={
                    "color": _club_color(int(club)),
                    "width": 2.5,
                    "dash": "dot" if club == 0 else "solid",
                },
                marker={"color": _club_color(int(club)), "size": 6},
                hovertemplate=(
                    f"{_club_name(int(club))}<br>{time_label}=%{{x}}<br>"
                    "relative=%{y:.3f}<extra></extra>"
                ),
            )
        )
    fig.add_hline(y=1.0, line_dash="dash", line_color="rgba(0,0,0,0.4)")
    apply_default_layout(
        fig,
        xaxis={"title": time_label},
        yaxis={"title": f"Relative {var_label} (cross-sectional mean = 1)"},
    )
    fig.update_layout(title=title or f"Convergence clubs: {var_label}")
    return fig


def _clubs_paths_fig(
    long: pd.DataFrame,
    entity: str,
    time: str,
    time_label: str,
    var_label: str,
) -> go.Figure:
    """All units' relative-transition paths, coloured by club (one legend entry per club)."""
    fig = go.Figure()
    seen: set[int] = set()
    for ent, sub in long.groupby(entity, observed=True, sort=False):
        sub = sub.sort_values(time)
        club = int(sub["club"].iloc[0])
        fig.add_trace(
            go.Scatter(
                x=sub[time].to_numpy(dtype=float),
                y=sub["relative"].to_numpy(dtype=float),
                mode="lines",
                line={"color": _club_color(club), "width": 1},
                opacity=0.55,
                legendgroup=_club_name(club),
                name=_club_name(club),
                showlegend=club not in seen,
                customdata=np.full(len(sub), str(ent)),
                hovertemplate=(
                    f"%{{customdata}} ({_club_name(club)})<br>{time_label}=%{{x}}<br>"
                    "relative=%{y:.3f}<extra></extra>"
                ),
            )
        )
        seen.add(club)
    fig.add_hline(y=1.0, line_dash="dash", line_color="rgba(0,0,0,0.4)")
    apply_default_layout(
        fig,
        xaxis={"title": time_label},
        yaxis={"title": f"Relative {var_label} (cross-sectional mean = 1)"},
    )
    fig.update_layout(title=f"Relative transition paths by club: {var_label}")
    return fig


def _clubs_facets_fig(
    long: pd.DataFrame,
    entity: str,
    time: str,
    time_label: str,
    var_label: str,
) -> go.Figure:
    """Small-multiple panels (one per club) of member paths with the club mean overlaid."""
    clubs = sorted(int(c) for c in long["club"].unique())
    n = len(clubs)
    ncols = min(3, n)
    nrows = math.ceil(n / ncols)
    titles = []
    for c in clubs:
        members = long[long["club"] == c][entity].nunique()
        titles.append(f"{_club_name(c)} (n={members})")
    fig = make_subplots(
        rows=nrows, cols=ncols, subplot_titles=titles, shared_yaxes=True
    )
    for idx, club in enumerate(clubs):
        row, col = idx // ncols + 1, idx % ncols + 1
        sub = long[long["club"] == club]
        for _ent, g in sub.groupby(entity, observed=True, sort=False):
            g = g.sort_values(time)
            fig.add_trace(
                go.Scatter(
                    x=g[time].to_numpy(dtype=float),
                    y=g["relative"].to_numpy(dtype=float),
                    mode="lines",
                    line={"color": _club_color(club), "width": 0.8},
                    opacity=0.4,
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=row,
                col=col,
            )
        avg = sub.groupby(time, observed=True)["relative"].mean().sort_index()
        fig.add_trace(
            go.Scatter(
                x=avg.index.to_numpy(dtype=float),
                y=avg.to_numpy(dtype=float),
                mode="lines",
                line={"color": _club_color(club), "width": 2.5},
                showlegend=False,
                hovertemplate="relative=%{y:.3f}<extra></extra>",
            ),
            row=row,
            col=col,
        )
    apply_default_layout(fig)
    fig.update_layout(title=f"Convergence clubs ({var_label}): member paths by club")
    fig.update_xaxes(title_text=time_label, row=nrows)
    return fig


def _clubs_summary_and_gt(
    club_of: dict[int, int],
    club_stats: dict[int, tuple[float, float, int]],
    entities: np.ndarray,
    var_label: str,
    n_units: int,
    n_periods: int,
    tcrit: float,
) -> tuple[pd.DataFrame, Any, pd.DataFrame]:
    """Build the per-club ``summary`` frame, its Great-Tables rendering, and the membership frame.

    ``club_stats`` maps a club number to ``(beta, tstat, n_members)``; club ``0`` (if present)
    is the divergent residual group, listed last.
    """
    from great_tables import GT

    members_by_club: dict[int, list[str]] = {}
    for cid, c in sorted(club_of.items()):
        members_by_club.setdefault(c, []).append(str(entities[cid]))
    for cid in range(n_units):  # never-assigned units are the divergent group (club 0)
        if cid not in club_of:
            members_by_club.setdefault(0, []).append(str(entities[cid]))

    order = [c for c in sorted(members_by_club) if c != 0] + (
        [0] if 0 in members_by_club else []
    )

    def member_str(names: list[str], limit: int = 8) -> str:
        names = sorted(names)
        if len(names) <= limit:
            return ", ".join(names)
        return ", ".join(names[:limit]) + f", ... (+{len(names) - limit})"

    summary = pd.DataFrame(
        {
            "club": [_club_name(c) for c in order],
            "n_members": [len(members_by_club[c]) for c in order],
            "beta": [
                club_stats.get(c, (float("nan"), float("nan"), 0))[0] for c in order
            ],
            "tstat": [
                club_stats.get(c, (float("nan"), float("nan"), 0))[1] for c in order
            ],
            "converging": [
                bool(club_stats.get(c, (float("nan"), float("nan"), 0))[1] > tcrit)
                if c != 0
                else False
                for c in order
            ],
            "members": [member_str(members_by_club[c]) for c in order],
        }
    )

    def fmt(value: float) -> str:
        return "—" if not math.isfinite(value) else f"{value:.3f}"

    disp = pd.DataFrame(
        {
            "Group": summary["club"],
            "N": summary["n_members"],
            "log(t) b": [fmt(v) for v in summary["beta"]],
            "t-stat": [fmt(v) for v in summary["tstat"]],
            "Converges": [
                "—" if g == "Divergent" else ("yes" if c else "no")
                for g, c in zip(summary["club"], summary["converging"], strict=True)
            ],
            "Members": summary["members"],
        }
    )
    gt = (
        GT(disp, rowname_col="Group")
        .tab_header(
            title=f"Convergence clubs: {var_label}",
            subtitle=f"Phillips-Sul log(t) clustering over {n_periods} periods, "
            f"{n_units} units",
        )
        .tab_source_note(
            f"Each club's log(t) t-stat exceeds {tcrit:g} (the convergence threshold); "
            "b = 2*alpha is the within-club convergence speed. The Divergent group does not "
            "form a convergence club."
        )
    )

    membership = pd.DataFrame(
        {
            "entity": [str(entities[i]) for i in range(n_units)],
            "club": [club_of.get(i, 0) for i in range(n_units)],
        }
    )
    membership["club_label"] = membership["club"].map(_club_name)
    membership = membership.sort_values(["club", "entity"]).reset_index(drop=True)
    return summary, gt, membership


def analyze_convergence_clubs(
    df: pd.DataFrame,
    var: str,
    *,
    entity: str | None = None,
    time: str | None = None,
    filter: Literal["hp"] | None = "hp",
    hp_lambda: float = 400.0,
    r: float = 0.3,
    method: Literal["adjust", "ps"] = "adjust",
    merge: Literal["iterative", "single", "none"] = "iterative",
    cr: float = 0.0,
    incr: float = 0.05,
    max_cr: float = 50.0,
    fr: float = 0.0,
    tcrit: float = _TCRIT,
    title: str | None = None,
) -> ConvergenceClubsResult:
    r"""Phillips-Sul log(t) convergence test and data-driven club clustering for a panel.

    Runs the full club-convergence workflow on one variable: optionally smooth each unit's
    series with the **Hodrick-Prescott filter** (``lambda = 400`` for annual data); form the
    **relative transition path** ``h_it = x_it / mean_i(x_it)``; run the **log(t) regression
    test** for the whole panel; and, when global convergence is rejected, apply the
    **clustering algorithm** to split the units into convergence **clubs**, then **merge**
    adjacent clubs that jointly converge. This is the descriptive question "do these units form
    one converging group, several catch-up clubs, or none?".

    The variable is used **as supplied** — no log is taken — so for the canonical income case
    pass *log* GDP per capita (or log labor productivity). The panel must be **balanced** (every
    unit present in every period) because the HP filter needs a gap-free series.

    Parameters
    ----------
    df
        Balanced panel data frame.
    var
        Numeric variable to analyse (e.g. ``"log_gdppc"``). Used as supplied.
    entity, time
        Panel identifiers. Default to those declared via :func:`expdpy.set_panel`.
    filter
        ``"hp"`` (default) applies the Hodrick-Prescott filter per unit and analyses the
        **trend**; ``None`` analyses the variable as given (already detrended).
    hp_lambda
        HP smoothing parameter (``400`` for annual data, the convergence-literature default).
    r
        Initiating sample fraction for the log(t) regression: the first ``round(r*T)`` periods
        are discarded. Phillips-Sul recommend ``0.3`` for small/moderate ``T`` and ``0.2`` for
        large ``T``.
    method
        Within-club sieve: ``"adjust"`` (default) is the Schnurbus et al. (2016) refinement
        (add the best candidate one at a time); ``"ps"`` is the original Phillips-Sul (2007)
        rule that raises the inclusion threshold ``cr`` by ``incr`` until the club converges.
    merge
        Adjacent-club merging after clustering: ``"iterative"`` (default) repeats until no clubs
        merge, ``"single"`` does one pass, ``"none"`` reports the raw clusters.
    cr, incr, max_cr
        Sieve threshold and (for ``method="ps"``) its increment and ceiling.
    fr
        Cross-section sort key: ``0`` (default) sorts by the last period; ``fr > 0`` sorts by
        the mean of the last ``(1 - fr)`` fraction of periods (for noisy endpoints).
    tcrit
        One-sided convergence critical value for the t-statistic (``-1.65``, the 5% level).
    title
        Title for the headline figure.

    Returns
    -------
    ConvergenceClubsResult
        The tidy long ``df`` (``entity``, ``time``, ``value`` = trend, ``relative`` = ``h_it``,
        ``club``); the within-club average figure ``fig``; the all-paths figure ``fig_paths``
        and the per-club small-multiples ``fig_clubs``; the classification table ``gt`` /
        ``summary`` and the ``membership`` frame; the panel dimensions; the whole-panel
        ``global_beta`` / ``global_tstat`` and ``converged`` flag; and ``n_clubs`` /
        ``n_divergent``. ``.interpret()`` describes how the panel splits into clubs.

    Notes
    -----
    The log(t) test regresses, for ``t = [rT] .. T``,

    .. math:: \log(H_1 / H_t) - 2 \log(\log t) = a + b \log t + \varepsilon_t,

    where ``H_t = N^{-1} \sum_i (h_{it} - 1)^2`` is the cross-sectional variance of the relative
    transition paths. Under the null of convergence ``b = 2\alpha \ge 0``; a one-sided
    ``t_b > -1.65`` fails to reject it. The standard error is the Phillips-Sul scalar long-run
    variance form ``\hat{\mathrm{var}}(b) = (X'X)^{-1}\hat\omega`` with ``\hat\omega`` an
    Andrews (1991) quadratic-spectral HAC of the residuals. The clustering sorts units by their
    final value, forms a core group by maximising ``t_b``, sieves in the remaining units, and
    recurses on the residual; adjacent clubs are then merged when they jointly converge. This is
    a faithful port of the Stata ``psecta`` package (Du 2017); see Phillips & Sul (2007, 2009)
    and Schnurbus et al. (2016).

    Examples
    --------
    Convergence clubs in (log) GDP per capita across countries:

    ```python
    import expdpy as ex
    from expdpy.data import load_productivity

    df = load_productivity()
    res = ex.analyze_convergence_clubs(df, "log_gdppc", entity="country", time="year")
    res.fig            # within-club average transition paths
    res.gt             # club classification table
    res.n_clubs, res.converged
    print(res.interpret())
    ```
    """
    df = ensure_dataframe(df)
    entity, time = resolve_panel(
        df, entity, time, require_entity=True, require_time=True
    )
    assert entity is not None and time is not None  # guaranteed by require_* above

    if var not in df.columns:
        raise KeyError(f"column not found in df: {var!r}")
    if not pdt.is_numeric_dtype(df[var]):
        raise TypeError(f"var {var!r} must be numeric")
    if not 0.0 < r < 1.0:
        raise ValueError(f"r (trimming fraction) must be in (0, 1); got {r}")

    var_label = resolve_label(df, var)
    time_label = resolve_label(df, time)
    notes: list[str] = []

    work = df[[entity, time, var]].copy()
    work[time] = pd.to_numeric(work[time], errors="coerce")
    work = work.dropna(subset=[time, var])
    if work.empty:
        raise ValueError(
            f"no rows with both a numeric {time!r} and a non-missing {var!r}"
        )

    before = len(work)
    work = work.groupby([entity, time], observed=True, as_index=False).first()
    if len(work) < before:
        notes.append(
            f"found duplicate (entity, time) rows; kept the first of each "
            f"({before - len(work)} dropped)"
        )

    wide = work.pivot(index=entity, columns=time, values=var).sort_index(axis=1)
    if bool(wide.isna().to_numpy().any()):
        n_missing = int(wide.isna().to_numpy().sum())
        raise ValueError(
            f"panel is not balanced: {n_missing} (entity, time) cells are missing. Club "
            "convergence needs a gap-free series per unit (the HP filter cannot span gaps); "
            "restrict to a balanced window or drop the offending units."
        )
    entities = wide.index.to_numpy()
    times = wide.columns.to_numpy(dtype=float)
    n_units, n_periods = wide.shape
    if n_units < 2:
        raise ValueError(f"need >= 2 units to form convergence clubs; got {n_units}")
    if n_periods - _sround(r * n_periods) < 4:
        raise ValueError(
            f"too few periods ({n_periods}) for a log(t) test trimmed at r={r}: only "
            f"{n_periods - _sround(r * n_periods)} remain after discarding the first "
            f"{_sround(r * n_periods)}; need >= 4. Use more periods or a smaller r."
        )

    raw = wide.to_numpy(dtype=float)
    trend = _hp_trend(raw, hp_lambda) if filter == "hp" else raw

    # The relative transition divides by the per-period cross-sectional mean, so a mean at or
    # near zero (e.g. a demeaned / centered / growth variable) blows it up to inf and silently
    # corrupts every downstream frame and figure. Phillips-Sul expects a strictly-positive
    # variable (levels or log-levels); reject a (near-)zero or sign-changing mean up front.
    xcm = trend.mean(axis=0)
    scale = float(np.nanmax(np.abs(trend))) if trend.size else 0.0
    if not np.all(np.isfinite(xcm)) or bool(
        np.any(np.abs(xcm) <= 1e-9 * (1.0 + scale))
    ):
        raise ValueError(
            f"the per-period cross-sectional mean of {var!r} is at or near zero in some "
            "period, so the relative transition h_it = x_it / mean_i(x_it) is undefined. "
            "Club convergence expects a strictly-positive variable (levels or log-levels), "
            "not a demeaned/centered or sign-changing series."
        )
    relative = _relative_transition(trend)

    global_beta, global_tstat = _log_t_test(trend, r)
    if not math.isfinite(global_tstat):
        raise ValueError(
            f"the log(t) convergence test for {var!r} is not estimable: the cross-sectional "
            "dispersion of the relative transitions is (near) zero in every period — the units "
            "are already identical (trivially converged), so the test statistic is undefined."
        )
    converged = bool(global_tstat > tcrit)

    if converged:
        club_of = {i: 1 for i in range(n_units)}
    else:
        club_of = _get_clusters(
            trend, r, tcrit, cr, incr, max_cr, fr, method == "adjust"
        )
        club_of = _merge_clubs(trend, club_of, r, tcrit, merge)

    # Per-club log(t) statistics (and the divergent group, if any has >= 2 members).
    club_stats: dict[int, tuple[float, float, int]] = {}
    members_by_club: dict[int, list[int]] = {}
    for cid, c in club_of.items():
        members_by_club.setdefault(c, []).append(cid)
    divergent_ids = [i for i in range(n_units) if i not in club_of]
    n_clubs = len(members_by_club)
    for c, ids in members_by_club.items():
        b, t = _log_t_test(trend[np.array(ids)], r)
        club_stats[c] = (b, t, len(ids))
    if len(divergent_ids) >= 2:
        b, t = _log_t_test(trend[np.array(divergent_ids)], r)
        club_stats[0] = (b, t, len(divergent_ids))
    elif len(divergent_ids) == 1:
        club_stats[0] = (float("nan"), float("nan"), 1)

    long = _clubs_long_frame(entities, times, trend, relative, club_of, entity, time)
    summary, gt, membership = _clubs_summary_and_gt(
        club_of, club_stats, entities, var_label, n_units, n_periods, tcrit
    )

    fig = _clubs_avg_fig(long, entity, time, time_label, var_label, title)
    fig_paths = _clubs_paths_fig(long, entity, time, time_label, var_label)
    fig_clubs = _clubs_facets_fig(long, entity, time, time_label, var_label)

    if converged:
        notes.append(
            f"the whole panel converges (global log(t) t-stat > {tcrit:g}); "
            "it forms a single club"
        )
    elif n_clubs == 0:
        notes.append(
            "global convergence is rejected and no convergence clubs were found; all units "
            "diverge"
        )

    return ConvergenceClubsResult(
        df=long,
        fig=fig,
        fig_paths=fig_paths,
        fig_clubs=fig_clubs,
        gt=gt,
        summary=summary,
        membership=membership,
        var=var,
        entity=entity,
        time=time,
        n_units=n_units,
        n_periods=n_periods,
        n_clubs=n_clubs,
        n_divergent=len(divergent_ids),
        global_beta=global_beta,
        global_tstat=global_tstat,
        converged=converged,
        hp_lambda=float(hp_lambda) if filter == "hp" else float("nan"),
        trim=float(r),
        tcrit=float(tcrit),
        method=method,
        merge=merge,
        notes=tuple(notes),
    )
