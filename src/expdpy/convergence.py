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
from expdpy._types import BetaConvergenceResult, SigmaConvergenceResult
from expdpy._validation import ensure_dataframe
from expdpy.fwl import _residualize
from expdpy.regression import _SSC, _as_list

__all__ = ["analyze_beta_convergence", "analyze_sigma_convergence"]

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
        df = df.drop_duplicates([entity, time], keep="first")
        notes.append(
            f"found duplicate (entity, time) rows; kept the first of each ({n_dup} dropped)"
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
