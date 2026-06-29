"""The extended Kuznets curve ("Kuznets waves") for panel data.

:func:`analyze_kuznets_waves` tests whether inequality and development trace the classic
inverted-U or a richer, highly nonlinear **wave**. It estimates the polynomial relationship

    inequality = b_1 g + b_2 g^2 + ... + b_degree g^degree   (g = log GDP per capita)

under three panel estimators and lays them side by side: **pooled OLS** (all variation),
the **between** estimator (country averages — the cross-country curve) and the **within**
estimator (two-way country + year fixed effects — within-country variation net of common year
shocks). Each is reported as a nested-specification comparison table (the linear model, then
the quadratic, up to the full ``degree``-order polynomial), built with cumulative-stepwise
estimation through :func:`expdpy.analyze_estimation` (pyfixest).

Three figures tell the pooled -> between -> within story. The first is the **raw** scatter of
inequality against development with the pooled polynomial overlaid. The second and third are
**partial-residual (component) plots**: the fitted polynomial wave drawn over inequality once
the optional ``controls`` (and, for the within view, the two-way fixed effects) have been
partialled out via the Frisch-Waugh-Lovell theorem, so the wave is read on the development axis
itself. The development variable is used **as supplied** (pass *log* GDP per capita); the
polynomial powers are formed internally.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pandas.api import types as pdt

from expdpy._common import entity_display_map as _entity_display_map
from expdpy._labels import resolve_label
from expdpy._panel import resolve_entity_name, resolve_panel
from expdpy._theme import apply_default_layout, color_for
from expdpy._types import KuznetsWavesResult
from expdpy._validation import ensure_dataframe
from expdpy.estimation import analyze_estimation
from expdpy.fwl import _residualize
from expdpy.regression import _as_list

__all__ = ["analyze_kuznets_waves"]

_MIN_DEGREE = 2
_MAX_DEGREE = 6


def _poly_terms(development: str, degree: int) -> list[str]:
    """Return the regressor names for ``g^1 .. g^degree`` (the linear term reuses ``g``).

    The linear term is the development column itself; higher powers get ``_p<k>`` suffixes
    (e.g. ``log_gdp_pc``, ``log_gdp_pc_p2``, ``log_gdp_pc_p3``, ``log_gdp_pc_p4``).
    """
    return [development] + [f"{development}_p{k}" for k in range(2, degree + 1)]


def _add_powers(frame: pd.DataFrame, development: str, degree: int) -> list[str]:
    """Add ``g^2 .. g^degree`` columns to ``frame`` in place and return all ``degree`` terms."""
    terms = _poly_terms(development, degree)
    g = frame[development].to_numpy(dtype=float)
    for k, name in zip(range(2, degree + 1), terms[1:], strict=True):
        frame[name] = g**k
    return terms


def _eval_poly(betas: Sequence[float], g: np.ndarray) -> np.ndarray:
    """Evaluate ``sum_k betas[k-1] * g^k`` for ``k = 1 .. len(betas)`` (no intercept)."""
    gg = np.asarray(g, dtype=float)
    out = np.zeros_like(gg)
    for k, b in enumerate(betas, start=1):
        out = out + float(b) * gg**k
    return out


def _poly_betas(model: Any, terms: Sequence[str]) -> list[float]:
    """Extract the polynomial coefficients ``b_1 .. b_degree`` (0.0 for any dropped term)."""
    coef = model.coef()
    return [float(coef[t]) if t in coef.index else 0.0 for t in terms]


def _turning_points(
    betas: Sequence[float], g_lo: float, g_hi: float
) -> list[dict[str, Any]]:
    """Real turning points of ``f(g) = sum_k betas[k-1] g^k`` inside ``[g_lo, g_hi]``.

    Solves ``f'(g) = 0`` and classifies each in-range real root by the sign of ``f''`` as a
    ``"peak"`` (local maximum), ``"trough"`` (local minimum) or ``"inflection"``. Returns an
    empty list when the derivative is constant (a linear fit) or has no real root in range.
    """
    b = np.asarray(betas, dtype=float)
    d = b.size
    if d < 2:
        return []
    # f'(g) = sum_{k=1..d} k b_k g^{k-1}; coefficient of g^{k-1} is k*b_k (ascending order).
    deriv_asc = np.array([k * b[k - 1] for k in range(1, d + 1)], dtype=float)
    deriv_desc = deriv_asc[::-1]  # np.roots wants the highest power first
    nz = np.flatnonzero(np.abs(deriv_desc) > 1e-15)
    if nz.size == 0:
        return []
    deriv_desc = deriv_desc[nz[0] :]
    if deriv_desc.size < 2:  # derivative is a non-zero constant -> no turning point
        return []
    out: list[dict[str, Any]] = []
    for root in np.roots(deriv_desc):
        if abs(root.imag) > 1e-7 * (1.0 + abs(root.real)):
            continue
        g_star = float(root.real)
        if not (g_lo <= g_star <= g_hi):
            continue
        f2 = sum(k * (k - 1) * b[k - 1] * g_star ** (k - 2) for k in range(2, d + 1))
        kind = "peak" if f2 < 0 else "trough" if f2 > 0 else "inflection"
        out.append({"g": g_star, "kind": kind})
    out.sort(key=lambda item: item["g"])
    return out


def _shape_lines(
    tps: list[dict[str, Any]], dev_label: str, n_obs: int, r2: float, r2_label: str
) -> list[str]:
    """Build the annotation-box lines: the turning-point shape, N and R²."""
    if not tps:
        shape = "monotonic (no turning point in range)"
    else:
        bits = [f"{tp['kind']} at {dev_label} = {tp['g']:.3g}" for tp in tps]
        shape = f"{len(tps)} turning point(s): " + "; ".join(bits)
    return [shape, f"N = {n_obs:,}", f"{r2_label} = {r2:.3f}"]


def _wave_fig(
    x: np.ndarray,
    y: np.ndarray,
    entities: np.ndarray,
    grid: np.ndarray,
    curve: np.ndarray,
    x_label: str,
    y_label: str,
    ann_lines: list[str],
    title: str | None,
    *,
    n_sample: int | None,
    seed: int,
) -> go.Figure:
    """Assemble a Kuznets-wave figure: fitted curve + (subsampled) points + annotation box."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=grid,
            y=curve,
            mode="lines",
            line={"color": color_for(2), "width": 2.5},
            name="fit",
            showlegend=False,
            hoverinfo="skip",
        )
    )
    idx = np.arange(len(x))
    if n_sample is not None and len(x) > n_sample:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(x), size=n_sample, replace=False))
    fig.add_trace(
        go.Scatter(
            x=x[idx],
            y=y[idx],
            mode="markers",
            marker={
                "color": color_for(0),
                "size": 7,
                "opacity": 0.6,
                "line": {"color": "white", "width": 0.4},
            },
            customdata=entities[idx],
            hovertemplate=(
                "%{customdata}<br>" + x_label + "=%{x:.4g}<br>"
                "" + y_label + "=%{y:.4g}<extra></extra>"
            ),
            name="points",
            showlegend=False,
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
        bordercolor="rgba(0,0,0,0.2)",
        borderwidth=1,
        bgcolor="rgba(255,255,255,0.7)",
        text="<br>".join(ann_lines),
    )
    apply_default_layout(fig, xaxis={"title": x_label}, yaxis={"title": y_label})
    if title:
        fig.update_layout(title=title)
    return fig


def _estimator_summary(
    name: str,
    full_model: Any,
    terms: Sequence[str],
    g_lo: float,
    g_hi: float,
    n_obs: int,
    r2: float,
) -> dict[str, Any]:
    """One summary row for an estimator: curvature (turning points) and the top-order term."""
    betas = _poly_betas(full_model, terms)
    tps = _turning_points(betas, g_lo, g_hi)
    peak = next((tp["g"] for tp in tps if tp["kind"] == "peak"), float("nan"))
    top = terms[-1]
    coef = full_model.coef()
    pval = full_model.pvalue()
    return {
        "estimator": name,
        "n_obs": int(n_obs),
        "r2": float(r2),
        "n_turning_points": len(tps),
        "peak_g": float(peak),
        "top_term": top,
        "top_estimate": float(coef[top]) if top in coef.index else float("nan"),
        "top_pvalue": float(pval[top]) if top in pval.index else float("nan"),
    }


def analyze_kuznets_waves(
    df: pd.DataFrame,
    inequality: str = "gini_regional",
    development: str = "log_gdp_pc",
    controls: Sequence[str] | str | None = None,
    *,
    entity: str | None = None,
    time: str | None = None,
    degree: int = 4,
    vcov: Literal["hetero", "iid"] = "hetero",
    n_sample: int | None = 1000,
    seed: int = 0,
    title: str | None = None,
) -> KuznetsWavesResult:
    r"""Estimate the extended Kuznets curve ("Kuznets waves") across three panel estimators.

    Fits ``inequality = b_1 g + b_2 g^2 + ... + b_degree g^degree`` (with ``g`` the
    ``development`` variable, used **as supplied** — pass *log* GDP per capita) under **pooled
    OLS**, the **between** estimator (one observation per entity, the polynomial formed from the
    entity means) and the **within** estimator (two-way ``entity`` + ``time`` fixed effects).
    Each estimator is reported as a cumulative-stepwise comparison table — the linear model, then
    the quadratic, up to the full ``degree``-order polynomial — so the curvature can be read as
    higher-order terms are added. The relationship is **associational**, not causal.

    Three figures accompany the tables. The first is the raw scatter of ``inequality`` against
    ``g`` with the pooled polynomial overlaid. The second (between) and third (within) are
    **partial-residual / component plots**: the fitted wave drawn over ``inequality`` after the
    optional ``controls`` — and, for the within view, the two-way fixed effects — are partialled
    out by the Frisch-Waugh-Lovell theorem, so the wave is read on the development axis itself.

    Parameters
    ----------
    df
        Panel data frame.
    inequality
        Numeric outcome (an inequality measure such as a Gini). Default ``"gini_regional"``.
    development
        Numeric development regressor, used as supplied (typically *log* GDP per capita); its
        powers ``g^2 .. g^degree`` are formed internally. Default ``"log_gdp_pc"``.
    controls
        Optional covariate name(s) partialled out of the **between** and **within** figures via
        the Frisch-Waugh-Lovell theorem. They do **not** enter the comparison tables, which are
        pure polynomial buildups. ``None`` (default) partials out only the fixed effects (in the
        within figure).
    entity, time
        Panel identifiers. Default to those declared via :func:`expdpy.set_panel`.
    degree
        Polynomial order in ``[2, 6]`` (default 4, the quartic "waves" specification). The tables
        show ``degree`` nested columns; ``degree=2`` recovers the classic inverted-U test.
    vcov
        Standard-error type for the **pooled** and **between** tables: ``"hetero"`` (HC1, the
        default) or ``"iid"``. The **within** table always uses standard errors clustered by
        ``entity``. Does not change any point estimate.
    n_sample
        Number of points drawn in the raw and within scatters (default 1000); the fitted curves
        always use every row. ``None`` plots every point. The between scatter (one point per
        entity) is never subsampled.
    seed
        Seed for the point-subsampling RNG (default 0), for reproducible figures.
    title
        Title for the raw figure (the between/within figures get descriptive titles).

    Returns
    -------
    KuznetsWavesResult
        The stacked tidy coefficient frame ``df``; the raw scatter ``fig`` and the between /
        within partial-residual figures ``fig_between`` / ``fig_within``; the three comparison
        tables ``gt_pooled`` / ``gt_between`` / ``gt_within``; the per-estimator curvature
        ``summary``; the fitted ``models`` (keyed by estimator); and the run metadata
        (``inequality``, ``development``, ``controls``, ``degree``, ``entity``, ``time``,
        ``n_obs``). Use ``.interpret()`` / ``.explain()`` for plain-language output.

    Notes
    -----
    The classic Kuznets (1955) hypothesis is the quadratic inverted-U; the **waves** extension
    raises the development term to the quartic, admitting up to three turning points. By the
    Frisch-Waugh-Lovell theorem the polynomial coefficients are identical whether the controls
    (and fixed effects) are included directly or first partialled out of both sides, which is what
    the component plots exploit: the fitted curve is the within/between polynomial, drawn over the
    residualized inequality. The between estimator forms the polynomial from each entity's **mean**
    development, so it is the cross-country curve; the within estimator uses two-way fixed effects,
    so it is identified purely from within-entity, within-year movements.

    Examples
    --------
    The headline Kuznets-waves analysis on the bundled dataset:

    ```python
    import expdpy as ex
    from expdpy.data import load_kuznets, load_kuznets_data_def

    df = ex.set_labels(load_kuznets(), load_kuznets_data_def(), set_panel=True)
    res = ex.analyze_kuznets_waves(df)
    res.gt_within        # the within (two-way FE) comparison table
    res.fig_within       # the within partial-residual wave
    print(res.interpret())
    ```

    Partialling covariates out of the between and within waves, with a cubic specification:

    ```python
    import expdpy as ex
    from expdpy.data import load_kuznets, load_kuznets_data_def

    df = ex.set_labels(load_kuznets(), load_kuznets_data_def(), set_panel=True)
    ex.analyze_kuznets_waves(
        df, controls=["trade_share", "resource_rents"], degree=3
    ).fig_between
    ```
    """
    df = ensure_dataframe(df)
    controls = _as_list(controls)
    entity, time = resolve_panel(
        df, entity, time, require_entity=True, require_time=True
    )
    assert entity is not None and time is not None  # guaranteed by require_* above

    if not isinstance(degree, int) or isinstance(degree, bool):
        raise TypeError(f"degree must be an int; got {type(degree).__name__}")
    if not (_MIN_DEGREE <= degree <= _MAX_DEGREE):
        raise ValueError(
            f"degree must be in [{_MIN_DEGREE}, {_MAX_DEGREE}]; got {degree}"
        )
    if development in controls:
        raise ValueError(f"development {development!r} must not also be a control")
    if inequality in controls:
        raise ValueError(f"inequality {inequality!r} must not also be a control")
    if inequality == development:
        raise ValueError("inequality and development must be different columns")

    needed = list(dict.fromkeys([entity, time, inequality, development, *controls]))
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"columns not found in df: {missing}")
    for col in [inequality, development, *controls]:
        if not pdt.is_numeric_dtype(df[col]):
            raise TypeError(f"column {col!r} must be numeric")

    ineq_label = resolve_label(df, inequality)
    dev_label = resolve_label(df, development)
    # Entity-name display map ("Name (id)"), built before the dedup groupby drops attrs.
    ent_disp = _entity_display_map(df, entity, resolve_entity_name(df))
    notes: list[str] = []

    n_dup = int(df.duplicated([entity, time]).sum())
    if n_dup:
        df = df.groupby([entity, time], observed=True, as_index=False).first()
        notes.append(
            f"found duplicate (entity, time) rows; kept the first non-missing of each "
            f"({n_dup} dropped)"
        )

    work = df[needed].dropna().copy()
    work[entity] = work[entity].astype(str)
    min_rows = max(degree + len(controls) + 2, 10)
    if len(work) < min_rows:
        raise ValueError(
            f"too few complete-case rows ({len(work)}); need >= {min_rows}. Check the "
            "columns for missing values or lower the polynomial degree."
        )
    g = work[development].to_numpy(dtype=float)
    if np.ptp(g) <= 1e-10 * (1.0 + float(np.abs(g).max())):
        raise ValueError(
            f"development {development!r} has (near) zero variance; the Kuznets curve is "
            "not identified"
        )

    terms = _add_powers(work, development, degree)

    # --- Between estimator: collapse to entity means, form the polynomial from the means ---
    agg = {c: "mean" for c in [inequality, development, *controls]}
    between_df = work.groupby(entity, observed=True, as_index=False).agg(agg)
    if len(between_df) < degree + 2:
        raise ValueError(
            f"too few entities ({len(between_df)}) for a degree-{degree} between curve; "
            f"need >= {degree + 2}"
        )
    _add_powers(between_df, development, degree)
    gbar = between_df[development].to_numpy(dtype=float)

    # --- Three comparison tables (pure polynomial buildup, cumulative-stepwise) ---
    pooled = analyze_estimation(work, inequality, idvs=terms, stepwise="csw", vcov=vcov)
    between = analyze_estimation(
        between_df, inequality, idvs=terms, stepwise="csw", vcov=vcov
    )
    within = analyze_estimation(
        work,
        inequality,
        idvs=terms,
        stepwise="csw",
        feffects=[entity, time],
        cluster=entity,
    )
    models = {
        "pooled": pooled.models,
        "between": between.models,
        "within": within.models,
    }

    # --- Tidy stacked coefficient frame ---
    def _tag(res: Any, name: str) -> pd.DataFrame:
        out = res.df.copy()
        out.insert(0, "estimator", name)
        return out.rename(columns={"model": "spec"})

    tidy = pd.concat(
        [_tag(pooled, "pooled"), _tag(between, "between"), _tag(within, "within")],
        ignore_index=True,
    )

    # --- Per-estimator curvature summary (from the full polynomial table model) ---
    g_lo, g_hi = float(g.min()), float(g.max())
    gbar_lo, gbar_hi = float(gbar.min()), float(gbar.max())
    pooled_full, between_full, within_full = (
        pooled.models[-1],
        between.models[-1],
        within.models[-1],
    )
    summary = pd.DataFrame(
        [
            _estimator_summary(
                "pooled",
                pooled_full,
                terms,
                g_lo,
                g_hi,
                int(getattr(pooled_full, "_N", len(work))),
                float(getattr(pooled_full, "_r2", float("nan"))),
            ),
            _estimator_summary(
                "between",
                between_full,
                terms,
                gbar_lo,
                gbar_hi,
                len(between_df),
                float(getattr(between_full, "_r2", float("nan"))),
            ),
            _estimator_summary(
                "within",
                within_full,
                terms,
                g_lo,
                g_hi,
                int(getattr(within_full, "_N", len(work))),
                float(getattr(within_full, "_r2_within", float("nan"))),
            ),
        ]
    )

    # --- Figure 1: raw pooled scatter + fitted polynomial ---
    grid = np.linspace(g_lo, g_hi, 200)
    betas_p = _poly_betas(pooled_full, terms)
    b0_p = float(pooled_full.coef().get("Intercept", 0.0))
    fig = _wave_fig(
        g,
        work[inequality].to_numpy(dtype=float),
        work[entity].map(lambda u: ent_disp.get(str(u), str(u))).to_numpy(),
        grid,
        b0_p + _eval_poly(betas_p, grid),
        dev_label,
        ineq_label,
        _shape_lines(
            _turning_points(betas_p, g_lo, g_hi),
            dev_label,
            int(getattr(pooled_full, "_N", len(work))),
            float(getattr(pooled_full, "_r2", float("nan"))),
            "R2",
        ),
        title or f"Kuznets waves (raw): {ineq_label} vs {dev_label}",
        n_sample=n_sample,
        seed=seed,
    )

    # --- Figure 2: between component plot (intercept + polynomial + controls partialled out) ---
    between_fig_model = analyze_estimation(
        between_df, inequality, idvs=[*terms, *controls], vcov=vcov
    ).models[0]
    betas_b = _poly_betas(between_fig_model, terms)
    b0_b = float(between_fig_model.coef().get("Intercept", 0.0))
    comp_b = _eval_poly(betas_b, gbar)
    y_between = b0_b + comp_b + np.asarray(between_fig_model.resid(), dtype=float)
    grid_b = np.linspace(gbar_lo, gbar_hi, 200)
    ctrl_note = " (controls partialled out)" if controls else ""
    fig_between = _wave_fig(
        gbar,
        y_between,
        between_df[entity].map(lambda u: ent_disp.get(str(u), str(u))).to_numpy(),
        grid_b,
        b0_b + _eval_poly(betas_b, grid_b),
        f"{dev_label} (entity mean)",
        f"{ineq_label} (entity mean{ctrl_note})",
        _shape_lines(
            _turning_points(betas_b, gbar_lo, gbar_hi),
            dev_label,
            len(between_df),
            float(getattr(between_fig_model, "_r2", float("nan"))),
            "R2",
        ),
        f"Kuznets waves (between estimator): {ineq_label}",
        n_sample=None,
        seed=seed,
    )

    # --- Figure 3: within component plot (controls + two-way FE partialled out, FWL) ---
    within_fig_model = analyze_estimation(
        work,
        inequality,
        idvs=[*terms, *controls],
        feffects=[entity, time],
        cluster=entity,
    ).models[0]
    betas_w = _poly_betas(within_fig_model, terms)
    fe_part = f" | {entity} + {time}"
    rhs = " + ".join(controls) if controls else "1"
    ry = _residualize(work, inequality, rhs, fe_part)  # gini net of controls + FE
    gmean = float(work[inequality].mean())
    y_within = ry + gmean
    comp_obs_w = _eval_poly(betas_w, g)
    curve_w = _eval_poly(betas_w, grid) - float(comp_obs_w.mean()) + gmean
    fig_within = _wave_fig(
        g,
        y_within,
        work[entity].map(lambda u: ent_disp.get(str(u), str(u))).to_numpy(),
        grid,
        curve_w,
        dev_label,
        f"{ineq_label} (within country & year{ctrl_note})",
        _shape_lines(
            _turning_points(betas_w, g_lo, g_hi),
            dev_label,
            int(getattr(within_fig_model, "_N", len(work))),
            float(getattr(within_fig_model, "_r2_within", float("nan"))),
            "within R2",
        ),
        f"Kuznets waves (within, two-way FE): {ineq_label}",
        n_sample=n_sample,
        seed=seed,
    )

    return KuznetsWavesResult(
        df=tidy,
        fig=fig,
        fig_between=fig_between,
        fig_within=fig_within,
        gt_pooled=pooled.etable,
        gt_between=between.etable,
        gt_within=within.etable,
        summary=summary,
        models=models,
        inequality=inequality,
        development=development,
        controls=tuple(controls),
        degree=degree,
        entity=entity,
        time=time,
        n_obs=len(work),
        notes=tuple(notes),
    )
