"""Plain-language interpretation of result objects.

Each function takes a *duck-typed* result object (it reads ``.df`` / ``.models`` / scalar
fields) and returns a Markdown string. Keeping the logic here — rather than in
``expdpy._types`` — keeps the result dataclasses thin and avoids an import cycle, since this
module never imports ``expdpy._types``.

Design rule: interpretations describe *associations*, never causal effects. The word
"causes" and the phrase "effect of" must not appear; a closing note points users to the
``correlation_vs_causation`` explainer.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from expdpy.pedagogy._format import (
    direction_word,
    fmt_num,
    sign_word,
    significance_phrase,
)

__all__ = [
    "interpret_beta_convergence",
    "interpret_coefficient_plot",
    "interpret_convergence_clubs",
    "interpret_correlation",
    "interpret_cre",
    "interpret_descriptive",
    "interpret_distribution_over_time",
    "interpret_estimation",
    "interpret_event_study",
    "interpret_fixef_plot",
    "interpret_fwl",
    "interpret_iv",
    "interpret_joint_test",
    "interpret_kuznets_waves",
    "interpret_marginal_effects",
    "interpret_missing_values",
    "interpret_panel_structure",
    "interpret_panel_view",
    "interpret_predictions",
    "interpret_regression",
    "interpret_robust_inference",
    "interpret_scatter_plot",
    "interpret_sandbox",
    "interpret_sigma_convergence",
    "interpret_spaghetti",
    "interpret_transition_matrix",
    "interpret_trend",
    "interpret_value_heatmap",
    "interpret_within_between",
    "interpret_within_persistence",
    "interpret_xtsum",
]

_ASSOC_NOTE = (
    "_These are associations, not causal effects. A causal reading needs a research "
    "design — see `explain('correlation_vs_causation')`._"
)
_MAX_VARS = 6


def interpret_beta_convergence(result: Any, *, lang: str = "en") -> str:
    """Interpret a β-convergence result: the slope, speed λ, half-life and conditioning."""
    var = str(getattr(result, "var", "the variable"))
    beta = float(result.beta)
    speed = float(result.speed)
    half_life = float(result.half_life)
    horizon = float(result.horizon)
    converging = beta < 0

    lines = [
        f"Across {int(result.n_obs):,} units over a {fmt_num(horizon)}-period horizon, the "
        f"average growth rate of **{var}** is "
        f"{'negatively' if converging else 'positively'} associated with its initial level "
        f"(β = {fmt_num(beta)}). "
        + (
            "Units that start lower tend to grow faster — the pattern of **unconditional "
            "β-convergence**."
            if converging
            else "Units that start higher tend to grow faster — **divergence** rather than "
            "convergence."
        )
    ]
    if converging and not math.isnan(speed) and speed > 0:
        lines.append(
            f"The implied speed of convergence is λ = {fmt_num(speed)} per period, so about "
            f"half of an initial gap closes every {fmt_num(half_life)} periods (the half-life)."
        )

    controls = tuple(getattr(result, "controls", ()) or ())
    beta_c = float(getattr(result, "beta_cond", float("nan")))
    if controls and not math.isnan(beta_c):
        speed_c = float(result.speed_cond)
        half_c = float(result.half_life_cond)
        steeper = abs(beta_c) > abs(beta)
        tail = (
            f", a speed of λ = {fmt_num(speed_c)} per period (half-life ≈ "
            f"{fmt_num(half_c)} periods)."
            if not math.isnan(speed_c) and speed_c > 0
            else "."
        )
        lines.append(
            f"Holding {', '.join(controls)} fixed at their initial values (via the "
            f"Frisch-Waugh-Lovell theorem), the convergence slope is β = {fmt_num(beta_c)} — "
            f"{'steeper' if steeper else 'flatter'} than the unconditional {fmt_num(beta)}, "
            f"the pattern of **conditional β-convergence**{tail}"
        )

    rolling = getattr(result, "rolling", None)
    if rolling is not None and len(rolling) >= 2:
        first = float(rolling["beta"].iloc[0])
        last = float(rolling["beta"].iloc[-1])
        lines.append(
            "Across fixed-width rolling windows the convergence slope moved "
            f"{direction_word(last - first)} over time (β = {fmt_num(first)} in the earliest "
            f"window to {fmt_num(last)} in the latest)."
        )

    lines += ["", _ASSOC_NOTE]
    return "\n".join(lines)


def interpret_sigma_convergence(result: Any, *, lang: str = "en") -> str:
    """Interpret a σ-convergence result: whether and how fast cross-sectional spread narrows."""
    var = str(getattr(result, "var", "the variable"))
    n_units = int(getattr(result, "n_units", 0))
    n_periods = int(getattr(result, "n_periods", 0))
    std_slope = float(result.std_slope)
    std_p = float(result.std_pvalue)
    gini_slope = float(getattr(result, "gini_slope", float("nan")))

    df = result.df
    std_first = float(df["std"].iloc[0])
    std_last = float(df["std"].iloc[-1])

    if not math.isfinite(std_slope):
        return "\n".join(
            [
                f"Across {n_units:,} units over {n_periods} periods, the cross-sectional "
                f"spread of **{var}** could not be summarised by a trend.",
                "",
                _ASSOC_NOTE,
            ]
        )

    converging = std_slope < 0
    pct = abs(math.expm1(std_slope)) * 100.0  # |e^b - 1|: per-period % change in spread
    lines = [
        f"Across {n_units:,} units over {n_periods} periods, the cross-sectional standard "
        f"deviation of **{var}** "
        + ("narrowed" if converging else "widened")
        + f" (from {fmt_num(std_first)} to {fmt_num(std_last)}). The log-dispersion trend is "
        f"{fmt_num(std_slope)} per period — about {fmt_num(pct)}% "
        + ("less" if converging else "more")
        + f" dispersion each period ({significance_phrase(std_p)}) — the pattern of "
        + (
            "**σ-convergence**."
            if converging
            else "**σ-divergence** rather than convergence."
        )
    ]
    if math.isfinite(gini_slope):
        agree = (gini_slope < 0) == converging
        lines.append(
            "The Gini index "
            + ("also " if agree else "")
            + ("narrowed" if gini_slope < 0 else "widened")
            + f" over the same span (trend {fmt_num(gini_slope)} per period)"
            + ("." if agree else ", a different direction from the standard deviation.")
        )
    lines += ["", _ASSOC_NOTE]
    return "\n".join(lines)


def interpret_convergence_clubs(result: Any, *, lang: str = "en") -> str:
    """Interpret a club-convergence result: how the panel splits into convergence clubs."""
    var = str(getattr(result, "var", "the variable"))
    n_units = int(getattr(result, "n_units", 0))
    n_periods = int(getattr(result, "n_periods", 0))
    n_clubs = int(getattr(result, "n_clubs", 0))
    n_div = int(getattr(result, "n_divergent", 0))
    converged = bool(getattr(result, "converged", False))
    g_t = float(getattr(result, "global_tstat", float("nan")))
    tcrit = float(getattr(result, "tcrit", -1.65))

    head = (
        f"Across {n_units:,} units over {n_periods} periods, the Phillips-Sul log(t) test "
        f"for **{var}** "
    )
    if converged:
        lines = [
            head + f"does not reject convergence for the whole panel "
            f"(t = {fmt_num(g_t)} > {tcrit:g}): the units form a **single convergence "
            "club** — they all approach a common path."
        ]
        lines += ["", _ASSOC_NOTE]
        return "\n".join(lines)

    if n_clubs == 0:
        lines = [
            head
            + f"rejects global convergence (t = {fmt_num(g_t)} <= {tcrit:g}) and the "
            "clustering algorithm finds **no convergence clubs** — the units diverge rather "
            "than forming catch-up groups."
        ]
        lines += ["", _ASSOC_NOTE]
        return "\n".join(lines)

    summary = result.summary
    clubs = summary[summary["club"] != "Divergent"]
    sizes = ", ".join(
        f"{row['club']} ({int(row['n_members'])})" for _, row in clubs.iterrows()
    )
    lines = [
        head
        + f"rejects global convergence (t = {fmt_num(g_t)} <= {tcrit:g}). The clustering "
        f"algorithm splits the panel into **{n_clubs} convergence club"
        + ("s" if n_clubs != 1 else "")
        + f"** — groups that each converge internally but not with one another: {sizes}."
    ]
    if n_div:
        lines.append(
            f"{n_div} unit"
            + ("s" if n_div != 1 else "")
            + " do not join any club (the divergent group)."
        )
    lines.append(
        "Club 1 collects the highest-ranked units; within each club the log(t) slope b = "
        f"2*alpha is positive enough that its t-statistic clears {tcrit:g}."
    )
    lines += ["", _ASSOC_NOTE]
    return "\n".join(lines)


def interpret_regression(result: Any, *, lang: str = "en") -> str:
    """Interpret a regression table: sign, magnitude and significance per coefficient."""
    model = result.models[0]
    df = result.df
    dv = str(getattr(model, "_depvar", "the outcome"))
    has_fe = bool(getattr(model, "_has_fixef", False))
    clustered = bool(getattr(model, "_is_clustered", False))

    first_id = df["model"].iloc[0]
    sub = df[df["model"] == first_id]
    n_models = int(df["model"].nunique())

    head = [f"This OLS regression relates **{dv}** to its regressors."]
    if has_fe:
        head.append(
            f"Fixed effects for *{getattr(model, '_fixef', '')}* absorb time-invariant "
            "differences, so coefficients reflect variation **within** each group."
        )
    if clustered:
        cl = ", ".join(getattr(model, "_clustervar", []) or [])
        head.append(f"Standard errors are clustered by *{cl}*.")

    bullets = []
    for _, row in sub.iterrows():
        term = str(row["term"])
        if term.lower() == "intercept":
            continue
        est = float(row["Estimate"])
        p = float(row["Pr(>|t|)"])
        bullets.append(
            f"- **{term}**: each one-unit increase is associated with {dv} that is "
            f"{fmt_num(abs(est))} {direction_word(est)} "
            f"({significance_phrase(p)})."
        )

    n_obs = int(getattr(model, "_N", len(df)))
    r2 = float(getattr(model, "_r2", math.nan))
    fit = f"Model fit: N = {n_obs:,}, R² = {fmt_num(r2)}"
    if has_fe:
        fit += f", within-R² = {fmt_num(float(getattr(model, '_r2_within', math.nan)))}"
    fit += "."

    parts = [" ".join(head), "", *bullets, "", fit]
    if n_models > 1:
        parts.append(f"(This reads the first of {n_models} models in the table.)")
    parts += ["", _ASSOC_NOTE]
    return "\n".join(parts)


def _coef_phrase(term: str, est: float, dv: str) -> str:
    """Return the plain-language phrase for one OLS coefficient."""
    return (
        f"each one-unit increase is associated with {dv} that is "
        f"{fmt_num(abs(est))} {direction_word(est)}"
    )


def interpret_estimation(result: Any, *, lang: str = "en") -> str:
    """Interpret an :class:`~expdpy.EstimationResult` (OLS)."""
    model = result.models[0]
    df = result.df
    dv = str(getattr(model, "_depvar", "the outcome"))
    has_fe = bool(getattr(model, "_has_fixef", False))
    clustered = bool(getattr(model, "_is_clustered", False))

    first_id = df["model"].iloc[0]
    sub = df[df["model"] == first_id]

    head = [f"This OLS model relates **{dv}** to its regressors."]
    if has_fe:
        head.append(
            f"Fixed effects for *{getattr(model, '_fixef', '')}* absorb time-invariant "
            "differences, so coefficients reflect variation **within** each group."
        )
    if clustered:
        cl = ", ".join(getattr(model, "_clustervar", []) or [])
        head.append(f"Standard errors are clustered by *{cl}*.")

    bullets = []
    for _, row in sub.iterrows():
        term = str(row["term"])
        if term.lower() == "intercept":
            continue
        phrase = _coef_phrase(term, float(row["Estimate"]), dv)
        bullets.append(
            f"- **{term}**: {phrase} ({significance_phrase(float(row['Pr(>|t|)']))})."
        )

    parts = [" ".join(head), "", *bullets, "", _ASSOC_NOTE]
    return "\n".join(parts)


def interpret_iv(result: Any, *, lang: str = "en") -> str:
    """Interpret an IV (2SLS) regression: instrumented slope plus the weak-instrument F."""
    df = result.df
    model = result.model
    dv = str(getattr(model, "_depvar", "the outcome"))
    endog = tuple(getattr(result, "endog", ()) or ())
    instruments = tuple(getattr(result, "instruments", ()) or ())
    f = float(getattr(result, "first_stage_f", math.nan))

    endog_set = set(endog)
    lines = [
        f"This IV (2SLS) regression instruments **{', '.join(endog)}** with "
        f"*{', '.join(instruments)}*, relating it to **{dv}**. Two-stage least squares "
        "isolates the part of the endogenous regressor that moves with the instruments, "
        "purging the endogeneity that biases ordinary least squares.",
        "",
    ]
    first_id = df["model"].iloc[0]
    sub = df[df["model"] == first_id]
    for _, row in sub.iterrows():
        term = str(row["term"])
        if term.lower() == "intercept" or term not in endog_set:
            continue
        est = float(row["Estimate"])
        lines.append(
            f"- **{term}** (instrumented): a one-unit increase is associated with {dv} "
            f"that is {fmt_num(abs(est))} {direction_word(est)} "
            f"({significance_phrase(float(row['Pr(>|t|)']))})."
        )

    if not math.isnan(f):
        weak = f < 10.0
        lines += [
            "",
            f"The first-stage F is {fmt_num(f)} — "
            + (
                "**below** the conventional weak-instrument threshold of 10, so the "
                "instruments are weak and the 2SLS estimates should be read with caution."
                if weak
                else "**above** the conventional weak-instrument threshold of 10, so the "
                "instruments are reasonably strong."
            ),
        ]
    lines.append(
        "IV is trustworthy only when the instruments are both **relevant** (a strong first "
        f"stage) and **excludable** — related to {dv} solely through the instrumented "
        "regressor."
    )
    lines += ["", _ASSOC_NOTE]
    return "\n".join(lines)


def interpret_marginal_effects(result: Any, *, lang: str = "en") -> str:
    """Interpret a marginal-effects plot: how the focal slope varies with the moderator."""
    foc = str(getattr(result, "focal", "the regressor"))
    mod = str(getattr(result, "moderator", "the moderator"))
    df = result.df
    grid = df[mod].to_numpy(dtype=float)
    me = df["me"].to_numpy(dtype=float)
    lo_x, hi_x = float(grid[0]), float(grid[-1])
    me_lo, me_hi = float(me[0]), float(me[-1])
    ame = float(getattr(result, "ame", math.nan))
    se = float(getattr(result, "ame_se", math.nan))

    lines = [
        f"The marginal association of **{foc}** with the outcome is not constant — it depends "
        f"on **{mod}** (their interaction).",
        f"Across the observed range it moves from {fmt_num(me_lo)} at {mod} = {fmt_num(lo_x)} "
        f"to {fmt_num(me_hi)} at {mod} = {fmt_num(hi_x)}.",
    ]

    # Sign change inside the moderator range (a linear marginal effect crosses zero at most once).
    sign = np.sign(me)
    flips = np.where(np.diff(sign) != 0)[0]
    if len(flips):
        k = int(flips[0])
        x0, x1, y0, y1 = grid[k], grid[k + 1], me[k], me[k + 1]
        cross = x0 if y1 == y0 else x0 - y0 * (x1 - x0) / (y1 - y0)
        lines.append(
            f"The association switches sign near {mod} = {fmt_num(float(cross))}: "
            f"{foc} relates to the outcome in opposite directions below and above that point."
        )

    if not (math.isnan(ame) or math.isnan(se)) and se > 0:
        z = abs(ame / se)
        p = math.erfc(z / math.sqrt(2.0))  # two-sided normal p-value
        lines.append(
            f"Averaged over the sample, the marginal association is {fmt_num(ame)} "
            f"({significance_phrase(p)})."
        )
    lines += ["", _ASSOC_NOTE]
    return "\n".join(lines)


def interpret_cre(result: Any, *, lang: str = "en") -> str:
    """Interpret a Correlated Random Effects (Mundlak) model: within estimates + FE-vs-RE."""
    model = result.models[0]
    df = result.df
    dv = str(getattr(model, "_depvar", "the outcome"))
    means = set(getattr(model, "_cre_means", []))
    stat = float(getattr(model, "_cre_mundlak_stat", math.nan))
    p = float(getattr(model, "_cre_mundlak_p", math.nan))
    k = int(getattr(model, "_cre_mundlak_df", len(means)))

    lines = [
        f"This Correlated Random Effects (Mundlak) model relates **{dv}** to its regressors "
        "*and* their unit (entity) means. By the Mundlak equivalence the coefficient on each "
        "original regressor equals its **within (fixed-effects)** estimate, while the "
        "coefficient on the mean is the gap between the between- and within-unit "
        "associations.",
        "",
    ]
    first_id = df["model"].iloc[0]
    sub = df[df["model"] == first_id]
    for _, row in sub.iterrows():
        term = str(row["term"])
        if term.lower() == "intercept" or term in means:
            continue
        est = float(row["Estimate"])
        lines.append(
            f"- **{term}** (within estimate): a one-unit increase is associated with {dv} "
            f"that is {fmt_num(abs(est))} {direction_word(est)} "
            f"({significance_phrase(float(row['Pr(>|t|)']))})."
        )
    if not math.isnan(p):
        verdict = (
            "differ from zero, so the unit effects are correlated with the regressors — "
            "**prefer fixed effects** (random effects would be biased)"
            if p < 0.05
            else "are indistinguishable from zero, so **random effects is admissible** "
            "(and more efficient than fixed effects)"
        )
        lines += [
            "",
            f"Joint test that the {k} mean coefficient(s) are zero — the regression-form "
            f"Hausman test — χ²({k}) = {fmt_num(stat)}, p = {fmt_num(p)}: this {verdict}.",
        ]
    lines += ["", _ASSOC_NOTE]
    return "\n".join(lines)


def interpret_event_study(result: Any, *, lang: str = "en") -> str:
    """Interpret an event study: pre-trend diagnostic and the dynamic post-treatment path."""
    df = result.df
    estimator = getattr(result, "estimator", "event study")
    pre = df[df["event_time"] < -1]
    post = df[df["event_time"] >= 0]

    lines = [
        f"This event study (estimator: **{estimator}**) traces the outcome by event time, "
        "with t = -1 as the baseline period."
    ]

    # Pre-trend diagnostic: do any pre-period intervals exclude zero?
    if len(pre):
        flags = (pre["ci_lower"] > 0) | (pre["ci_upper"] < 0)
        if bool(flags.any()):
            lines.append(
                "⚠️ Some **pre-treatment** coefficients differ from zero, which weakens the "
                "parallel-trends assumption — read the post-treatment path with caution."
            )
        else:
            lines.append(
                "Pre-treatment coefficients are statistically indistinguishable from zero, "
                "which is consistent with parallel trends."
            )

    # Dynamic effect: the latest post-treatment estimate.
    if len(post):
        last = post.sort_values("event_time").iloc[-1]
        excl = (last["ci_lower"] > 0) or (last["ci_upper"] < 0)
        lines.append(
            f"By event time {int(last['event_time'])}, the estimated effect is "
            f"{fmt_num(float(last['estimate']))} "
            + (
                "(95% interval excludes zero)"
                if excl
                else "(95% interval includes zero)"
            )
            + "."
        )

    lines += ["", _ASSOC_NOTE]
    return "\n".join(lines)


def interpret_sandbox(result: Any, *, lang: str = "en") -> str:
    """Interpret a teaching sandbox from its ``summary`` scalars and ``topic``."""
    s = result.summary
    topic = result.topic
    if topic == "omitted_variable_bias":
        bias = s["short_coef"] - s["true_beta_x"]
        return (
            f"Leaving out the confounder, the short regression estimates "
            f"{fmt_num(s['short_coef'])} for the focal regressor — biased by "
            f"{fmt_num(bias)} away from the true {fmt_num(s['true_beta_x'])}. Controlling for "
            f"the confounder, the long regression recovers {fmt_num(s['long_coef'])}. "
            "The bias is the confounder's effect times its correlation with the regressor."
        )
    if topic == "fixed_effects":
        return (
            f"Pooled OLS estimates {fmt_num(s['pooled_coef'])} for the slope, biased by the "
            "correlation between the regressor and the unit effects. Adding unit fixed "
            f"effects recovers {fmt_num(s['fe_coef'])}, close to the true "
            f"{fmt_num(s['true_beta'])} — the within estimator removes the bias."
        )
    if topic == "clustered_se":
        return (
            f"The point estimate ({fmt_num(s['coef'])}) is identical either way — clustering "
            f"changes only the standard error, from {fmt_num(s['iid_se'])} (iid) to "
            f"{fmt_num(s['clustered_se'])} (clustered), a {fmt_num(s['se_ratio'])}x increase. "
            "Ignoring within-cluster correlation overstates precision."
        )
    if topic == "first_differences":
        return (
            f"Pooled OLS estimates {fmt_num(s['pooled_coef'])}, biased by the unit effects. "
            f"First differencing gives {fmt_num(s['fd_coef'])} and the within (demeaning) "
            f"estimator {fmt_num(s['within_coef'])} — both recover the true "
            f"{fmt_num(s['true_beta'])}, because differencing and demeaning each cancel the "
            "unit effect. On this two-period panel they coincide (gap "
            f"{fmt_num(s['fd_within_gap'])})."
        )
    if topic == "within_transformation":
        return (
            f"The within (demeaning) estimator gives {fmt_num(s['within_coef'])} and "
            f"least-squares dummy variables {fmt_num(s['lsdv_coef'])} — the same slope (gap "
            f"{fmt_num(s['within_lsdv_gap'])}) versus the true {fmt_num(s['true_beta'])}. "
            "Demeaning and a dummy per unit do the same job (Frisch-Waugh-Lovell)."
        )
    if topic == "beta_convergence":
        return (
            f"Omitting the steady-state determinant, the unconditional slope is "
            f"{fmt_num(s['unconditional_coef'])} — biased away from the true convergence "
            f"slope {fmt_num(s['true_beta'])}, so the units look like they barely converge. "
            f"Conditioning on the determinant recovers {fmt_num(s['conditional_coef'])}, "
            "matching the truth — that is conditional β-convergence. The recovered speed is "
            f"{fmt_num(s['conditional_speed'])} per period (true "
            f"{fmt_num(s['true_speed'])}), a half-life of "
            f"{fmt_num(s['conditional_half_life'])} periods."
        )
    if topic == "sigma_convergence":
        return (
            f"With every unit's value contracting toward a common mean at rate "
            f"{fmt_num(s['rho'])} per period, the cross-sectional dispersion narrows at a known "
            f"log-rate of {fmt_num(s['true_slope'])} per period. The standard-deviation trend "
            f"recovers {fmt_num(s['std_slope'])} and the Gini trend {fmt_num(s['gini_slope'])} "
            "— both matching the truth, the hallmark of σ-convergence."
        )
    if topic == "kuznets_waves":
        return (
            f"The panel was built with a known degree-{int(s['degree'])} Kuznets wave whose "
            f"top-order coefficient is {fmt_num(s['true_top'])}. The within (two-way "
            f"fixed-effects) estimator recovers {fmt_num(s['within_top'])} and pooled OLS "
            f"{fmt_num(s['pooled_top'])} — both close to the truth, because the wave is a "
            "within-unit relationship. The between estimator gives "
            f"{fmt_num(s['between_top'])}, which differs: it compares unit averages, and the "
            "average of a nonlinear curve is not the curve of the average."
        )
    if topic == "convergence_clubs":
        return (
            f"The panel was built with {int(s['true_clubs'])} planted convergence clubs. The "
            f"whole-panel log(t) test rejects global convergence "
            f"(t = {fmt_num(s['global_tstat'])} <= -1.65), and the data-driven clustering "
            f"recovers {int(s['detected_clubs'])} club(s), placing "
            f"{fmt_num(s['accuracy'] * 100)}% of the {int(s['n_units'])} units in their true "
            "club — the algorithm finds the structure without being told it."
        )
    if topic == "hausman":
        reject = s["hausman_p"] < 0.05
        verdict = (
            "**reject** their equality and prefer fixed effects"
            if reject
            else "do not reject their equality"
        )
        return (
            f"With the regressor correlated with the unit effects, random effects estimates "
            f"{fmt_num(s['re_coef'])} while fixed effects gives {fmt_num(s['fe_coef'])} "
            f"(true {fmt_num(s['true_beta'])}). The Hausman statistic is "
            f"{fmt_num(s['hausman_stat'])} (p = {fmt_num(s['hausman_p'])}), so we {verdict} — "
            "random effects is the biased one here."
        )
    if topic == "correlated_random_effects":
        return (
            f"Adding each unit's time-mean of the regressor, the correlated-random-effects "
            f"slope on the original regressor is {fmt_num(s['cre_within_coef'])} — the same as "
            f"plain fixed effects ({fmt_num(s['fe_coef'])}) and close to the true "
            f"{fmt_num(s['true_beta'])}. The Mundlak test on the added means is "
            f"{fmt_num(s['mundlak_stat'])} (p = {fmt_num(s['mundlak_p'])}), the "
            "random-effects-versus-fixed-effects comparison written as a regression."
        )
    if topic == "nickell_bias":
        return (
            f"At the shortest panel the within (fixed-effects) estimate of the autoregressive "
            f"parameter is {fmt_num(s['fe_rho_short'])} — {fmt_num(abs(s['bias_short']))} below "
            f"the true {fmt_num(s['true_rho'])}. As the panel lengthens the bias shrinks to "
            f"{fmt_num(abs(s['bias_long']))} (estimate {fmt_num(s['fe_rho_long'])}). Demeaning "
            "correlates the lagged outcome with the transformed error — the Nickell bias — and "
            "it fades only as T grows."
        )
    if topic == "measurement_error":
        return (
            f"Regressing on the noisy measurement, the slope is {fmt_num(s['naive_coef'])} — "
            f"attenuated toward zero from the true {fmt_num(s['true_beta'])}. The attenuation "
            f"matches the reliability ratio {fmt_num(s['reliability'])} "
            f"(observed {fmt_num(s['attenuation'])}): classical noise in a regressor pulls its "
            "coefficient toward zero."
        )
    return "Sandbox demonstration."  # pragma: no cover - defensive


def interpret_correlation(result: Any, *, lang: str = "en") -> str:
    """Interpret a correlation table: strongest pair and Pearson-vs-Spearman divergence."""
    corr = result.df_corr
    prob = result.df_prob
    names = list(corr.columns)

    best: tuple[float, str, str] | None = None
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            r = float(corr.loc[a, b])  # Pearson lives above the diagonal
            if not math.isnan(r) and (best is None or abs(r) > abs(best[0])):
                best = (r, a, b)

    if best is None:
        return (
            "No correlations could be computed (need at least two numeric variables)."
        )

    r, a, b = best
    p = float(prob.loc[a, b])
    spearman = float(corr.loc[b, a])  # Spearman lives below the diagonal
    lines = [
        f"The strongest linear association is between **{a}** and **{b}** "
        f"(Pearson r = {fmt_num(r)}, {significance_phrase(p)})."
    ]
    if not math.isnan(spearman) and abs(r - spearman) > 0.1:
        lines.append(
            f"Its Spearman (rank) correlation is {fmt_num(spearman)} — the gap from Pearson "
            "hints at non-linearity or outliers."
        )
    lines += ["", _ASSOC_NOTE]
    return "\n".join(lines)


def interpret_fwl(result: Any, *, lang: str = "en") -> str:
    """Interpret a Frisch-Waugh-Lovell partial-regression plot."""
    slope = float(result.slope)
    se = float(result.se)
    try:
        x_label = result.fig.layout.xaxis.title.text or "the focal regressor"
        y_label = result.fig.layout.yaxis.title.text or "the outcome"
    except AttributeError:  # pragma: no cover - figures always carry axis titles
        x_label, y_label = "the focal regressor", "the outcome"
    x_label = str(x_label).replace("Residualized ", "")
    y_label = str(y_label).replace("Residualized ", "")

    lines = [
        f"Holding the other regressors and fixed effects fixed, **{x_label}** has a partial "
        f"slope of {fmt_num(slope)} on **{y_label}** (SE {fmt_num(se)}).",
    ]
    if not math.isnan(se) and se > 0:
        excludes = abs(slope) > 1.96 * se
        lines.append(
            "Its 95% confidence interval "
            + ("excludes" if excludes else "includes")
            + " zero."
        )
    lines.append(
        "By the Frisch-Waugh-Lovell theorem this slope equals the focal coefficient in the "
        "regression table."
    )
    r2w = float(getattr(result, "r2_within", math.nan))
    if not math.isnan(r2w):
        lines.append(f"Within-R² of the full model: {fmt_num(r2w)}.")
    lines += ["", _ASSOC_NOTE]
    return "\n".join(lines)


_WAVE_SHAPE = {
    0: "no turning point (a monotonic relationship)",
    1: "a single turning point (an inverted-U or U shape — the classic Kuznets curve)",
    2: "two turning points (an N or inverted-N shape)",
    3: "three turning points (a full Kuznets wave)",
}
_WAVE_ESTIMATORS = (
    ("pooled", "Pooled OLS", "the raw cross-sectional pattern"),
    ("between", "Between (cross-country)", "comparing country averages"),
    (
        "within",
        "Within (two-way FE)",
        "within-country variation net of common year effects",
    ),
)


def interpret_kuznets_waves(result: Any, *, lang: str = "en") -> str:
    """Interpret a Kuznets-waves result: the nonlinear shape under three panel estimators."""
    ineq = str(getattr(result, "inequality", "inequality"))
    dev = str(getattr(result, "development", "development"))
    degree = int(getattr(result, "degree", 4))
    n_obs = int(getattr(result, "n_obs", 0))
    controls = tuple(getattr(result, "controls", ()) or ())
    summary = result.summary
    by = {str(row["estimator"]): row for _, row in summary.iterrows()}

    lines = [
        f"Across {n_obs:,} observations, a degree-{degree} polynomial relates **{ineq}** to "
        f"**{dev}** (the extended Kuznets-waves specification). The three estimators read the "
        "association at different levels of variation:"
    ]
    shapes: dict[str, int] = {}
    for key, label, gloss in _WAVE_ESTIMATORS:
        if key not in by:
            continue
        row = by[key]
        ntp = int(row["n_turning_points"])
        shapes[key] = ntp
        desc = _WAVE_SHAPE.get(ntp, f"{ntp} turning points")
        peak = float(row["peak_g"])
        peak_txt = (
            f", peaking near {dev} = {fmt_num(peak)}" if math.isfinite(peak) else ""
        )
        sig = significance_phrase(float(row["top_pvalue"]))
        lines.append(
            f"- **{label}** ({gloss}): the fitted curve shows {desc}{peak_txt}; its "
            f"highest-order term is {sig} (R² = {fmt_num(float(row['r2']))})."
        )

    if shapes:
        distinct = set(shapes.values())
        if len(distinct) == 1:
            lines.append(
                "All three estimators agree on the curvature, so the shape is not an artefact "
                "of cross-country versus within-country variation."
            )
        else:
            lines.append(
                "The estimators disagree on the curvature: the cross-country (between) and "
                "within-country (fixed-effects) pictures differ, a sign that unit-level "
                "confounders shape the apparent curve."
            )
    if controls:
        lines.append(
            f"The between and within figures partial out {', '.join(controls)} via the "
            "Frisch-Waugh-Lovell theorem, so the plotted wave is net of those covariates."
        )

    lines += ["", _ASSOC_NOTE]
    return "\n".join(lines)


def interpret_trend(result: Any, *, lang: str = "en") -> str:
    """Interpret a time-trend graph: direction of change for each series."""
    df = result.df
    time_col = df.columns[1]  # columns are: variable, <time>, mean, se
    lines = ["Over the observed period:"]
    for var, grp in df.groupby("variable", sort=False):
        g = grp.sort_values(time_col)
        first, last = g.iloc[0], g.iloc[-1]
        delta = float(last["mean"]) - float(first["mean"])
        moved = direction_word(delta)
        verb = "stayed roughly flat" if moved == "unchanged" else f"moved {moved}"
        lines.append(
            f"- **{var}** {verb}, from {fmt_num(float(first['mean']))} in "
            f"{first[time_col]} to {fmt_num(float(last['mean']))} in {last[time_col]} "
            f"(change of {fmt_num(delta)})."
        )
    return "\n".join(lines)


def interpret_descriptive(result: Any, *, lang: str = "en") -> str:
    """Interpret a descriptive table: central tendency, spread and skew per variable."""
    df = result.df
    lines = []
    for var in list(df.index)[:_MAX_VARS]:
        row = df.loc[var]
        mean = float(row["Mean"])
        median = float(row["Median"])
        sd = float(row["Std. dev."])
        n = int(row["N"])
        if not math.isnan(sd) and sd > 0 and (mean - median) > 0.1 * sd:
            skew = "right-skewed (mean above median)"
        elif not math.isnan(sd) and sd > 0 and (median - mean) > 0.1 * sd:
            skew = "left-skewed (mean below median)"
        else:
            skew = "fairly symmetric"
        lines.append(
            f"- **{var}**: mean {fmt_num(mean)}, median {fmt_num(median)}, "
            f"SD {fmt_num(sd)} (N = {n:,}); {skew}."
        )
    if len(df.index) > _MAX_VARS:
        lines.append(f"(Showing the first {_MAX_VARS} of {len(df.index)} variables.)")
    return "\n".join(lines)


def interpret_xtsum(result: Any, *, lang: str = "en") -> str:
    """Interpret a within/between (xtsum) table: where each variable's variation lives."""
    df = result.df
    variables = list(dict.fromkeys(df["variable"]))[:_MAX_VARS]
    lines = [
        "Splitting each variable's variation into **between** (differences across units) "
        "and **within** (variation over time inside a unit):"
    ]
    for var in variables:
        g = df[df["variable"] == var].set_index("component")
        try:
            b = float(g.loc["between", "sd"])
            w = float(g.loc["within", "sd"])
        except KeyError:  # pragma: no cover - defensive
            continue
        if math.isnan(b) or math.isnan(w) or (b == 0 and w == 0):
            continue
        share = b * b / (b * b + w * w)
        where = (
            "mostly **across units** (between)"
            if share > 0.6
            else "mostly **over time within units** (within)"
            if share < 0.4
            else "split between cross-unit and over-time"
        )
        lines.append(
            f"- **{var}**: between SD {fmt_num(b)}, within SD {fmt_num(w)} — variation is "
            f"{where}."
        )
    if len(set(df["variable"])) > _MAX_VARS:
        lines.append(f"(Showing the first {_MAX_VARS} variables.)")
    lines += [
        "",
        "_Between variation drives cross-sectional comparisons; within variation is what "
        "fixed-effects (within) estimators rely on._",
    ]
    return "\n".join(lines)


def interpret_within_between(result: Any, *, lang: str = "en") -> str:
    """Interpret a within-vs-between scatter: how the pooled slope decomposes."""
    bp, bb, bw = (
        float(result.slope_pooled),
        float(result.slope_between),
        float(result.slope_within),
    )
    try:
        x = result.fig.layout.xaxis.title.text or "x"
        y = result.fig.layout.yaxis.title.text or "y"
    except AttributeError:  # pragma: no cover - figures carry axis titles
        x, y = "x", "y"
    lines = [
        f"The pooled association between **{x}** and **{y}** has slope {fmt_num(bp)}. It "
        f"blends a **between-unit** slope of {fmt_num(bb)} (comparing unit averages) with a "
        f"**within-unit** slope of {fmt_num(bw)} (variation over time inside a unit)."
    ]
    if not (math.isnan(bb) or math.isnan(bw)) and abs(bb - bw) > 0.5 * max(
        abs(bb), abs(bw), 1e-9
    ):
        lines.append(
            "The two diverge markedly — the cross-unit and over-time relationships tell "
            "different stories, a sign that unit-level confounders matter and that a "
            "fixed-effects specification would change the estimate."
        )
    lines += ["", _ASSOC_NOTE]
    return "\n".join(lines)


def interpret_spaghetti(result: Any, *, lang: str = "en") -> str:
    """Interpret a spaghetti plot: central trend and whether trajectories fan out."""
    df = result.df
    time_col, val_col = df.columns[1], df.columns[2]
    by_t = df.groupby(time_col, observed=True)[val_col]
    mean_t = by_t.mean()
    sd_t = by_t.std(ddof=1)
    head = f"Each faint line is one of {result.n_shown:,} unit trajectories"
    if result.n_shown < result.n_units:
        head += f" (a sample of {result.n_units:,})"
    head += "; the bold line is the cross-unit central tendency."
    lines = [head]
    if len(mean_t) > 1:
        delta = float(mean_t.iloc[-1] - mean_t.iloc[0])
        lines.append(
            f"The central path moved {direction_word(delta)} overall (change {fmt_num(delta)})."
        )
        first_sd, last_sd = float(sd_t.iloc[0]), float(sd_t.iloc[-1])
        if not (math.isnan(first_sd) or math.isnan(last_sd)):
            spread = (
                "widened — units diverged"
                if last_sd > first_sd * 1.1
                else "narrowed — units converged"
                if last_sd < first_sd * 0.9
                else "stayed about the same"
            )
            lines.append(
                f"Cross-unit dispersion {spread} (SD {fmt_num(first_sd)} → {fmt_num(last_sd)})."
            )
    return "\n".join(lines)


def interpret_panel_structure(result: Any, *, lang: str = "en") -> str:
    """Interpret a panel-structure summary: balance, coverage and gaps."""
    d = dict(
        zip(result.df_summary["statistic"], result.df_summary["value"], strict=True)
    )
    n_units, n_periods = int(d["units"]), int(d["periods"])
    balanced, gaps = bool(d["balanced"]), int(d["internal gaps"])
    mino, maxo = int(d["min obs per unit"]), int(d["max obs per unit"])
    head = (
        f"The panel has **{n_units:,} units** observed over **{n_periods:,} periods**."
    )
    if balanced:
        body = "It is **balanced**: every unit appears in every period, with no gaps."
    else:
        gap_txt = (
            f", with {gaps:,} interior gap(s)"
            if gaps
            else " (no interior gaps — only ragged starts/ends)"
        )
        body = (
            f"It is **unbalanced**: units are observed between {mino} and {maxo} periods"
            f"{gap_txt}."
        )
    tail = (
        "Unbalanced panels and interior gaps reduce the information a within (fixed-effects) "
        "estimator can use, and shifting unit composition over time can drive apparent trends."
    )
    return "\n".join([head, "", body, "", tail])


def interpret_distribution_over_time(result: Any, *, lang: str = "en") -> str:
    """Interpret a distribution-over-time view: shifts in center and spread."""
    df = result.df
    time_col, val_col = df.columns[0], df.columns[1]
    grouped = df.groupby(time_col, observed=True)[val_col]
    med = grouped.median()
    iqr = grouped.quantile(0.75) - grouped.quantile(0.25)
    if len(med) < 2:
        return f"The distribution of **{val_col}** is shown for a single period."
    dmed = float(med.iloc[-1] - med.iloc[0])
    dspr = float(iqr.iloc[-1] - iqr.iloc[0])
    spread_word = "widened" if dspr > 0 else "narrowed" if dspr < 0 else "was stable"
    return "\n".join(
        [
            f"Tracking the distribution of **{val_col}** across periods:",
            f"- The median moved {direction_word(dmed)} "
            f"(from {fmt_num(float(med.iloc[0]))} to {fmt_num(float(med.iloc[-1]))}).",
            f"- The spread (IQR) {spread_word} "
            f"(from {fmt_num(float(iqr.iloc[0]))} to {fmt_num(float(iqr.iloc[-1]))}).",
        ]
    )


def interpret_transition_matrix(result: Any, *, lang: str = "en") -> str:
    """Interpret a transition matrix: overall persistence and the stickiest states."""
    states = list(result.states)
    c = result.counts.to_numpy(dtype=float)
    row_sums = c.sum(axis=1, keepdims=True)
    shares = np.divide(c, row_sums, out=np.zeros_like(c), where=row_sums > 0)
    diag = np.diag(shares)
    persistence = float(np.nanmean(diag))
    stick_i, loose_i = int(np.nanargmax(diag)), int(np.nanargmin(diag))
    return "\n".join(
        [
            f"Across {len(states)} states, the average **persistence** (probability of "
            f"staying in the same state from one period to the next) is {fmt_num(persistence)}.",
            f"- Stickiest: **{states[stick_i]}** (stays with probability "
            f"{fmt_num(float(diag[stick_i]))}).",
            f"- Least sticky: **{states[loose_i]}** (stays with probability "
            f"{fmt_num(float(diag[loose_i]))}).",
        ]
    )


def interpret_within_persistence(result: Any, *, lang: str = "en") -> str:
    """Interpret within-unit persistence (serial correlation of the within component)."""
    rho = float(result.rho)
    scope = (
        "within-unit (after removing each unit's mean)"
        if bool(result.demeaned)
        else "within-unit"
    )
    mag = "strong" if abs(rho) >= 0.7 else "moderate" if abs(rho) >= 0.4 else "weak"
    if rho > 0:
        shape = "positively — high values tend to follow high values (persistence)"
    elif rho < 0:
        shape = "negatively — high values tend to follow low values (mean reversion)"
    else:
        shape = "not at all"
    lines = [
        f"The {scope} serial correlation is rho = {fmt_num(rho)} "
        f"(n = {int(result.n_pairs):,} consecutive pairs): a **{mag}** relationship — this "
        f"period's value relates to the previous one {shape}.",
    ]
    if abs(rho) < 0.1:
        lines.append(
            "Near-zero persistence means the series behaves like noise within units."
        )
    lines += ["", _ASSOC_NOTE]
    return "\n".join(lines)


def interpret_coefficient_plot(result: Any, *, lang: str = "en") -> str:
    """Interpret a coefficient plot: sign, magnitude and whether each CI excludes zero."""
    df = result.df
    first = df["model"].iloc[0]
    sub = df[df["model"] == first]
    n_models = int(df["model"].nunique())
    parts = [
        "This coefficient plot shows each term's estimate with its confidence interval."
    ]
    for _, row in sub.iterrows():
        term = str(row["term"])
        est = float(row["estimate"])
        lo, hi = float(row["ci_lower"]), float(row["ci_upper"])
        excl = (lo > 0) or (hi < 0)
        parts.append(
            f"- **{term}**: estimate {fmt_num(est)} ({sign_word(est)}); its confidence "
            "interval "
            + ("excludes" if excl else "includes")
            + " zero, so it is "
            + ("distinguishable from" if excl else "not distinguishable from")
            + " zero."
        )
    if n_models > 1:
        parts.append(f"(This reads the first of {n_models} models shown side by side.)")
    parts += ["", _ASSOC_NOTE]
    return "\n".join(parts)


def interpret_fixef_plot(result: Any, *, lang: str = "en") -> str:
    """Interpret a fixed-effects plot: the spread of the estimated group intercepts."""
    df = result.df
    dim = str(df["fixef"].iloc[0]) if len(df) else "the group"
    vals = df["value"].to_numpy(dtype=float)
    n = len(vals)
    lo_i, hi_i = int(np.nanargmin(vals)), int(np.nanargmax(vals))
    lo_lvl, hi_lvl = str(df["level"].iloc[lo_i]), str(df["level"].iloc[hi_i])
    rng = float(vals[hi_i] - vals[lo_i])
    sd = float(np.nanstd(vals))
    lines = [
        f"These are the estimated **{dim}** fixed effects — each group's intercept relative "
        f"to the others. Across {n:,} level(s) shown they span {fmt_num(rng)}, from "
        f"**{lo_lvl}** ({fmt_num(float(vals[lo_i]))}) to **{hi_lvl}** "
        f"({fmt_num(float(vals[hi_i]))}); the spread (SD) is {fmt_num(sd)}.",
        "A wide spread means the groups differ substantially in baseline level once the "
        "regressors are held fixed.",
        "",
        _ASSOC_NOTE,
    ]
    return "\n".join(lines)


def interpret_predictions(result: Any, *, lang: str = "en") -> str:
    """Interpret predictions: in-sample fit (r, RMSE) or the range of new-data forecasts."""
    df = result.df
    n = len(df)
    if n == 0:
        return "No predictions were produced."
    if "residual" not in df.columns:
        pred = df["predicted"].to_numpy(dtype=float)
        return (
            f"These are {n:,} fitted predictions on new data, ranging from "
            f"{fmt_num(float(np.nanmin(pred)))} to {fmt_num(float(np.nanmax(pred)))} "
            f"(mean {fmt_num(float(np.nanmean(pred)))}). No actuals are available, so fit "
            "cannot be assessed here."
        )
    actual = df["actual"].to_numpy(dtype=float)
    pred = df["predicted"].to_numpy(dtype=float)
    resid = df["residual"].to_numpy(dtype=float)
    rmse = float(np.sqrt(np.nanmean(resid**2)))
    sd_y = float(np.nanstd(actual))
    with np.errstate(invalid="ignore"):
        r = float(np.corrcoef(actual, pred)[0, 1]) if n > 1 else float("nan")
    quality = (
        "a tight fit"
        if (sd_y > 0 and rmse < 0.5 * sd_y)
        else "a loose fit"
        if (sd_y > 0 and rmse > sd_y)
        else "a moderate fit"
    )
    lines = [
        f"On the {n:,} estimation-sample rows, fitted values track the actual outcome with "
        f"correlation r = {fmt_num(r)} (r-squared = {fmt_num(r * r)}).",
        f"Residuals have a root-mean-square of {fmt_num(rmse)} against an outcome SD of "
        f"{fmt_num(sd_y)} — {quality}.",
        "",
        _ASSOC_NOTE,
    ]
    return "\n".join(lines)


def interpret_joint_test(result: Any, *, lang: str = "en") -> str:
    """Interpret a joint (Wald/F) significance test: the null, the statistic and the verdict."""
    terms = ", ".join(result.hypotheses)
    stat = float(result.statistic)
    p = float(result.p_value)
    dist = str(result.distribution)
    reject = p < 0.05
    verdict = (
        "**reject** the joint null: at least one of these terms is statistically "
        "distinguishable from zero (the set is jointly significant)"
        if reject
        else "**fail to reject** the joint null: the terms are jointly indistinguishable "
        "from zero at the 5% level"
    )
    lines = [
        f"Joint **{dist}-test** that all of [{terms}] equal zero: "
        f"statistic = {fmt_num(stat)}, p = {fmt_num(p)} ({significance_phrase(p)}).",
        f"At the 5% level we {verdict}.",
        "Failing to reject reflects a lack of evidence against the null, not proof that the "
        "terms are exactly zero.",
    ]
    return "\n".join(lines)


_RI_METHOD = {
    "ritest": "randomization inference (permuting the regressor across the sample)",
    "wildboot": "the wild cluster bootstrap (resampling residuals within clusters)",
}


def interpret_robust_inference(result: Any, *, lang: str = "en") -> str:
    """Interpret robust inference: the resampling method, the robust p-value and the CI."""
    method = str(result.method)
    name = _RI_METHOD.get(method, method)
    param = str(result.param)
    est = float(result.estimate)
    p = float(result.p_value)
    reps = int(result.reps)
    lines = [
        f"Robust inference on **{param}** via {name}, over {reps:,} resamples.",
        f"The point estimate is {fmt_num(est)} and its robust p-value is {fmt_num(p)} "
        f"({significance_phrase(p)}).",
    ]
    lo, hi = float(result.conf_int[0]), float(result.conf_int[1])
    if math.isfinite(lo) and math.isfinite(hi):
        excl = (lo > 0) or (hi < 0)
        lines.append(
            f"Its {fmt_num(lo)} to {fmt_num(hi)} interval "
            + ("excludes" if excl else "includes")
            + " zero."
        )
    lines.append(
        "When clusters are few, this resampling p-value is more trustworthy than the naive "
        "large-sample cluster-robust p-value, which tends to overstate precision."
    )
    return "\n".join(lines)


def interpret_panel_view(result: Any, *, lang: str = "en") -> str:
    """Interpret a panel view: treatment timing/balance, or outcome-trajectory coverage."""
    df = result.df
    cols = list(df.columns)
    is_outcome = len(cols) == 3 and not set(df[cols[2]].dropna().unique()) <= {0, 1}
    if is_outcome:
        unit_col, time_col, out_col = cols
        n_units = int(df[unit_col].nunique())
        n_per = int(df[time_col].nunique())
        y = df[out_col].to_numpy(dtype=float)
        return (
            f"Outcome trajectories for **{n_units:,} units** across **{n_per:,} periods**. "
            f"The outcome **{out_col}** ranges from {fmt_num(float(np.nanmin(y)))} to "
            f"{fmt_num(float(np.nanmax(y)))} (mean {fmt_num(float(np.nanmean(y)))}); each "
            "faint line traces one unit over time."
        )
    n_units, n_per = int(df.shape[0]), int(df.shape[1])
    ever = df.eq(1).any(axis=1)
    n_treated = int(ever.sum())
    n_never = n_units - n_treated
    periods = list(df.columns)
    first_periods = []
    for _, row in df[ever].iterrows():
        hits = [c for c in periods if row.get(c) == 1]
        if hits:
            first_periods.append(min(hits))
    lines = [
        f"This treatment quilt covers **{n_units:,} units** over **{n_per:,} periods**: "
        f"{n_treated:,} are ever treated and {n_never:,} are never treated."
    ]
    if first_periods:
        staggered = len(set(first_periods)) > 1
        lines.append(
            f"First treatment ranges from {min(first_periods)} to {max(first_periods)} — "
            "adoption is "
            + ("**staggered** across periods." if staggered else "**simultaneous**.")
        )
    return "\n".join(lines)


def interpret_scatter_plot(result: Any, *, lang: str = "en") -> str:
    """Interpret a scatter plot: the direction and strength of the bivariate association."""
    df = result.df
    xcol, ycol = df.columns[0], df.columns[1]
    x = df[xcol].to_numpy(dtype=float)
    y = df[ycol].to_numpy(dtype=float)
    n = len(df)
    try:
        x_label = result.fig.layout.xaxis.title.text or str(xcol)
        y_label = result.fig.layout.yaxis.title.text or str(ycol)
    except AttributeError:  # pragma: no cover - defensive
        x_label, y_label = str(xcol), str(ycol)
    if n < 3 or np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return (
            f"With {n:,} complete points, the bivariate association between **{x_label}** "
            f"and **{y_label}** cannot be summarised reliably.\n\n" + _ASSOC_NOTE
        )
    r = float(np.corrcoef(x, y)[0, 1])
    slope = float(np.polyfit(x, y, 1)[0])
    mag = "strong" if abs(r) >= 0.7 else "moderate" if abs(r) >= 0.4 else "weak"
    direc = "positive" if r > 0 else "negative" if r < 0 else "essentially no"
    lines = [
        f"Across {n:,} points, **{y_label}** and **{x_label}** show a **{mag}** **{direc}** "
        f"linear association (Pearson r = {fmt_num(r)}).",
        f"A simple line through the cloud has slope {fmt_num(slope)}: each one-unit increase "
        f"in {x_label} goes with {y_label} that is {fmt_num(abs(slope))} "
        f"{direction_word(slope)} on average.",
        "",
        _ASSOC_NOTE,
    ]
    return "\n".join(lines)


def interpret_missing_values(result: Any, *, lang: str = "en") -> str:
    """Interpret a missing-values map: overall completeness and the worst variable/level."""
    df = result.df
    n_rows, n_cols = df.shape
    z = df.to_numpy(dtype=float)
    overall = float(np.nanmean(z)) if z.size else float("nan")
    if overall == 0:
        return (
            f"Across {n_cols:,} variables and {n_rows:,} levels there are no missing "
            "values — the panel is fully complete on these variables."
        )
    col_miss = df.mean(axis=0)
    row_miss = df.mean(axis=1)
    worst_var = str(col_miss.idxmax())
    worst_lvl = str(row_miss.idxmax())
    lines = [
        f"Across {n_cols:,} variables and {n_rows:,} levels, overall "
        f"**{fmt_num(overall * 100)}%** of cells are missing "
        f"(so the data are {fmt_num((1 - overall) * 100)}% complete).",
        f"The variable with the most missingness is **{worst_var}** "
        f"({fmt_num(float(col_miss.max()) * 100)}% missing); the level with the most "
        f"missingness is **{worst_lvl}** ({fmt_num(float(row_miss.max()) * 100)}% missing).",
    ]
    return "\n".join(lines)


def interpret_value_heatmap(result: Any, *, lang: str = "en") -> str:
    """Interpret a value heatmap: grid dimensions, coverage and the value range."""
    df = result.df
    n_units, n_per = df.shape
    z = df.to_numpy(dtype=float)
    total = z.size
    finite = z[np.isfinite(z)]
    if finite.size == 0:
        return (
            f"This heatmap covers {n_units:,} units over {n_per:,} periods but has no "
            "values to read."
        )
    coverage = finite.size / total if total else float("nan")
    lines = [
        f"This heatmap shows one variable over a grid of **{n_units:,} units** by "
        f"**{n_per:,} periods**; {fmt_num(coverage * 100)}% of cells are observed.",
        f"Values range from {fmt_num(float(np.min(finite)))} to "
        f"{fmt_num(float(np.max(finite)))} (mean {fmt_num(float(np.mean(finite)))}). "
        "Colour reveals which units and periods sit high or low.",
    ]
    return "\n".join(lines)
