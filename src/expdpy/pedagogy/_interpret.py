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

from expdpy.pedagogy._format import (
    direction_word,
    fmt_num,
    significance_phrase,
)

__all__ = [
    "interpret_correlation",
    "interpret_cre",
    "interpret_descriptive",
    "interpret_estimation",
    "interpret_event_study",
    "interpret_fwl",
    "interpret_regression",
    "interpret_sandbox",
    "interpret_trend",
]

_ASSOC_NOTE = (
    "_These are associations, not causal effects. A causal reading needs a research "
    "design — see `explain('correlation_vs_causation')`._"
)
_MAX_VARS = 6


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
