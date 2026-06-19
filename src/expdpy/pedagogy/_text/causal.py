"""Explainers for staggered difference-in-differences and event studies."""

from __future__ import annotations

from expdpy.pedagogy._registry import Explainer, register_topic

register_topic(
    Explainer(
        topic="event_study",
        title="Event study / staggered difference-in-differences",
        what=(
            "An event study plots the outcome relative to the timing of treatment: "
            "coefficients at each *event time* (periods before and after treatment) measure "
            "how the treated and control groups diverge, with the period just before "
            "treatment (t = -1) as the baseline. Coefficients before t = 0 probe parallel "
            "trends; coefficients after t = 0 trace the dynamic treatment effect."
        ),
        when_to_use=(
            "When treatment turns on at different times for different units (staggered "
            "adoption) and you want both a pre-treatment check and the path of the effect "
            "over time. Modern estimators (Gardner's did2s, Sun-Abraham, local-projections "
            "DiD) are robust to the bias that plagues a naive two-way fixed-effects event "
            "study under heterogeneous effects."
        ),
        caveats=(
            "Non-zero pre-treatment coefficients warn that the parallel-trends assumption is "
            "questionable — interpret the post-treatment path cautiously.",
            "Two-way fixed-effects (twfe) event studies are biased when effects vary across "
            "cohorts or over time; prefer did2s, saturated (Sun-Abraham) or lpdid.",
            "The never-treated or not-yet-treated units form the control group — make sure "
            "the cohort variable encodes never-treated correctly.",
        ),
        see_also=("parallel_trends", "fixed_effects", "clustered_se"),
        references=(
            "Callaway & Sant'Anna (2021); Sun & Abraham (2021); Gardner (2022)",
        ),
    ),
    aliases=("did", "diff_in_diff", "staggered_did"),
)

register_topic(
    Explainer(
        topic="parallel_trends",
        title="The parallel-trends assumption",
        what=(
            "Difference-in-differences identifies a treatment effect by assuming that, absent "
            "treatment, treated and control groups would have followed *parallel* paths. The "
            "treatment effect is the deviation of the treated group from that counterfactual "
            "parallel path."
        ),
        when_to_use=(
            "It underlies every DiD and event-study estimate. The pre-treatment coefficients "
            "of an event study are the main diagnostic: if they are flat and near zero, the "
            "assumption is more credible."
        ),
        caveats=(
            "Parallel trends is an assumption about an unobserved counterfactual — flat "
            "pre-trends are supportive evidence, not proof.",
            "Anticipation effects (units reacting before treatment) violate it and show up as "
            "non-zero coefficients just before t = 0.",
        ),
        see_also=("event_study", "fixed_effects"),
        references=("Angrist & Pischke, Mostly Harmless Econometrics, ch. 5",),
    )
)
