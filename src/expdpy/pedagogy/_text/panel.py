"""Explainers for panel-structure concepts: within/between variation, balance, transitions."""

from __future__ import annotations

from expdpy.pedagogy._registry import Explainer, register_topic

register_topic(
    Explainer(
        topic="within_between_variation",
        title="Within and between variation",
        what=(
            "Any panel variable varies in two ways: **between** units (some units sit higher "
            "than others on average) and **within** units (a unit moves up and down over "
            "time). The xtsum decomposition reports the standard deviation of each component; "
            "the within-vs-between scatter shows how a relationship looks through each lens."
        ),
        when_to_use=(
            "Before choosing an estimator: if almost all of a regressor's variation is "
            "between units, a fixed-effects (within) model has little signal to work with. "
            "Comparing the between and within slopes also reveals whether unit-level "
            "confounders are distorting a pooled relationship."
        ),
        caveats=(
            "The decomposition is exact only for balanced panels; for unbalanced panels the "
            "between/within split is approximate.",
            "A variable that is constant within units (e.g. a country's region) has zero "
            "within variation and drops out of a fixed-effects model.",
        ),
        see_also=("fixed_effects", "descriptive_stats"),
        references=(
            "Wooldridge, Econometric Analysis of Cross Section and Panel Data, ch. 10",
        ),
    ),
    aliases=("xtsum", "between_within"),
)

register_topic(
    Explainer(
        topic="panel_structure",
        title="Panel structure (balance and gaps)",
        what=(
            "A panel is **balanced** when every unit is observed in every period, and "
            "**unbalanced** otherwise. Gaps (missing interior periods) and ragged start/end "
            "dates change how many observations each unit contributes."
        ),
        when_to_use=(
            "As a first diagnostic on any panel: to see how many units and periods you have, "
            "whether attrition or late entry is present, and how much the within estimator "
            "has to work with per unit."
        ),
        caveats=(
            "Unbalanced panels are usually fine to estimate, but if the reason data is "
            "missing relates to the outcome (selection), estimates can be biased.",
            "When the set of units changes across periods, a raw time trend mixes real change "
            "with compositional change.",
        ),
        see_also=("within_between_variation", "time_trends"),
        references=("Wooldridge, Introductory Econometrics, ch. 13-14",),
    ),
    aliases=("balanced_panel", "panel_balance"),
)

register_topic(
    Explainer(
        topic="transition_matrix",
        title="Transition matrix",
        what=(
            "A transition matrix counts how often units move from one state to another "
            "between consecutive periods. Row-normalized, cell (i, j) is the probability of "
            "being in state j next period given state i this period; the diagonal measures "
            "persistence (staying put). Continuous variables are first binned (e.g. into "
            "quantiles) to form states."
        ),
        when_to_use=(
            "To study mobility and persistence: income or rating mobility, whether countries "
            "stay in the same development tier, how sticky a categorical status is over time."
        ),
        caveats=(
            "Transitions are only counted between consecutive observed periods; gaps are "
            "skipped, which can understate movement.",
            "Quantile bins are computed on the pooled distribution, so 'moving up a bin' is "
            "relative to all units and periods, not an absolute threshold.",
        ),
        see_also=("within_between_variation", "descriptive_stats"),
        references=("Markov chains; Quah (1993) on income distribution dynamics",),
    ),
    aliases=("markov", "transitions"),
)
