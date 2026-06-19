"""Explainers for outlier treatment (winsorizing vs truncating)."""

from __future__ import annotations

from expdpy.pedagogy._registry import Explainer, register_topic

register_topic(
    Explainer(
        topic="winsorize",
        title="Winsorizing",
        what=(
            "Winsorizing caps extreme values at a chosen percentile (e.g. the 1st and 99th): "
            "values beyond the cutoff are *replaced* by the cutoff value rather than removed. "
            "The number of observations is preserved."
        ),
        when_to_use=(
            "When a few extreme values would dominate means, regressions or correlations, but "
            "you want to keep every observation in the sample."
        ),
        caveats=(
            "It is a discretionary transformation — report the percentile you used and check "
            "that results are not driven by it.",
            "Caps distort the tails of the distribution; do not winsorize variables whose tails "
            "are the object of study.",
        ),
        see_also=("truncate",),
        references=("Tukey (1962), The Future of Data Analysis",),
    )
)

register_topic(
    Explainer(
        topic="truncate",
        title="Truncating",
        what=(
            "Truncating sets values beyond a chosen percentile to missing (NaN), effectively "
            "*dropping* the extreme observations from any analysis that needs that variable. "
            "Unlike winsorizing, it reduces the sample."
        ),
        when_to_use=(
            "When extreme values are likely data errors rather than genuine observations, and "
            "removing them is more defensible than capping them."
        ),
        caveats=(
            "Reduces the sample size and can introduce selection bias if the dropped points "
            "are not random.",
            "Different variables truncated separately can leave different samples — watch the "
            "observation counts.",
        ),
        see_also=("winsorize",),
        references=("Wooldridge, Introductory Econometrics, ch. 9",),
    )
)
