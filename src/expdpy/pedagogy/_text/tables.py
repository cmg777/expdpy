"""Explainers for descriptive summaries and time trends."""

from __future__ import annotations

from expdpy.pedagogy._registry import Explainer, register_topic

register_topic(
    Explainer(
        topic="descriptive_stats",
        title="Descriptive statistics",
        what=(
            "A descriptive table summarizes each variable's central tendency (mean, median), "
            "spread (standard deviation, quartiles) and range (min, max), plus the number of "
            "non-missing observations. It is the first sanity check on any dataset."
        ),
        when_to_use=(
            "At the very start of an analysis: to spot implausible values, gauge skewness "
            "(mean far from median), check units, and see how much data is missing."
        ),
        caveats=(
            "The mean is sensitive to outliers; compare it with the median to judge skew.",
            "Means computed over a panel mix within-unit and between-unit variation — they do "
            "not describe a single unit's typical value.",
        ),
        see_also=("winsorize", "time_trends"),
        references=("Wooldridge, Introductory Econometrics, ch. 1",),
    )
)

register_topic(
    Explainer(
        topic="time_trends",
        title="Time trends",
        what=(
            "A trend graph plots the average of a variable in each period, with a standard-"
            "error band, to show how it evolves over time across the panel."
        ),
        when_to_use=(
            "To see the overall time pattern (growth, decline, cycles, structural breaks) "
            "before modelling, and to motivate the inclusion of year fixed effects."
        ),
        caveats=(
            "A cross-sectional average over time can hide divergent paths of individual units "
            "(Simpson's paradox); compare with a by-group trend.",
            "If the panel is unbalanced, the set of units in the average changes across "
            "periods, which can drive apparent trends.",
        ),
        see_also=("descriptive_stats", "fixed_effects"),
        references=("Wooldridge, Introductory Econometrics, ch. 10",),
    )
)
