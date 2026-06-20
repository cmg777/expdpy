"""Explainer for the Correlated Random Effects (Mundlak) estimator."""

from __future__ import annotations

from expdpy.pedagogy._registry import Explainer, register_topic

register_topic(
    Explainer(
        topic="correlated_random_effects",
        title="Correlated Random Effects (Mundlak device)",
        what=(
            "The Mundlak device augments a random-effects model with the unit-level mean of "
            "each time-varying regressor. The coefficient on the original regressor then "
            "equals the fixed-effects (within) estimate, while the coefficient on the mean "
            "measures how much the between-unit association differs from the within-unit one. "
            "A joint test that the mean coefficients are zero is algebraically the Hausman "
            "test, so it makes the fixed-vs-random decision in ordinary, testable terms."
        ),
        when_to_use=(
            "When you want the robustness of fixed effects but also a single, interpretable "
            "model that *includes* time-invariant regressors, or when you want the "
            "fixed-vs-random-effects decision expressed as testable coefficients rather than a "
            "separate test statistic."
        ),
        caveats=(
            "It recovers the within estimate only for the time-varying regressors; truly "
            "time-invariant variables still cannot be separated from the unit effect.",
            "Like random effects, it assumes the *idiosyncratic* error is uncorrelated with "
            "the regressors — the device only relaxes correlation that runs through the unit "
            "mean.",
            "The mean-coefficient test is the Hausman test, so the same small-sample and "
            "power caveats apply.",
        ),
        see_also=("fixed_effects", "random_effects", "hausman"),
        references=(
            "Mundlak (1978), On the Pooling of Time Series and Cross Section Data",
            "Wooldridge, Econometric Analysis of Cross Section and Panel Data, ch. 10",
        ),
    ),
    aliases=("cre", "mundlak"),
)
