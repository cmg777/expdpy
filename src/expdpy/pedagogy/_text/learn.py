"""Explainers for the core panel-data identities: first differences, demeaning, dummies."""

from __future__ import annotations

from expdpy.pedagogy._registry import Explainer, register_topic

register_topic(
    Explainer(
        topic="within_transformation",
        title="The within transformation (demeaning)",
        what=(
            "The within transformation subtracts each unit's own time-average from every "
            "variable, so a unit's constant characteristics — including its unobserved fixed "
            "effect — cancel out. Regressing the demeaned outcome on the demeaned regressors "
            "gives the fixed-effects estimate using only variation *within* each unit over "
            "time."
        ),
        when_to_use=(
            "It *is* the fixed-effects estimator: use it whenever you want to control for "
            "stable, unobserved unit characteristics. It is the route most software "
            "(including pyfixest) takes, because it avoids estimating one dummy per unit."
        ),
        caveats=(
            "Anything constant within a unit is demeaned away and cannot be estimated.",
            "On a two-period panel it is identical to first differences; for more periods the "
            "two differ in how they weight the data, though both are consistent.",
            "Demeaning uses a degree of freedom per unit — account for that when computing "
            "standard errors by hand.",
        ),
        see_also=("fixed_effects", "first_differences", "dummy_variables"),
        references=("Wooldridge, Introductory Econometrics, ch. 14",),
    ),
    aliases=("demeaning",),
)

register_topic(
    Explainer(
        topic="first_differences",
        title="First differences",
        what=(
            "First differencing subtracts each unit's previous-period value from its current "
            "value (Δy, Δx). Because a unit's fixed effect is the same in both periods, it "
            "cancels in the difference, so a regression of Δy on Δx removes the unit effect — "
            "another route to the within slope."
        ),
        when_to_use=(
            "A natural choice with two periods (where it equals the within estimator), or "
            "when the idiosyncratic errors are highly persistent, where differencing can be "
            "more efficient than demeaning."
        ),
        caveats=(
            "Differencing drops the first period of every unit, costing observations.",
            "It coincides with the within estimator only for two periods; for more than two "
            "the two estimators differ in finite samples.",
            "Differencing can amplify measurement error in the regressor.",
        ),
        see_also=("within_transformation", "fixed_effects", "dummy_variables"),
        references=("Wooldridge, Introductory Econometrics, ch. 13-14",),
    ),
    aliases=("fd",),
)

register_topic(
    Explainer(
        topic="dummy_variables",
        title="Least-squares dummy variables (LSDV)",
        what=(
            "LSDV estimates fixed effects directly by adding one dummy variable per unit to "
            "an OLS regression. The Frisch-Waugh-Lovell theorem guarantees the slope on the "
            "regressors is identical to the within (demeaned) estimate — the dummies and the "
            "demeaning do the same job."
        ),
        when_to_use=(
            "As the most transparent way to *see* what fixed effects do, and when you "
            "actually want the estimated unit intercepts. For many units the demeaning route "
            "is far cheaper for the same slopes."
        ),
        caveats=(
            "With many units LSDV estimates a huge number of parameters and is slow; the "
            "within transformation gives identical slopes at a fraction of the cost.",
            "The estimated unit intercepts are noisy when each unit has few observations "
            "(the incidental-parameters problem).",
        ),
        see_also=("within_transformation", "fixed_effects", "fwl"),
        references=("Wooldridge, Introductory Econometrics, ch. 14",),
    ),
    aliases=("lsdv",),
)
