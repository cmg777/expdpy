"""Explainers for the regression-family methods (OLS, fixed effects, clustering, FWL)."""

from __future__ import annotations

from expdpy.pedagogy._registry import Explainer, register_topic

register_topic(
    Explainer(
        topic="ols",
        title="Ordinary least squares (OLS)",
        what=(
            "OLS fits a straight-line relationship between an outcome and one or more "
            "regressors by minimizing the sum of squared residuals. Each coefficient is the "
            "average change in the outcome associated with a one-unit increase in that "
            "regressor, holding the others fixed."
        ),
        when_to_use=(
            "As the default workhorse for a continuous outcome and a first look at how "
            "variables move together. It is fast, transparent, and the baseline every other "
            "estimator is compared against."
        ),
        caveats=(
            "Coefficients are associations, not causal effects, unless a research design "
            "(randomization, instrument, difference-in-differences) justifies a causal read.",
            "Default standard errors assume independent, equal-variance errors — rarely true "
            "in panel data, where clustered standard errors are usually needed.",
            "Sensitive to outliers and to omitted variables correlated with the regressors.",
        ),
        see_also=("fixed_effects", "clustered_se", "fwl"),
        references=("Wooldridge, Introductory Econometrics, ch. 2-4",),
    )
)

register_topic(
    Explainer(
        topic="fixed_effects",
        title="Fixed effects",
        what=(
            "Fixed effects add a separate intercept for every level of a grouping variable "
            "(e.g. each country, each year). This absorbs all *time-invariant* differences "
            "across groups, so the remaining coefficients are identified from variation "
            "*within* a group over time rather than from differences *between* groups."
        ),
        when_to_use=(
            "In panel data, to control for stable, unobserved characteristics of units (country "
            "institutions, firm culture) and for common shocks in a period (year effects). "
            "Two-way (unit + time) fixed effects are the standard panel specification."
        ),
        caveats=(
            "Fixed effects cannot estimate the effect of anything constant within the group "
            "(a country's region, a person's sex) — that variation is absorbed.",
            "They control only for *unobserved* confounders that are constant within the "
            "group; time-varying confounders remain a threat.",
            "Many groups with few observations each can leave little within-variation, "
            "inflating standard errors.",
        ),
        see_also=("ols", "clustered_se", "fwl"),
        references=("Wooldridge, Introductory Econometrics, ch. 13-14",),
    ),
    aliases=("fe",),
)

register_topic(
    Explainer(
        topic="clustered_se",
        title="Clustered standard errors",
        what=(
            "Clustering allows the regression errors to be correlated *within* groups "
            "(e.g. repeated observations of the same country) while remaining independent "
            "*across* groups. It changes the standard errors and therefore the t-statistics "
            "and p-values — never the coefficients themselves."
        ),
        when_to_use=(
            "Whenever observations are grouped and likely correlated within group: panels "
            "(cluster by unit), repeated cross-sections, or treatment assigned at a group "
            "level. The cluster should match the level at which shocks are correlated."
        ),
        caveats=(
            "With few clusters (rule of thumb: fewer than ~40) the asymptotics are unreliable "
            "and a wild cluster bootstrap is safer.",
            "Clustering at too fine a level understates uncertainty; choose the level where "
            "correlation actually lives.",
            "Clustered standard errors are usually larger than default (iid) ones — if yours "
            "shrink, re-check the cluster choice.",
        ),
        see_also=("fixed_effects", "ols"),
        references=(
            "Cameron & Miller (2015), A Practitioner's Guide to Cluster-Robust Inference",
        ),
    ),
    aliases=("clustering",),
)

register_topic(
    Explainer(
        topic="omitted_variable_bias",
        title="Omitted-variable bias",
        what=(
            "When a regression leaves out a variable that both belongs in the model and is "
            "correlated with an included regressor, the coefficient on that regressor absorbs "
            "part of the omitted variable's influence and is biased. The bias equals the "
            "omitted variable's effect times how strongly it co-moves with the included "
            "regressor."
        ),
        when_to_use=(
            "A diagnostic mindset for any observational regression: ask what is left out and "
            "whether it correlates with your regressor of interest."
        ),
        caveats=(
            "The sign of the bias follows the product of (effect of the omitted variable) and "
            "(its correlation with the included regressor).",
            "Adding controls reduces bias only if they are the right controls — conditioning "
            "on a collider or a mediator can introduce new bias.",
        ),
        see_also=("ols", "fixed_effects", "correlation_vs_causation"),
        references=("Wooldridge, Introductory Econometrics, ch. 3",),
    ),
    aliases=("ovb",),
)

register_topic(
    Explainer(
        topic="fwl",
        title="Frisch-Waugh-Lovell (partial regression)",
        what=(
            "The Frisch-Waugh-Lovell theorem says a multivariate coefficient can be recovered "
            "in two steps: residualize both the outcome and the focal regressor on all the "
            "*other* regressors (and fixed effects), then regress one residual on the other. "
            "The slope of that residual scatter equals the focal coefficient in the full "
            "model — which is exactly what the partial-regression plot shows."
        ),
        when_to_use=(
            "To *visualize* one coefficient from a multivariate model: the plot reveals "
            "non-linearity, influential points and the strength of the partial relationship "
            "that a single number in a table hides."
        ),
        caveats=(
            "If the controls and fixed effects absorb nearly all of the focal regressor's "
            "variation, the residualized slope is poorly identified.",
            "The plotted confidence band is the simple residual-regression band; it can "
            "differ from the (clustered) standard error reported in the table.",
        ),
        see_also=("ols", "fixed_effects"),
        references=("Lovell (1963); Frisch & Waugh (1933)",),
    )
)

register_topic(
    Explainer(
        topic="instrumental_variables",
        title="Instrumental variables (2SLS)",
        what=(
            "An instrumental variable is a source of variation in an endogenous regressor that "
            "is unrelated to the outcome's error term. Two-stage least squares (2SLS) uses only "
            "the part of the regressor predicted by the instruments, isolating variation that "
            "is free of the confounding that biases ordinary least squares. The first stage "
            "regresses the endogenous regressor on the instruments; the second regresses the "
            "outcome on its fitted values."
        ),
        when_to_use=(
            "When a regressor is endogenous — correlated with the error through reverse "
            "causation, omitted variables or measurement error — and you have instruments that "
            "shift the regressor but are otherwise unrelated to the outcome (e.g. settler "
            "mortality for institutions, lagged weather for local economic activity)."
        ),
        caveats=(
            "Instruments must be relevant: a weak first stage (rule of thumb, first-stage F "
            "below about 10) makes 2SLS badly biased and its standard errors unreliable.",
            "Instruments must satisfy the exclusion restriction — they may influence the "
            "outcome only through the instrumented regressor — which the data cannot verify.",
            "Under heterogeneous responses, 2SLS recovers a local estimand for the units the "
            "instruments move (the compliers), not a population-wide average.",
        ),
        see_also=("ols", "fixed_effects", "correlation_vs_causation"),
        references=(
            "Wooldridge, Introductory Econometrics, ch. 15; Angrist & Pischke (2009), ch. 4",
        ),
    ),
    aliases=("iv", "2sls"),
)

register_topic(
    Explainer(
        topic="marginal_effects",
        title="Marginal effects & interactions",
        what=(
            "In a model with an interaction term (``focal * moderator``), the slope of the "
            "focal regressor is not one number: it is ``b_focal + b_interaction * moderator``, "
            "a line traced across the moderator's range. The marginal effect is that partial "
            "derivative, and its confidence band (via the delta method) widens away from the "
            "centre of the data."
        ),
        when_to_use=(
            "Whenever a model includes an interaction and you want to read how the focal "
            "regressor's association with the outcome changes as the moderator varies — and "
            "where, if anywhere, that association is distinguishable from zero."
        ),
        caveats=(
            "The marginal effect is only meaningful over the moderator's observed range; "
            "extrapolating beyond it is unsupported.",
            "Significance varies along the moderator — a marginal effect can be significant at "
            "some values and not others, so read the whole band, not one point.",
            "Interactions describe associations; a causal reading still needs a research design.",
        ),
        see_also=("ols", "fwl", "correlation_vs_causation"),
        references=(
            "Brambor, Clark & Golder (2006); Wooldridge, Introductory Econometrics, ch. 6",
        ),
    ),
    aliases=("interaction", "interactions", "marginal_effect"),
)

register_topic(
    Explainer(
        topic="random_effects",
        title="Random effects",
        what=(
            "The random-effects estimator treats each unit's effect as a random draw "
            "uncorrelated with the regressors. It uses both within- and between-unit "
            "variation, making it more efficient than fixed effects — but only valid when "
            "that no-correlation assumption holds."
        ),
        when_to_use=(
            "When unit effects are plausibly unrelated to the regressors (e.g. random "
            "sampling of units) and you want to estimate effects of time-invariant variables, "
            "which fixed effects cannot."
        ),
        caveats=(
            "If the unit effects are correlated with the regressors, random effects is biased "
            "— the Hausman test checks exactly this against fixed effects.",
            "More efficient than fixed effects only when its assumption holds.",
        ),
        see_also=("fixed_effects", "hausman"),
        references=("Wooldridge, Introductory Econometrics, ch. 14",),
    ),
    aliases=("re",),
)

register_topic(
    Explainer(
        topic="hausman",
        title="Hausman test (fixed vs random effects)",
        what=(
            "The Hausman test compares the fixed-effects and random-effects coefficient "
            "estimates. Under the random-effects assumption both are consistent and should "
            "agree; a large difference signals that the unit effects are correlated with the "
            "regressors, so random effects is biased."
        ),
        when_to_use=(
            "To choose between fixed and random effects: reject the null (large statistic, "
            "small p-value) and prefer fixed effects; fail to reject and random effects is "
            "admissible (and more efficient)."
        ),
        caveats=(
            "Failing to reject is not proof that random effects is correct — it may just be "
            "low power.",
            "The covariance difference can be non-positive-definite in finite samples; a "
            "generalized inverse is used, which makes the statistic approximate.",
        ),
        see_also=("fixed_effects", "random_effects"),
        references=("Hausman (1978); Wooldridge, Introductory Econometrics, ch. 14",),
    )
)
