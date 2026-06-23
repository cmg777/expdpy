"""Explainer for the Kuznets curve and its extended 'Kuznets waves' generalization.

``kuznets_waves`` covers the polynomial (up to quartic) inequality-development relationship
estimated side by side under pooled OLS, the between estimator and the within (two-way
fixed-effects) estimator, with Frisch-Waugh-Lovell partial-residual plots.
"""

from __future__ import annotations

from expdpy.pedagogy._registry import Explainer, register_topic

register_topic(
    Explainer(
        topic="kuznets_waves",
        title="Kuznets waves (extended Kuznets curve)",
        what=(
            "Kuznets (1955) conjectured an **inverted-U** relationship between income inequality "
            "and economic development: as an economy industrializes, inequality first rises "
            "(labour shifts from agriculture to higher-paid industry at uneven speeds), then "
            "falls (education and redistribution spread the gains). The classic test regresses an "
            "inequality measure (a Gini) on log GDP per capita and its **square**; a positive "
            "linear term with a negative quadratic term traces the inverted-U. The **Kuznets "
            "waves** hypothesis generalizes this: the relationship may be far more nonlinear, so "
            "the development term enters up to the **fourth power** — "
            "``gini = f(g, g^2, g^3, g^4)`` with ``g = log GDP per capita``. A cubic admits an "
            "N-shape (inequality falls then rises again as a post-industrial, skill-biased phase "
            "sets in); a quartic admits a full **wave** with up to three turning points. "
            "Estimating the same polynomial under three panel estimators disentangles where the "
            "shape lives: **pooled OLS** uses all variation at once, the **between** estimator "
            "compares country averages (the cross-country curve), and the **within** "
            "(two-way fixed-effects) estimator uses only within-country movements net of common "
            "year shocks. Frisch-Waugh-Lovell partial-residual plots show the fitted wave after "
            "covariates and fixed effects are partialled out."
        ),
        when_to_use=(
            "Use it to test whether inequality and development trace an inverted-U, an N-shape or "
            "a richer wave across a panel of countries or regions, and to check whether that "
            "shape is a cross-sectional phenomenon (countries at different stages differ) or a "
            "within-country dynamic (a single country's inequality rises then falls as it "
            "develops). Comparing the pooled, between and within estimators is the point: if the "
            "between curve is hump-shaped but the within curve is flat, the 'curve' is really a "
            "comparison of different countries, not a path any one country travels. Raise the "
            "polynomial degree only when theory or the data motivate the extra turning points — "
            "a higher-order term that is statistically indistinguishable from zero is just "
            "overfitting."
        ),
        caveats=(
            "The Kuznets curve is a *descriptive* pattern, not a causal mechanism; omitted "
            "determinants (institutions, technology, trade, redistribution) can generate the "
            "same shape.",
            "A significant negative quadratic term is necessary but not sufficient for an "
            "inverted-U: the implied peak must lie inside the observed range of development, "
            "otherwise the curve is effectively monotonic over the data.",
            "High-order polynomials are unstable at the edges of the data and sensitive to a few "
            "extreme units; read the turning points only where the data are dense, and prefer "
            "the lowest degree that fits.",
            "The between and within estimators answer different questions and need not agree; the "
            "within (fixed-effects) curve discards all cross-country variation, so it can be flat "
            "even when the between curve is strongly hump-shaped (and vice versa).",
            "The specific inequality measure matters: a Gini, a top-income share and a Palma "
            "ratio can trace different curvature on the same data.",
        ),
        see_also=(
            "fwl",
            "fixed_effects",
            "within_between_variation",
            "correlation_vs_causation",
        ),
        references=(
            "Kuznets (1955), 'Economic Growth and Income Inequality', American Economic "
            "Review 45(1): 1-28",
            "Gallup (2012), 'Is there a Kuznets Curve?', Portland State University",
            "Palma (2011), 'Homogeneous middles vs. heterogeneous tails', Development and "
            "Change 42(1)",
        ),
    ),
    aliases=("kuznets", "kuznets_curve", "inverted_u"),
)
