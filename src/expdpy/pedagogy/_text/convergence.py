"""Explainers for convergence analysis: β-convergence and σ-convergence.

``beta_convergence`` covers the growth-vs-initial-level regression (unconditional, conditional,
speed and half-life); ``sigma_convergence`` covers the dispersion-over-time view.
"""

from __future__ import annotations

from expdpy.pedagogy._registry import Explainer, register_topic

register_topic(
    Explainer(
        topic="beta_convergence",
        title="Beta convergence",
        what=(
            "β-convergence asks whether units that start *behind* grow *faster* and so catch "
            "up. The test regresses each unit's average growth rate over a horizon on its "
            "**initial level** — canonically the growth of GDP per capita on initial log GDP "
            "per capita. A **negative** slope β is convergence: lower starting points are "
            "associated with faster growth. The slope maps to a structural **speed of "
            "convergence** λ = -ln(1 + β·T) / T (per period) and a **half-life** ln 2 / λ, the "
            "time to close half of an initial gap. **Unconditional** (absolute) convergence "
            "uses the initial level alone; **conditional** convergence adds controls for each "
            "unit's steady-state determinants and, by the Frisch-Waugh-Lovell theorem, reads "
            "the convergence slope from a partial-regression scatter that holds those controls "
            "fixed. The same machinery works for any variable — income, schooling, health."
        ),
        when_to_use=(
            "Use it to summarise catch-up dynamics in a panel: are poorer economies (or "
            "lower-scoring regions/firms) closing the gap, and how fast? Reach for "
            "**unconditional** convergence to describe raw catch-up, and **conditional** "
            "convergence when units have different steady states (different savings, human "
            "capital, institutions) so that catch-up is only expected *relative to* each "
            "unit's own steady state. A rolling-window version shows whether the convergence "
            "speed has itself changed over time."
        ),
        caveats=(
            "β-convergence is a *descriptive association* between growth and an initial level, "
            "not a causal mechanism; regression to the mean and measurement error in the "
            "initial level can both produce a negative slope (Galton's fallacy / Quah's "
            "critique).",
            "The estimate depends on the chosen start and end years and the horizon T — report "
            "them, and prefer a common window across units when comparing.",
            "Conditional convergence is conditional on the controls you include; a different "
            "control set implies a different steady state and can change the slope.",
            "Speed and half-life are only well defined when 1 + β·T > 0; a non-negative slope "
            "(divergence) has no finite positive half-life.",
        ),
        see_also=("fwl", "fixed_effects", "correlation_vs_causation"),
        references=(
            "Barro & Sala-i-Martin, Economic Growth (2nd ed.), ch. 11-12",
            "Sala-i-Martin (1996), 'The Classical Approach to Convergence Analysis', EJ",
        ),
    ),
    aliases=("convergence", "conditional_convergence"),
)

register_topic(
    Explainer(
        topic="sigma_convergence",
        title="Sigma convergence",
        what=(
            "σ-convergence asks whether the *cross-sectional dispersion* of a variable shrinks "
            "over time — whether units become more alike. At each period the dispersion is "
            "measured across units (the **standard deviation**, the **Gini index**, the "
            "**coefficient of variation**), and the test regresses the **log dispersion** on "
            "time: a **negative** slope means dispersion falls by a roughly constant proportion "
            "each period, the hallmark of σ-convergence. It is the distributional complement to "
            "β-convergence: β-convergence (poorer units growing faster) is *necessary but not "
            "sufficient* for σ-convergence, because new shocks can re-spread the distribution "
            "even while laggards catch up (Quah's critique)."
        ),
        when_to_use=(
            "Use it to describe whether a cross-section is compressing or fanning out over time "
            "— income or productivity across regions, test scores across schools, health across "
            "countries. Pair it with β-convergence: β answers 'do laggards grow faster?' while "
            "σ answers 'is the whole distribution narrowing?'. Report several dispersion "
            "measures, since the standard deviation is scale-dependent while the Gini and the "
            "coefficient of variation are scale-free."
        ),
        caveats=(
            "σ-convergence is a *descriptive* statement about the distribution, not a causal "
            "mechanism; a narrowing spread does not say why units converged.",
            "The standard deviation is in the variable's own units and grows with its level; "
            "the Gini and the coefficient of variation are scale-free and often tell a clearer "
            "story — compare them.",
            "Dispersion is only comparable across periods when the set of units is fixed, so a "
            "balanced panel is required; a changing composition can masquerade as convergence "
            "or divergence.",
            "The Gini index is only defined for non-negative values, and the coefficient of "
            "variation is unstable when the mean is near zero.",
        ),
        see_also=("beta_convergence", "fwl", "correlation_vs_causation"),
        references=(
            "Barro & Sala-i-Martin, Economic Growth (2nd ed.), ch. 11",
            "Sala-i-Martin (1996), 'The Classical Approach to Convergence Analysis', EJ",
            "Quah (1993), 'Galton's Fallacy and Tests of the Convergence Hypothesis', SJE",
        ),
    ),
    aliases=("dispersion_convergence", "sigma"),
)
