"""The analysis pages and the ``st.navigation`` page set.

Each page reads the active prepared sample (stashed by the sidebar) and renders the
components assigned to it, reusing the Plotly compute helpers
(:mod:`expdpy.streamlit_app._components`) and the native table renderers
(:mod:`expdpy.streamlit_app._render`). Pages whose components are all unavailable for the
current data (e.g. time-series views on cross-sectional data) are omitted from the navigation.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import streamlit as st

from expdpy._theme import PLOTLY_CONFIG
from expdpy.streamlit_app import _components as comp
from expdpy.streamlit_app import _render as render
from expdpy.streamlit_app import _widgets as w
from expdpy.streamlit_app._sidebar import Active, get_active

__all__ = ["build_pages", "selected_specs"]

# A page spec is (title, icon, url, render_fn, gate). The gate is None (always shown),
# a list of component names, or a callable ``(active) -> bool`` (e.g. panel-structure pages).
PageGate = list[str] | Callable[[Active], bool] | None
PageSpec = tuple[str, str, str, Callable[[], None], PageGate]


def _active_or_stop() -> Active:
    active = get_active()
    if active is None or active.sample is None or len(active.sample) == 0:
        st.info("Upload a data file or pick a dataset in the sidebar to begin.")
        st.stop()
    assert active is not None  # st.stop() halts above; narrows the type for mypy
    return active


def _has(active: Active, name: str) -> bool:
    return name in active.active_components


def _numeric(active: Active) -> list[str]:
    return active.var_cats.numeric_logical or ["None"]


def _factors(active: Active) -> list[str]:
    return active.var_cats.grouping or ["None"]


def _fe_choices(active: Active) -> list[str]:
    return active.var_cats.fe_choices or ["None"]


# --------------------------------------------------------------------------------- pages ---
def page_overview() -> None:
    """Sample preview, descriptive statistics, extreme observations, missing values."""
    active = _active_or_stop()
    st.header("Overview & Data")
    st.caption(
        f"**{active.source_name}** — {len(active.sample):,} rows, "
        f"{active.sample.shape[1]} columns after subsetting / outlier treatment."
    )
    with st.expander("Preview the analysis sample", expanded=False):
        st.dataframe(active.sample.head(100), width="stretch", hide_index=True)

    if _has(active, "descriptive_table"):
        st.subheader("Descriptive statistics")
        render.render_descriptive(active.sample)
    if _has(active, "ext_obs"):
        st.subheader("Extreme observations")
        var = w.selectbox("Variable", _numeric(active), key="ext_obs_var")
        render.render_ext_obs(active.sample, var)
    if _has(active, "missing_values"):
        st.subheader("Missing values")
        render.render_plotly(lambda: comp.missing(active.sample, active.time))


def page_distributions() -> None:
    """Histogram and category bar chart."""
    active = _active_or_stop()
    st.header("Distributions")
    if _has(active, "histogram"):
        st.subheader("Histogram")
        var = w.selectbox("Variable", _numeric(active), key="hist_var")
        bins = w.slider("Bins", 5, 100, key="hist_nr_of_breaks", default=20)
        render.render_plotly(lambda: comp.histogram(active.sample, var, int(bins)))
    if _has(active, "bar_chart"):
        st.subheader("Bar chart")
        var = w.selectbox("Variable", _factors(active), key="bar_chart_var1")
        render.render_plotly(lambda: comp.bar_chart(active.sample, var))


def page_by_group() -> None:
    """By-group bar, violin and trend graphs."""
    active = _active_or_stop()
    st.header("By group")
    if _has(active, "by_group_bar_graph"):
        st.subheader("Group means (bar)")
        byvar = w.selectbox("Group by", _factors(active), key="bgbg_byvar")
        var = w.selectbox("Variable", _numeric(active), key="bgbg_var")
        render.render_plotly(lambda: comp.by_group_bar(active.sample, byvar, var))
    if _has(active, "by_group_violin_graph"):
        st.subheader("Distribution by group (violin)")
        byvar = w.selectbox("Group by", _factors(active), key="bgvg_byvar")
        var = w.selectbox("Variable", _numeric(active), key="bgvg_var")
        render.render_plotly(lambda: comp.by_group_violin(active.sample, byvar, var))
    if _has(active, "by_group_trend_graph"):
        st.subheader("Group means over time")
        byvar = w.selectbox("Group by", _factors(active), key="bgtg_byvar")
        var = w.selectbox("Variable", _numeric(active), key="bgtg_var")
        render.render_plotly(
            lambda: comp.by_group_trend(active.sample, active.time, byvar, var)
        )


def page_trends() -> None:
    """Variable trends and quantile trends over the panel's time dimension."""
    active = _active_or_stop()
    st.header("Trends")
    if _has(active, "trend_graph"):
        st.subheader("Trend graph")
        variables = [
            w.selectbox(
                f"Variable {i}",
                _numeric(active),
                key=f"trend_graph_var{i}",
                none=(i > 1),
            )
            for i in (1, 2, 3)
        ]
        render.render_plotly(lambda: comp.trend(active.sample, active.time, variables))
    if _has(active, "quantile_trend_graph"):
        st.subheader("Quantile trend graph")
        var = w.selectbox("Variable", _numeric(active), key="quantile_trend_graph_var")
        render.render_plotly(
            lambda: comp.quantile_trend(active.sample, active.time, var)
        )


def page_correlations() -> None:
    """Correlation table + heatmap and the scatter plot."""
    active = _active_or_stop()
    st.header("Correlations & Scatter")
    if _has(active, "corrplot"):
        st.subheader("Correlations")
        render.render_correlation(active.sample)
    if _has(active, "scatter_plot"):
        st.subheader("Scatter plot")
        nums = _numeric(active)
        c1, c2 = st.columns(2)
        with c1:
            x = w.selectbox("X", nums, key="scatter_x")
            color = w.selectbox(
                "Color", _factors(active) + nums, key="scatter_color", none=True
            )
        with c2:
            y = w.selectbox(
                "Y", nums, key="scatter_y", default=nums[1] if len(nums) > 1 else None
            )
            size = w.selectbox("Size", nums, key="scatter_size", none=True)
        loess = w.checkbox("LOESS smoother", key="scatter_loess", default=True)
        render.render_plotly(
            lambda: comp.scatter(active.sample, x, y, color, size, loess)
        )


def page_regression() -> None:
    """OLS regression with fixed effects and clustered standard errors."""
    active = _active_or_stop()
    st.header("Regression")
    if not _has(active, "regression"):
        st.info("The regression component is disabled for this app.")
        return
    y = w.selectbox("Dependent variable", _numeric(active), key="reg_y")
    xs = w.multiselect("Independent variables", _numeric(active), key="reg_x")
    c1, c2 = st.columns(2)
    with c1:
        fe1 = w.selectbox(
            "Fixed effect 1", _fe_choices(active), key="reg_fe1", none=True
        )
    with c2:
        fe2 = w.selectbox(
            "Fixed effect 2", _fe_choices(active), key="reg_fe2", none=True
        )

    cl_opts = ["1", "2", "3"]
    cur = str(st.session_state.get("cluster", "1"))
    st.session_state["cluster"] = cur if cur in cl_opts else "1"
    choice = st.selectbox(
        "Cluster standard errors",
        cl_opts,
        key="cluster",
        format_func=lambda k: {"1": "None", "2": "FE 1", "3": "FE 1 + FE 2"}[k],
    )
    clusters = w.cluster_vars(choice, fe1, fe2)
    render.render_regression(active.sample, y, list(xs), [fe1, fe2], clusters)

    valid_xs = [x for x in xs if x not in (None, "", "None")]
    if valid_xs:
        st.divider()
        st.subheader("Frisch-Waugh-Lovell plot")
        st.caption(
            "Residualizes the dependent variable and the focal regressor on the *other* "
            "regressors **and** the fixed effects above, then plots the two residuals. "
            "The fitted slope equals the focal coefficient in the table above."
        )
        focal = w.selectbox(
            "Focal variable",
            valid_xs,
            key="fwl_focal",
            help="Its coefficient is the FWL slope; the remaining regressors become controls.",
        )
        render.render_plotly(
            lambda: comp.fwl_plot(
                active.sample, y, valid_xs, focal, [fe1, fe2], clusters
            )
        )


# ----------------------------------------------------------------------------- navigation ---
# (title, icon, url_path, function, components that justify showing the page)
_PAGE_SPECS: list[PageSpec] = [
    ("Overview & Data", "🏠", "overview", page_overview, None),
    (
        "Distributions",
        "📊",
        "distributions",
        page_distributions,
        ["histogram", "bar_chart"],
    ),
    (
        "By group",
        "👥",
        "by_group",
        page_by_group,
        ["by_group_bar_graph", "by_group_violin_graph", "by_group_trend_graph"],
    ),
    ("Trends", "📈", "trends", page_trends, ["trend_graph", "quantile_trend_graph"]),
    (
        "Correlations & Scatter",
        "🔗",
        "correlations",
        page_correlations,
        ["corrplot", "scatter_plot"],
    ),
    ("Regression", "🧮", "regression", page_regression, ["regression"]),
]


def page_sandboxes() -> None:
    """Interactive teaching demos that simulate data — no dataset required."""
    from expdpy import (
        learn_beta_convergence,
        learn_clustering_se,
        learn_first_differences,
        learn_omitted_variable_bias,
        learn_pooled_vs_fixed_effects,
        learn_within_vs_lsdv,
    )

    st.header("Concept sandboxes")
    st.caption(
        "Simulated demonstrations — turn the knobs to see each concept in action. "
        "These need no dataset."
    )
    tabs = st.tabs(
        [
            "Omitted-variable bias",
            "Pooled vs fixed effects",
            "Clustered standard errors",
            "First differences",
            "Within vs LSDV",
            "Beta convergence",
        ]
    )
    with tabs[0]:
        corr = st.slider(
            "Correlation between x and the omitted z", 0.0, 0.95, 0.6, 0.05
        )
        bz = st.slider("Effect of the omitted z", -2.0, 2.0, 1.0, 0.25)
        res = learn_omitted_variable_bias(corr_xz=corr, beta_z=bz)
        st.plotly_chart(res.fig, width="stretch", config=PLOTLY_CONFIG)
        st.markdown(res.interpret())
        with st.expander("❓ What is this?"):
            st.markdown(res.explain().to_markdown())
    with tabs[1]:
        uc = st.slider(
            "Correlation between x and the unit effect", 0.0, 0.95, 0.8, 0.05
        )
        res = learn_pooled_vs_fixed_effects(unit_effect_corr=uc)
        st.plotly_chart(res.fig, width="stretch", config=PLOTLY_CONFIG)
        st.markdown(res.interpret())
        with st.expander("❓ What is this?"):
            st.markdown(res.explain().to_markdown())
    with tabs[2]:
        icc = st.slider("Intra-cluster correlation (ICC)", 0.0, 0.9, 0.3, 0.05)
        res = learn_clustering_se(icc=icc)
        st.plotly_chart(res.fig, width="stretch", config=PLOTLY_CONFIG)
        st.markdown(res.interpret())
        with st.expander("❓ What is this?"):
            st.markdown(res.explain().to_markdown())
    with tabs[3]:
        periods = st.slider("Periods per unit", 2, 8, 2, 1, key="fd_periods")
        res = learn_first_differences(n_periods=int(periods))
        st.plotly_chart(res.fig, width="stretch", config=PLOTLY_CONFIG)
        st.markdown(res.interpret())
        with st.expander("❓ What is this?"):
            st.markdown(res.explain().to_markdown())
    with tabs[4]:
        periods = st.slider("Periods per unit", 2, 12, 6, 1, key="wl_periods")
        res = learn_within_vs_lsdv(n_periods=int(periods))
        st.plotly_chart(res.fig, width="stretch", config=PLOTLY_CONFIG)
        st.markdown(res.interpret())
        with st.expander("❓ What is this?"):
            st.markdown(res.explain().to_markdown())
    with tabs[5]:
        rho = st.slider(
            "AR(1) persistence rho (higher = slower convergence)",
            0.5,
            0.99,
            0.9,
            0.01,
            key="bc_rho",
        )
        corr = st.slider(
            "Correlation between the determinant z and the initial level",
            0.0,
            0.95,
            0.7,
            0.05,
            key="bc_corr",
        )
        gamma = st.slider(
            "Loading on the determinant z", 0.0, 2.0, 0.6, 0.1, key="bc_g"
        )
        res = learn_beta_convergence(rho=rho, corr=corr, gamma=gamma)
        st.plotly_chart(res.fig, width="stretch", config=PLOTLY_CONFIG)
        st.markdown(res.interpret())
        with st.expander("❓ What is this?"):
            st.markdown(res.explain().to_markdown())


def _is_panel(active: Active) -> bool:
    """Return ``True`` when the data has both a cross-section id and a time dimension."""
    return bool(active.time and active.entities)


def page_panel_structure() -> None:
    """Within/between variation, panel balance, spaghetti, dynamics and transitions."""
    from expdpy import (
        explore_distribution_over_time,
        explore_panel_structure,
        explore_scatter_plot_within_between,
        explore_spaghetti_plot,
        explore_transition_matrix,
        explore_value_heatmap,
        explore_within_persistence,
        explore_xtsum_table,
    )

    active = _active_or_stop()
    st.header("Panel structure & dynamics")
    if not _is_panel(active):
        st.info("These views need a panel: a cross-section id and a time dimension.")
        return
    assert active.time is not None  # narrowed by _is_panel
    entity, time = active.entities[0], active.time
    st.caption(f"Entity: **{entity}** · Time: **{time}**")
    nums = _numeric(active)
    factors = _factors(active)

    def _show(make, *, interpret=True) -> None:
        """Render a result's figure (safely) plus its plain-language interpretation."""
        try:
            res = make()
        except Exception as exc:  # surface the message, keep the page alive
            st.info(str(exc))
            return
        if res is None:
            return
        st.plotly_chart(res.fig, width="stretch", config=PLOTLY_CONFIG)
        if interpret:
            st.markdown(res.interpret())

    st.subheader("Within / between variation")
    xtsum_vars = w.multiselect(
        "Variables", nums, key="ps_xtsum_vars", default=nums[: min(4, len(nums))]
    )
    if [v for v in xtsum_vars if v not in (None, "None")]:
        try:
            res = explore_xtsum_table(
                active.sample, var=list(xtsum_vars), entity=entity, time=time
            )
            st.html(res.gt.as_raw_html())
            st.markdown(res.interpret())
        except Exception as exc:
            st.info(str(exc))

    st.divider()
    st.subheader("Within-vs-between scatter")
    c1, c2 = st.columns(2)
    with c1:
        wb_x = w.selectbox("X", nums, key="ps_wb_x")
    with c2:
        wb_y = w.selectbox(
            "Y", nums, key="ps_wb_y", default=nums[1] if len(nums) > 1 else None
        )
    _show(
        lambda: explore_scatter_plot_within_between(
            active.sample, wb_x, wb_y, entity=entity, time=time
        )
    )

    st.divider()
    st.subheader("Panel balance & coverage")
    try:
        struct = explore_panel_structure(active.sample, entity=entity, time=time)
        st.markdown(struct.interpret())
        st.plotly_chart(struct.fig, width="stretch", config=PLOTLY_CONFIG)
        with st.expander("Balance summary"):
            st.dataframe(struct.df_summary, width="stretch", hide_index=True)
    except Exception as exc:
        st.info(str(exc))
    vh_var = w.selectbox("Value heatmap variable", nums, key="ps_vh_var")
    standardize = w.selectbox(
        "Standardize", ["none", "by_time", "by_entity", "global"], key="ps_vh_std"
    )
    _show(
        lambda: explore_value_heatmap(
            active.sample, vh_var, entity=entity, time=time, standardize=standardize
        ),
        interpret=False,
    )

    st.divider()
    st.subheader("Per-unit trajectories (spaghetti)")
    spag_var = w.selectbox("Variable", nums, key="ps_spag_var")
    _show(
        lambda: explore_spaghetti_plot(
            active.sample, spag_var, entity=entity, time=time
        )
    )

    st.divider()
    st.subheader("Distribution over time")
    dot_var = w.selectbox("Variable", nums, key="ps_dot_var")
    dot_style = w.selectbox(
        "Style", ["ridgeline", "animated_hist", "animated_violin"], key="ps_dot_style"
    )
    _show(
        lambda: explore_distribution_over_time(
            active.sample, dot_var, time=time, style=dot_style
        )
    )

    st.divider()
    st.subheader("Transition matrix")
    tm_var = w.selectbox("State variable", factors + nums, key="ps_tm_var")
    tm_bins = w.slider("Bins (numeric only)", 2, 8, key="ps_tm_bins", default=4)
    _show(
        lambda: explore_transition_matrix(
            active.sample, tm_var, entity=entity, time=time, n_bins=int(tm_bins)
        )
    )

    st.divider()
    st.subheader("Within-unit persistence")
    wp_var = w.selectbox("Variable", nums, key="ps_wp_var")
    _show(
        lambda: explore_within_persistence(
            active.sample, wp_var, entity=entity, time=time
        )
    )


def page_event_study() -> None:
    """Event study / staggered difference-in-differences for a treated panel."""
    from expdpy import analyze_event_study

    active = _active_or_stop()
    st.header("Event study & staggered DiD")
    if not _is_panel(active):
        st.info("Event studies need a panel: a cross-section id and a time dimension.")
        return
    factors = active.var_cats.factor or []
    if not factors:
        st.info(
            "Need a *cohort* column giving each unit's first-treated period "
            "(0 = never treated)."
        )
        return
    assert active.time is not None  # narrowed by _is_panel
    unit, time = active.entities[0], active.time
    st.caption(f"Unit: **{unit}** · Time: **{time}**")
    c1, c2, c3 = st.columns(3)
    with c1:
        outcome = st.selectbox("Outcome", _numeric(active), key="es_outcome")
    with c2:
        cohort = st.selectbox("First-treatment cohort", factors, key="es_cohort")
    with c3:
        estimator = st.selectbox(
            "Estimator", ["did2s", "twfe", "saturated", "lpdid"], key="es_estimator"
        )
    try:
        res = analyze_event_study(
            active.sample,
            outcome=outcome,
            unit=unit,
            time=time,
            cohort=cohort,
            estimator=cast(Any, estimator),
        )
    except Exception as exc:  # surface the message, keep the page alive
        st.info(str(exc))
        return
    st.plotly_chart(res.fig, width="stretch", config=PLOTLY_CONFIG)
    st.markdown(res.interpret())
    with st.expander("❓ What is this? (method explainer)"):
        st.markdown(res.explain().to_markdown())


def page_panel_models() -> None:
    """Pooled / between / fixed / random-effects comparison, Hausman, and CRE (Mundlak)."""
    from expdpy import analyze_cre_table, analyze_hausman_test, analyze_panel_table

    active = _active_or_stop()
    st.header("Panel models")
    if not _is_panel(active):
        st.info("Panel models need a cross-section id and a time dimension.")
        return
    assert active.time is not None  # narrowed by _is_panel
    entity, time = active.entities[0], active.time
    st.caption(f"Entity: **{entity}** · Time: **{time}**")
    dv = st.selectbox("Dependent variable", _numeric(active), key="pm_dv")
    xs = st.multiselect("Independent variables", _numeric(active), key="pm_xs")
    if not (dv and xs):
        st.info("Choose a dependent variable and at least one independent variable.")
        return
    try:
        res = analyze_panel_table(
            active.sample, dv=dv, idvs=xs, entity=entity, time=time, format="md"
        )
    except ImportError as exc:  # linearmodels not installed
        st.warning(str(exc))
        return
    except Exception as exc:  # surface the message, keep the page alive
        st.info(str(exc))
        return
    st.text(res.etable)  # the linearmodels side-by-side comparison
    st.markdown(res.interpret())

    st.subheader("Hausman test (fixed vs random effects)")
    try:
        hausman = analyze_hausman_test(
            active.sample, dv=dv, idvs=xs, entity=entity, time=time
        )
    except Exception as exc:  # surface the message, keep the page alive
        st.info(str(exc))
        return
    st.markdown(hausman.interpret())
    with st.expander("❓ What is this? (method explainer)"):
        st.markdown(hausman.explain().to_markdown())

    st.subheader("Correlated random effects (Mundlak)")
    st.caption(
        "A random-effects model augmented with each regressor's unit mean. The coefficient "
        "on the regressor equals its fixed-effects (within) estimate; a joint test on the "
        "mean terms is the regression-form Hausman test."
    )
    try:
        cre = analyze_cre_table(
            active.sample, dv=dv, idvs=xs, entity=entity, time=time, format="md"
        )
    except Exception as exc:  # surface the message, keep the page alive
        st.info(str(exc))
        return
    st.text(cre.etable)
    st.markdown(cre.interpret())
    with st.expander("❓ What is this? (method explainer)"):
        st.markdown(cre.explain().to_markdown())


def page_sigma_convergence() -> None:
    """Track whether the cross-sectional dispersion of a variable shrinks over time."""
    from expdpy import analyze_sigma_convergence

    active = _active_or_stop()
    st.header("Sigma convergence")
    if not _is_panel(active):
        st.info(
            "Sigma convergence needs a panel: a cross-section id and a time dimension."
        )
        return
    assert active.time is not None  # narrowed by _is_panel
    entity, time = active.entities[0], active.time
    st.caption(f"Entity: **{entity}** · Time: **{time}**")
    var = st.selectbox("Variable", _numeric(active), key="sigma_var")
    if not var or var == "None":
        st.info("Choose a numeric variable to track its cross-sectional dispersion.")
        return
    try:
        res = analyze_sigma_convergence(active.sample, var, entity=entity, time=time)
    except Exception as exc:  # surface the message, keep the page alive
        st.info(str(exc))
        return
    st.plotly_chart(res.fig, width="stretch", config=PLOTLY_CONFIG)
    st.markdown(res.interpret())
    for note in res.notes:
        st.caption(f"⚠️ {note}")
    with st.expander("❓ What is this? (method explainer)"):
        st.markdown(res.explain().to_markdown())


def page_explainers() -> None:
    """Browse the concept explainers — the Learn module's topic index."""
    from expdpy import explain, list_topics

    st.header("Concept explainers")
    st.caption(
        "Plain-language explainers for every method and idea in expdpy — what it is, when "
        "to use it, and the caveats. These need no dataset."
    )
    topics = list_topics()
    topic = st.selectbox("Topic", topics, key="explainer_topic")
    if topic:
        st.markdown(explain(topic).to_markdown())


# Extra pages appended after the data-driven ones. Event study and panel models need a
# panel structure (callable gate); the sandboxes and explainers pages are always available.
_PAGE_SPECS.append(
    ("Panel structure", "🧱", "panel_structure", page_panel_structure, _is_panel)
)
_PAGE_SPECS.append(
    ("Event study & DiD", "🎯", "event_study", page_event_study, _is_panel)
)
_PAGE_SPECS.append(("Panel models", "🪧", "panel_models", page_panel_models, _is_panel))
_PAGE_SPECS.append(
    ("Sigma convergence", "📉", "sigma_convergence", page_sigma_convergence, _is_panel)
)
_PAGE_SPECS.append(("Concept sandboxes", "🧪", "sandboxes", page_sandboxes, None))
_PAGE_SPECS.append(("Concept explainers", "📚", "explainers", page_explainers, None))

# Which module each page belongs to. The three apps (Explore / Analyze / Learn) each render
# only the pages mapped to their module; the combined nav (module=None) shows them all.
_MODULE: dict[str, str] = {
    "overview": "explore",
    "distributions": "explore",
    "by_group": "explore",
    "trends": "explore",
    "correlations": "explore",
    "panel_structure": "explore",
    "regression": "analyze",
    "event_study": "analyze",
    "panel_models": "analyze",
    "sigma_convergence": "analyze",
    "sandboxes": "learn",
    "explainers": "learn",
}


def _spec_available(gate: PageGate, active: Active) -> bool:
    """Whether a page spec's gate admits ``active``.

    ``gate`` is ``None`` (always shown), a callable ``(active) -> bool`` (custom condition,
    e.g. panel structure), or a list of component names (shown when any are available).
    """
    if gate is None:
        return True
    if callable(gate):
        return bool(gate(active))
    return bool(set(gate) & set(active.active_components))


def selected_specs(active: Active, module: str | None = None) -> list[tuple]:
    """Return the page specs available for ``active``, optionally limited to one ``module``.

    ``module`` is ``"explore"``, ``"analyze"``, ``"learn"`` (one app's pages) or ``None``
    (every page, the combined navigation).
    """
    return [
        spec
        for spec in _PAGE_SPECS
        if (module is None or _MODULE.get(spec[2]) == module)
        and _spec_available(spec[4], active)
    ]


def build_pages(active: Active, module: str | None = None) -> list:
    """Return the ``st.Page`` list for ``active``, limited to ``module`` when given."""
    return [
        st.Page(func, title=title, icon=icon, url_path=url)
        for title, icon, url, func, _ in selected_specs(active, module)
    ]
