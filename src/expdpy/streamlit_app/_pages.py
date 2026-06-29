"""The analysis pages and the ``st.navigation`` page set.

Each page reads the active prepared sample (stashed by the sidebar) and renders the
components assigned to it, reusing the Plotly compute helpers
(:mod:`expdpy.streamlit_app._components`) and the native table renderers
(:mod:`expdpy.streamlit_app._render`). Pages whose components are all unavailable for the
current data (e.g. time-series views on cross-sectional data) are omitted from the navigation.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, cast

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


def _prioritize(names: list[str], active: Active) -> list[str]:
    """Float the declared key variables (outcome, then covariates) to the front of ``names``.

    A no-op when no roles are declared, so role-less data keeps its original column order.
    """
    front = [n for n in [active.outcome, *active.covariates] if n in set(names)]
    if not front:
        return names
    front = list(dict.fromkeys(front))
    rest = [n for n in names if n not in set(front)]
    return [*front, *rest]


def _numeric(active: Active) -> list[str]:
    return _prioritize(active.var_cats.numeric_logical, active) or ["None"]


def _factors(active: Active) -> list[str]:
    return _prioritize(active.var_cats.grouping, active) or ["None"]


def _d_outcome(active: Active) -> str | None:
    """Default a single numeric selector to the declared main outcome."""
    return active.outcome


def _d_cov(active: Active) -> str | None:
    """Default a single covariate selector to the first declared covariate."""
    return active.covariates[0] if active.covariates else None


def _d_covs(active: Active) -> list[str]:
    """Default a multiselect to all declared covariates."""
    return list(active.covariates)


def _x_default(active: Active, nums: list[str]) -> str | None:
    """Scatter x default: the first covariate, else a numeric that is not the outcome."""
    if active.covariates:
        return active.covariates[0]
    rest = [n for n in nums if n != active.outcome and n != "None"]
    return rest[0] if rest else None


def _y_default(active: Active, nums: list[str]) -> str | None:
    """Scatter y default: the main outcome, else the second numeric (prior behaviour)."""
    if active.outcome in nums:
        return active.outcome
    return nums[1] if len(nums) > 1 else None


def _key_numeric_default(active: Active, nums: list[str]) -> list[str]:
    """Multiselect default: the key variables present, else the first few numerics."""
    keys = [n for n in [active.outcome, *active.covariates] if n in nums]
    return keys or nums[: min(4, len(nums))]


def _fe_choices(active: Active) -> list[str]:
    # In a single-period cross-section (``active.time is None``) there is one observation per
    # entity, so entity/time fixed effects are degenerate — offer only the grouping factors.
    if active.time is None:
        return active.var_cats.grouping or ["None"]
    return active.var_cats.fe_choices or ["None"]


def _is_unbalanced(active: Active) -> bool:
    """Return ``True`` for a panel where not every entity is observed in every period."""
    entity = active.entities[0] if active.entities else None
    time = active.time
    df = active.sample
    if not entity or not time or entity not in df.columns or time not in df.columns:
        return False
    return len(df) != df[entity].nunique() * df[time].nunique()


def _render_missing(active: Active) -> None:
    """Missing-value heatmap, or a clear message when the sample has no missing values."""
    time = active.time
    df = active.sample
    if not time or time not in df.columns:
        return
    others = df.drop(columns=[time], errors="ignore")
    if len(others.columns) and not bool(others.isna().any().any()):
        st.success("No missing values in this sample.")
        if _is_unbalanced(active):
            st.caption(
                "This is an unbalanced panel — the absent unit-years (structural gaps) are "
                "shown in **Panel balance & coverage** above, not here."
            )
        return
    render.render_plotly(lambda: comp.missing(df, time))


def _show_panel_result(make: Callable[[], Any], *, interpret: bool = True) -> None:
    """Render a panel result's figure (safely), interpretation tucked in an expander.

    Surfaces the exception message instead of crashing the page on bad input, and keeps the
    plain-language ``.interpret()`` behind a collapsed expander so the default view is just
    the tool and its result (the app is a tool, not a tutorial).
    """
    try:
        res = make()
    except Exception as exc:  # surface the message, keep the page alive
        st.info(str(exc))
        return
    if res is None:
        return
    st.plotly_chart(res.fig, width="stretch", config=PLOTLY_CONFIG)
    if interpret and hasattr(res, "interpret"):
        with st.expander("Plain-language reading"):
            st.markdown(res.interpret())


# --------------------------------------------------------------------------------- pages ---
def page_overview() -> None:
    """Sample preview plus the panel skeleton: balance, missing values, value heatmap."""
    from expdpy import explore_panel_structure, explore_value_heatmap

    active = _active_or_stop()
    st.header("Overview & Data")
    st.caption(
        f"**{active.source_name}** — {len(active.sample):,} rows, "
        f"{active.sample.shape[1]} columns after subsetting / outlier treatment."
    )
    with st.expander("Preview the analysis sample", expanded=False):
        st.dataframe(active.sample.head(100), width="stretch", hide_index=True)

    panel = _is_panel(active)
    entity = active.entities[0] if panel else None
    time = active.time if panel else None

    if panel:
        st.subheader("Panel balance & coverage")
        try:
            struct = explore_panel_structure(active.sample, entity=entity, time=time)
            st.plotly_chart(struct.fig, width="stretch", config=PLOTLY_CONFIG)
            with st.expander("Balance summary"):
                # The summary's ``value`` column mixes bool/int/float; stringify it so
                # Streamlit's Arrow serialization renders it cleanly.
                summary = struct.df_summary.astype({"value": str})
                st.dataframe(summary, width="stretch", hide_index=True)
            with st.expander("Plain-language reading"):
                st.markdown(struct.interpret())
        except Exception as exc:  # surface the message, keep the page alive
            st.info(str(exc))

    if _has(active, "missing_values"):
        st.subheader("Missing values")
        _render_missing(active)

    if panel:
        st.subheader("Value heatmap")
        vh_var = w.selectbox(
            "Variable", _numeric(active), key="ps_vh_var", default=_d_outcome(active)
        )
        standardize = w.selectbox(
            "Standardize",
            ["none", "by_time", "by_entity", "global"],
            key="ps_vh_std",
            relabel=False,
        )
        _show_panel_result(
            lambda: explore_value_heatmap(
                active.sample, vh_var, entity=entity, time=time, standardize=standardize
            ),
            interpret=False,
        )


def page_describe() -> None:
    """Descriptive statistics, histogram, category bar chart, extreme observations."""
    active = _active_or_stop()
    st.header("Describe variables")
    if _has(active, "descriptive_table"):
        st.subheader("Descriptive statistics")
        render.render_descriptive(
            active.sample, code=render.code_for("descriptive_table", active.time)
        )
    if _has(active, "histogram"):
        st.subheader("Histogram")
        var = w.selectbox(
            "Variable", _numeric(active), key="hist_var", default=_d_outcome(active)
        )
        bins = w.slider("Bins", 5, 100, key="hist_nr_of_breaks", default=20)
        render.render_plotly(
            lambda: comp.histogram(active.sample, var, int(bins)),
            code=render.code_for("histogram", active.time),
        )
    if _has(active, "bar_chart"):
        st.subheader("Bar chart")
        var = w.selectbox(
            "Variable", _factors(active), key="bar_chart_var1", default=_d_cov(active)
        )
        render.render_plotly(
            lambda: comp.bar_chart(active.sample, var),
            code=render.code_for("bar_chart", active.time),
        )
    if _has(active, "ext_obs"):
        st.subheader("Extreme observations")
        var = w.selectbox(
            "Variable", _numeric(active), key="ext_obs_var", default=_d_outcome(active)
        )
        render.render_ext_obs(
            active.sample, var, code=render.code_for("ext_obs", active.time)
        )


def page_within_between() -> None:
    """Within/between variance decomposition (xtsum) and per-unit trajectories."""
    from expdpy import explore_spaghetti_plot, explore_xtsum_table

    active = _active_or_stop()
    st.header("Within & between")
    if not _is_panel(active):
        st.info("These views need a panel: a cross-section id and a time dimension.")
        return
    assert active.time is not None  # narrowed by _is_panel
    entity, time = active.entities[0], active.time
    st.caption(f"Entity: **{entity}** · Time: **{time}**")
    nums = _numeric(active)

    st.subheader("Within / between variation")
    xtsum_vars = w.multiselect(
        "Variables",
        nums,
        key="ps_xtsum_vars",
        default=_key_numeric_default(active, nums),
    )
    if [v for v in xtsum_vars if v not in (None, "None")]:
        try:
            res = explore_xtsum_table(
                active.sample, var=list(xtsum_vars), entity=entity, time=time
            )
            st.html(res.gt.as_raw_html())
            with st.expander("Plain-language reading"):
                st.markdown(res.interpret())
        except Exception as exc:  # surface the message, keep the page alive
            st.info(str(exc))

    st.divider()
    st.subheader("Per-unit trajectories (spaghetti)")
    spag_var = w.selectbox(
        "Variable", nums, key="ps_spag_var", default=_d_outcome(active)
    )
    _show_panel_result(
        lambda: explore_spaghetti_plot(
            active.sample, spag_var, entity=entity, time=time
        )
    )


def page_by_group() -> None:
    """By-group bar, violin and trend graphs."""
    active = _active_or_stop()
    st.header("By group")
    if _has(active, "by_group_bar_graph"):
        st.subheader("Group means (bar)")
        byvar = w.selectbox(
            "Group by", _factors(active), key="bgbg_byvar", default=_d_cov(active)
        )
        var = w.selectbox(
            "Variable", _numeric(active), key="bgbg_var", default=_d_outcome(active)
        )
        render.render_plotly(
            lambda: comp.by_group_bar(active.sample, byvar, var),
            code=render.code_for("by_group_bar_graph", active.time),
        )
    if _has(active, "by_group_violin_graph"):
        st.subheader("Distribution by group (violin)")
        byvar = w.selectbox(
            "Group by", _factors(active), key="bgvg_byvar", default=_d_cov(active)
        )
        var = w.selectbox(
            "Variable", _numeric(active), key="bgvg_var", default=_d_outcome(active)
        )
        render.render_plotly(
            lambda: comp.by_group_violin(active.sample, byvar, var),
            code=render.code_for("by_group_violin_graph", active.time),
        )
    if _has(active, "by_group_trend_graph"):
        st.subheader("Group means over time")
        byvar = w.selectbox(
            "Group by", _factors(active), key="bgtg_byvar", default=_d_cov(active)
        )
        var = w.selectbox(
            "Variable", _numeric(active), key="bgtg_var", default=_d_outcome(active)
        )
        render.render_plotly(
            lambda: comp.by_group_trend(active.sample, active.time, byvar, var),
            code=render.code_for("by_group_trend_graph", active.time),
        )


def page_trends() -> None:
    """Variable trends, quantile trends, and the distribution shifting over time."""
    from expdpy import explore_distribution_over_time

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
                default=_d_outcome(active) if i == 1 else None,
            )
            for i in (1, 2, 3)
        ]
        render.render_plotly(
            lambda: comp.trend(active.sample, active.time, variables),
            code=render.code_for("trend_graph", active.time),
        )
    if _has(active, "quantile_trend_graph"):
        st.subheader("Quantile trend graph")
        var = w.selectbox(
            "Variable",
            _numeric(active),
            key="quantile_trend_graph_var",
            default=_d_outcome(active),
        )
        render.render_plotly(
            lambda: comp.quantile_trend(active.sample, active.time, var),
            code=render.code_for("quantile_trend_graph", active.time),
        )
    if _is_panel(active):
        st.subheader("Distribution over time")
        dot_var = w.selectbox(
            "Variable", _numeric(active), key="ps_dot_var", default=_d_outcome(active)
        )
        dot_style = w.selectbox(
            "Style",
            ["ridgeline", "animated_hist", "animated_violin"],
            key="ps_dot_style",
            relabel=False,
        )
        _show_panel_result(
            lambda: explore_distribution_over_time(
                active.sample, dot_var, time=active.time, style=dot_style
            )
        )


def page_correlations() -> None:
    """Correlation table + heatmap, the scatter plot, and the within/between split."""
    from expdpy import explore_scatter_plot_within_between

    active = _active_or_stop()
    st.header("Relationships")
    if _has(active, "corrplot"):
        st.subheader("Correlations")
        render.render_correlation(
            active.sample, code=render.code_for("corrplot", active.time)
        )
    if _has(active, "scatter_plot"):
        st.subheader("Scatter plot")
        nums = _numeric(active)
        c1, c2 = st.columns(2)
        with c1:
            x = w.selectbox(
                "X", nums, key="scatter_x", default=_x_default(active, nums)
            )
            color = w.selectbox(
                "Color", _factors(active) + nums, key="scatter_color", none=True
            )
        with c2:
            y = w.selectbox(
                "Y", nums, key="scatter_y", default=_y_default(active, nums)
            )
            size = w.selectbox("Size", nums, key="scatter_size", none=True)
        loess = w.checkbox("LOESS smoother", key="scatter_loess", default=True)
        render.render_plotly(
            lambda: comp.scatter(active.sample, x, y, color, size, loess),
            code=render.code_for("scatter_plot", active.time),
        )
    if _is_panel(active):
        assert active.time is not None  # narrowed by _is_panel
        entity, time = active.entities[0], active.time
        st.divider()
        st.subheader("Within-vs-between scatter")
        nums = _numeric(active)
        c1, c2 = st.columns(2)
        with c1:
            wb_x = w.selectbox(
                "X", nums, key="ps_wb_x", default=_x_default(active, nums)
            )
        with c2:
            wb_y = w.selectbox(
                "Y", nums, key="ps_wb_y", default=_y_default(active, nums)
            )
        _show_panel_result(
            lambda: explore_scatter_plot_within_between(
                active.sample, wb_x, wb_y, entity=entity, time=time
            )
        )


def _render_spec_comparison(
    active: Active,
    y: str | None,
    xs: list[str],
    fes: list[str],
    clusters: list[str],
) -> None:
    """Save the current regression spec and compare all saved specs side by side (item 5)."""

    def _ok(v: str | None) -> bool:
        return v not in (None, "", "None")

    valid_xs = [x for x in xs if _ok(x)]
    specs: list[dict] = st.session_state.setdefault("_saved_specs", [])

    c1, c2, _ = st.columns([1, 1, 2])
    if c1.button(
        "💾 Save spec for comparison",
        disabled=not (_ok(y) and valid_xs),
        help="Stash the current dependent/independent variables, fixed effects and clusters.",
    ):
        specs.append(
            {
                "label": f"({len(specs) + 1})",
                "y": y,
                "xs": valid_xs,
                "fes": [f for f in fes if _ok(f)],
                "clusters": list(clusters),
            }
        )
    if specs and c2.button("🗑 Clear saved"):
        st.session_state["_saved_specs"] = []
        return
    if not specs:
        return

    # Only specs whose columns still exist in the current sample are comparable.
    usable = [
        s
        for s in specs
        if s["y"] in active.sample.columns
        and all(x in active.sample.columns for x in s["xs"])
    ]
    st.divider()
    st.subheader(f"Saved specifications ({len(usable)})")
    if not usable:
        st.info(
            "Saved specs reference columns not in the current sample. Clear and re-save."
        )
        return

    from expdpy import analyze_coefficient_plot, analyze_regression_table

    try:
        res = analyze_regression_table(
            active.sample,
            dvs=[s["y"] for s in usable],
            idvs=[s["xs"] for s in usable],
            feffects=[s["fes"] for s in usable],
            clusters=[s["clusters"] for s in usable],
        )
    except Exception as exc:
        st.info(str(exc))
        return
    if hasattr(res.etable, "as_raw_html"):
        st.html(res.etable.as_raw_html())
    labels = [
        f"{s['label']} {s['y']} ~ {' + '.join(s['xs'])}"
        + (f" | {', '.join(s['fes'])}" if s["fes"] else "")
        for s in usable
    ]
    try:
        cp = analyze_coefficient_plot(res, model_labels=labels)
        st.plotly_chart(cp.fig, width="stretch", config=PLOTLY_CONFIG)
    except Exception as exc:
        st.info(str(exc))


def page_regression() -> None:
    """OLS regression with fixed effects and clustered standard errors."""
    active = _active_or_stop()
    st.header("Regression")
    if not _has(active, "regression"):
        st.info("The regression component is disabled for this app.")
        return
    y = w.selectbox(
        "Dependent variable", _numeric(active), key="reg_y", default=_d_outcome(active)
    )
    xs = w.multiselect(
        "Independent variables", _numeric(active), key="reg_x", default=_d_covs(active)
    )
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
    render.render_regression(
        active.sample,
        y,
        list(xs),
        [fe1, fe2],
        clusters,
        code=render.code_for("regression", active.time),
    )

    _render_spec_comparison(active, y, list(xs), [fe1, fe2], clusters)

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
            default=_d_cov(active),
            help="Its coefficient is the FWL slope; the remaining regressors become controls.",
        )
        render.render_plotly(
            lambda: comp.fwl_plot(
                active.sample, y, valid_xs, focal, [fe1, fe2], clusters
            ),
            code=render.code_for("fwl_plot", active.time),
        )

        st.divider()
        st.subheader("Stepwise estimation")
        st.caption(
            "A cumulative-stepwise (`csw`) comparison adds one regressor at a time — watch "
            "each estimate move as terms enter. `analyze_estimation` also offers "
            "serial-correlation-robust standard errors (Newey-West, Driscoll-Kraay) and weights."
        )
        if w.checkbox(
            "Show cumulative-stepwise table", key="est_stepwise", default=True
        ):
            from expdpy import analyze_estimation

            try:
                est = analyze_estimation(
                    active.sample, dv=y, idvs=valid_xs, stepwise="csw"
                )
            except Exception as exc:  # surface the message, keep the page alive
                st.info(str(exc))
            else:
                if hasattr(est.etable, "as_raw_html"):
                    st.html(est.etable.as_raw_html())
                else:
                    st.text(est.etable)


def page_postestimation() -> None:
    """Read a fitted model: predictions, absorbed fixed effects, joint test, robust inference."""
    from expdpy import (
        analyze_fixef_plot,
        analyze_joint_test,
        analyze_predictions,
        analyze_regression_table,
        analyze_robust_inference,
    )

    active = _active_or_stop()
    st.header("Post-estimation")
    if not _has(active, "regression"):
        st.info("The regression component is disabled for this app.")
        return
    st.caption(
        "Fit a model from the same controls as the Regression page, then interrogate it."
    )
    y = w.selectbox(
        "Dependent variable", _numeric(active), key="reg_y", default=_d_outcome(active)
    )
    xs = w.multiselect(
        "Independent variables", _numeric(active), key="reg_x", default=_d_covs(active)
    )
    fe1 = w.selectbox("Fixed effect", _fe_choices(active), key="reg_fe1", none=True)
    valid_xs = [x for x in xs if x not in (None, "", "None")]
    if not (y and y != "None" and valid_xs):
        st.info("Choose a dependent variable and at least one independent variable.")
        return
    fes = [fe1] if fe1 not in (None, "", "None") else []
    try:
        model = analyze_regression_table(
            active.sample, dvs=y, idvs=valid_xs, feffects=fes
        )
    except Exception as exc:  # surface the message, keep the page alive
        st.info(str(exc))
        return

    st.subheader("Predictions")
    st.caption("Fitted values, residuals and actuals on the estimation sample.")
    try:
        st.dataframe(
            analyze_predictions(model).df.head(20), width="stretch", hide_index=True
        )
    except Exception as exc:  # surface the message, keep the page alive
        st.info(str(exc))

    if fes:
        st.subheader("Fixed-effect estimates")
        st.caption("The group intercepts the fixed effects absorbed, ranked (top 20).")
        _show_panel_result(
            lambda: analyze_fixef_plot(model, fixef=fes[0], top_n=20), interpret=False
        )

    st.subheader("Joint (Wald) test")
    st.caption("Test that a set of coefficients are jointly zero.")
    hyps = w.multiselect(
        "Coefficients jointly = 0", valid_xs, key="pe_hyps", default=valid_xs
    )
    valid_hyps = [h for h in hyps if h not in (None, "", "None")]
    if valid_hyps:
        try:
            st.text(analyze_joint_test(model, valid_hyps).summary())
        except Exception as exc:  # surface the message, keep the page alive
            st.info(str(exc))

    st.subheader("Robust inference (randomization)")
    st.caption(
        "A randomization-inference p-value for one coefficient — more cautious than the "
        "asymptotic standard error when clusters are few."
    )
    param = w.selectbox("Coefficient", valid_xs, key="pe_param", default=_d_cov(active))
    if st.button("Run randomization inference", key="pe_run"):
        try:
            ri = analyze_robust_inference(
                model, param, method="ritest", reps=200, seed=0
            )
            st.write(
                f"Estimate {ri.estimate:.4f} — randomization-inference p = "
                f"{ri.p_value:.4f} over {ri.reps} permutations."
            )
        except Exception as exc:  # surface the message, keep the page alive
            st.info(str(exc))


# The page specs (title, icon, url_path, function, gate) are assembled at the bottom of this
# module, once every page function and the ``_is_panel`` gate below have been defined.


def _show_sandbox(res: Any) -> None:
    """Render a ``learn_*`` sandbox result: figure, inline reading, data download, explainer."""
    st.plotly_chart(res.fig, width="stretch", config=PLOTLY_CONFIG)
    st.markdown(res.interpret())
    csv = res.data.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇ Download simulated data (CSV)",
        data=csv,
        file_name=f"{res.topic}_simulated.csv",
        mime="text/csv",
        key=f"sandbox-dl-{abs(hash(csv)) % 10**9}",
    )
    with st.expander("❓ What is this?"):
        st.markdown(res.explain().to_markdown())


def page_sandboxes() -> None:
    """Interactive teaching demos that simulate data — no dataset required."""
    from expdpy import (
        learn_beta_convergence,
        learn_clustering_se,
        learn_convergence_clubs,
        learn_correlated_random_effects,
        learn_first_differences,
        learn_hausman_test,
        learn_kuznets_waves,
        learn_measurement_error,
        learn_nickell_bias,
        learn_omitted_variable_bias,
        learn_pooled_vs_fixed_effects,
        learn_sigma_convergence,
        learn_within_vs_lsdv,
    )

    st.header("Concept sandboxes")
    st.caption(
        "Simulated demonstrations — pick one and turn the knobs to see the concept in "
        "action. These need no dataset; only the chosen sandbox runs."
    )

    def _first_differences() -> None:
        periods = st.slider("Periods per unit", 2, 8, 2, 1, key="fd_periods")
        _show_sandbox(learn_first_differences(n_periods=int(periods)))

    def _within_vs_lsdv() -> None:
        periods = st.slider("Periods per unit", 2, 12, 6, 1, key="wl_periods")
        _show_sandbox(learn_within_vs_lsdv(n_periods=int(periods)))

    def _pooled_vs_fe() -> None:
        uc = st.slider(
            "Correlation between x and the unit effect",
            0.0,
            0.95,
            0.8,
            0.05,
            key="pfe_uc",
        )
        _show_sandbox(learn_pooled_vs_fixed_effects(unit_effect_corr=uc))

    def _hausman() -> None:
        uc = st.slider(
            "Correlation between x and the unit effect",
            0.0,
            0.95,
            0.8,
            0.05,
            key="ha_uc",
        )
        _show_sandbox(learn_hausman_test(unit_effect_corr=uc))

    def _cre() -> None:
        uc = st.slider(
            "Correlation between x and the unit effect",
            0.0,
            0.95,
            0.8,
            0.05,
            key="cre_uc",
        )
        _show_sandbox(learn_correlated_random_effects(unit_effect_corr=uc))

    def _ovb() -> None:
        corr = st.slider(
            "Correlation between x and the omitted z",
            0.0,
            0.95,
            0.6,
            0.05,
            key="ovb_corr",
        )
        bz = st.slider("Effect of the omitted z", -2.0, 2.0, 1.0, 0.25, key="ovb_bz")
        _show_sandbox(learn_omitted_variable_bias(corr_xz=corr, beta_z=bz))

    def _measurement_error() -> None:
        noise = st.slider(
            "Measurement-noise SD on x", 0.0, 3.0, 1.0, 0.1, key="me_noise"
        )
        _show_sandbox(learn_measurement_error(noise_x=noise))

    def _clustering() -> None:
        icc = st.slider(
            "Intra-cluster correlation (ICC)", 0.0, 0.9, 0.3, 0.05, key="cl_icc"
        )
        _show_sandbox(learn_clustering_se(icc=icc))

    def _nickell() -> None:
        rho = st.slider("True AR(1) persistence rho", 0.1, 0.9, 0.6, 0.05, key="nk_rho")
        _show_sandbox(learn_nickell_bias(rho=rho))

    def _beta_convergence() -> None:
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
        _show_sandbox(learn_beta_convergence(rho=rho, corr=corr, gamma=gamma))

    def _sigma_convergence() -> None:
        rho = st.slider(
            "Contraction rate rho (lower = faster convergence)",
            0.5,
            0.99,
            0.93,
            0.01,
            key="sc_rho",
        )
        _show_sandbox(learn_sigma_convergence(rho=rho))

    def _convergence_clubs() -> None:
        rho = st.slider(
            "Within-club AR(1) persistence rho", 0.5, 0.99, 0.9, 0.01, key="cc_rho"
        )
        spread = st.slider("Within-club spread", 0.1, 1.0, 0.4, 0.05, key="cc_spread")
        _show_sandbox(learn_convergence_clubs(rho=rho, spread=spread))

    def _kuznets() -> None:
        n_units = st.slider("Number of units", 30, 150, 80, 10, key="kw_units")
        within_sd = st.slider(
            "Within-unit spread of development", 0.3, 2.0, 0.9, 0.1, key="kw_within"
        )
        _show_sandbox(learn_kuznets_waves(n_units=int(n_units), within_sd=within_sd))

    sandboxes: dict[str, Any] = {
        "First differences": _first_differences,
        "Within vs LSDV": _within_vs_lsdv,
        "Pooled vs fixed effects": _pooled_vs_fe,
        "Hausman test (FE vs RE)": _hausman,
        "Correlated random effects (Mundlak)": _cre,
        "Omitted-variable bias": _ovb,
        "Measurement error (attenuation)": _measurement_error,
        "Clustered standard errors": _clustering,
        "Nickell bias (dynamic panels)": _nickell,
        "Beta convergence": _beta_convergence,
        "Sigma convergence": _sigma_convergence,
        "Convergence clubs": _convergence_clubs,
        "Kuznets waves": _kuznets,
    }
    choice = st.selectbox("Sandbox", list(sandboxes), key="sandbox_pick")
    sandboxes[choice]()


def _is_panel(active: Active) -> bool:
    """Return ``True`` when the data has both a cross-section id and a time dimension."""
    return bool(active.time and active.entities)


def page_dynamics() -> None:
    """State transitions and within-unit persistence over the panel's time dimension."""
    from expdpy import explore_transition_matrix, explore_within_persistence

    active = _active_or_stop()
    st.header("Dynamics")
    if not _is_panel(active):
        st.info("These views need a panel: a cross-section id and a time dimension.")
        return
    assert active.time is not None  # narrowed by _is_panel
    entity, time = active.entities[0], active.time
    st.caption(f"Entity: **{entity}** · Time: **{time}**")
    nums = _numeric(active)
    factors = _factors(active)

    st.subheader("Transition matrix")
    tm_var = w.selectbox(
        "State variable", factors + nums, key="ps_tm_var", default=_d_outcome(active)
    )
    tm_bins = w.slider("Bins (numeric only)", 2, 8, key="ps_tm_bins", default=4)
    _show_panel_result(
        lambda: explore_transition_matrix(
            active.sample, tm_var, entity=entity, time=time, n_bins=int(tm_bins)
        )
    )

    st.divider()
    st.subheader("Within-unit persistence")
    wp_var = w.selectbox("Variable", nums, key="ps_wp_var", default=_d_outcome(active))
    _show_panel_result(
        lambda: explore_within_persistence(
            active.sample, wp_var, entity=entity, time=time
        )
    )


def page_event_study() -> None:
    """Treatment structure (panel view) and the event-study / staggered DiD path."""
    from expdpy import analyze_event_study, analyze_panel_view

    active = _active_or_stop()
    st.header("Event study & DiD")
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
        outcome = w.selectbox(
            "Outcome", _numeric(active), key="es_outcome", default=_d_outcome(active)
        )
    with c2:
        cohort = w.selectbox("First-treatment cohort", factors, key="es_cohort")
    with c3:
        estimator = st.selectbox(
            "Estimator", ["did2s", "twfe", "saturated", "lpdid"], key="es_estimator"
        )

    st.subheader("Treatment structure")
    st.caption(
        "Who is treated, and when — the first thing to inspect in a staggered design."
    )
    _show_panel_result(
        lambda: analyze_panel_view(active.sample, unit=unit, time=time, cohort=cohort),
        interpret=False,
    )

    st.divider()
    st.subheader("Event study")
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
    dv = w.selectbox(
        "Dependent variable", _numeric(active), key="pm_dv", default=_d_outcome(active)
    )
    xs = w.multiselect(
        "Independent variables", _numeric(active), key="pm_xs", default=_d_covs(active)
    )
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


def page_convergence() -> None:
    """Do incomes converge? Beta, sigma, and Phillips-Sul club convergence side by side."""
    from expdpy import (
        analyze_beta_convergence,
        analyze_convergence_clubs,
        analyze_sigma_convergence,
    )

    active = _active_or_stop()
    st.header("Convergence")
    if not _is_panel(active):
        st.info("Convergence needs a panel: a cross-section id and a time dimension.")
        return
    assert active.time is not None  # narrowed by _is_panel
    entity, time = active.entities[0], active.time
    st.caption(f"Entity: **{entity}** · Time: **{time}**")
    nums = _numeric(active)

    st.subheader("Beta convergence")
    st.caption(
        "Do initially poorer units grow faster? (speed and half-life of catch-up.)"
    )
    beta_var = w.selectbox("Variable", nums, key="beta_var", default=_d_outcome(active))
    if beta_var and beta_var != "None":
        _show_panel_result(
            lambda: analyze_beta_convergence(
                active.sample, beta_var, entity=entity, time=time
            )
        )

    st.divider()
    st.subheader("Sigma convergence")
    st.caption("Does the cross-sectional dispersion shrink over time?")
    sigma_var = w.selectbox(
        "Variable", nums, key="sigma_var", default=_d_outcome(active)
    )
    if sigma_var and sigma_var != "None":
        try:
            sig = analyze_sigma_convergence(
                active.sample, sigma_var, entity=entity, time=time
            )
        except Exception as exc:  # surface the message, keep the page alive
            st.info(str(exc))
        else:
            st.plotly_chart(sig.fig, width="stretch", config=PLOTLY_CONFIG)
            for note in sig.notes:
                st.caption(f"⚠️ {note}")
            with st.expander("Plain-language reading"):
                st.markdown(sig.interpret())

    st.divider()
    st.subheader("Convergence clubs")
    st.caption("Phillips-Sul log(t) test plus data-driven clustering into clubs.")
    clubs_var = w.selectbox(
        "Variable (pass it in logs)", nums, key="clubs_var", default=_d_outcome(active)
    )
    col1, col2 = st.columns(2)
    use_hp = col1.checkbox("HP-filter trend (lambda=400)", value=True, key="clubs_hp")
    merge = col2.selectbox(
        "Merge adjacent clubs", ("iterative", "single", "none"), key="clubs_merge"
    )
    if clubs_var and clubs_var != "None":
        try:
            clubs = analyze_convergence_clubs(
                active.sample,
                clubs_var,
                entity=entity,
                time=time,
                filter="hp" if use_hp else None,
                merge=cast('Literal["iterative", "single", "none"]', merge),
            )
        except Exception as exc:  # surface the message, keep the page alive
            st.info(str(exc))
        else:
            if clubs.converged:
                st.success(
                    f"The whole panel converges (global log(t) t = {clubs.global_tstat:.2f} "
                    "> -1.65): a single convergence club."
                )
            else:
                st.info(
                    f"Global convergence rejected (t = {clubs.global_tstat:.2f}). "
                    f"{clubs.n_clubs} club(s); {clubs.n_divergent} divergent unit(s)."
                )
            st.plotly_chart(clubs.fig, width="stretch", config=PLOTLY_CONFIG)
            st.dataframe(clubs.summary, width="stretch", hide_index=True)
            for note in clubs.notes:
                st.caption(f"⚠️ {note}")
            with st.expander("Plain-language reading"):
                st.markdown(clubs.interpret())


def page_kuznets_waves() -> None:
    """Estimate the extended Kuznets curve: pooled / between / within waves side by side."""
    from expdpy import analyze_kuznets_waves

    active = _active_or_stop()
    st.header("Kuznets waves")
    if not _is_panel(active):
        st.info("Kuznets waves need a panel: a cross-section id and a time dimension.")
        return
    assert active.time is not None  # narrowed by _is_panel
    entity, time = active.entities[0], active.time
    st.caption(f"Entity: **{entity}** · Time: **{time}**")
    numeric = _numeric(active)
    col1, col2 = st.columns(2)
    with col1:
        inequality = w.selectbox(
            "Inequality (outcome)", numeric, key="kw_ineq", default=_d_outcome(active)
        )
    dev_opts = [c for c in numeric if c != inequality] or numeric
    with col2:
        development = w.selectbox(
            "Development (e.g. log GDP per capita)",
            dev_opts,
            key="kw_dev",
            default=_d_cov(active),
        )
    controls = w.multiselect(
        "Controls to partial out of the between/within waves (optional)",
        [c for c in numeric if c not in (inequality, development)],
        key="kw_controls",
    )
    degree = st.slider("Polynomial degree", 2, 6, 4, 1, key="kw_degree")
    if (
        not inequality
        or inequality == "None"
        or not development
        or development == "None"
    ):
        st.info("Choose an inequality outcome and a development variable.")
        return
    try:
        res = analyze_kuznets_waves(
            active.sample,
            inequality,
            development,
            controls=controls or None,
            entity=entity,
            time=time,
            degree=int(degree),
        )
    except Exception as exc:  # surface the message, keep the page alive
        st.info(str(exc))
        return

    st.subheader("Raw relationship")
    st.plotly_chart(res.fig, width="stretch", config=PLOTLY_CONFIG)
    st.subheader("Between estimator (cross-country wave)")
    st.plotly_chart(res.fig_between, width="stretch", config=PLOTLY_CONFIG)
    st.subheader("Within estimator (two-way fixed effects)")
    st.plotly_chart(res.fig_within, width="stretch", config=PLOTLY_CONFIG)

    st.subheader("Comparison tables (each column adds the next power)")
    t1, t2, t3 = st.tabs(["Pooled OLS", "Between", "Within (two-way FE)"])
    with t1:
        st.html(res.gt_pooled.as_raw_html())
    with t2:
        st.html(res.gt_between.as_raw_html())
    with t3:
        st.html(res.gt_within.as_raw_html())

    st.markdown(res.interpret())
    for note in res.notes:
        st.caption(f"⚠️ {note}")
    with st.expander("❓ What is this? (method explainer)"):
        st.markdown(res.explain().to_markdown())


def page_explainers() -> None:
    """Browse the concept explainers — the Learn module's searchable topic index."""
    from expdpy import explain, list_topics

    st.header("Concept explainers")
    st.caption(
        "Plain-language explainers for every method and idea in expdpy — what it is, when "
        "to use it, and the caveats. These need no dataset."
    )
    topics = list_topics()
    query = (
        st.text_input(
            "Search topics",
            key="explainer_search",
            placeholder="e.g. fixed, cluster, IV",
        )
        .strip()
        .lower()
    )
    if query:

        def _matches(name: str) -> bool:
            exp = explain(name)
            return query in f"{name} {exp.title} {exp.what}".lower()

        topics = [t for t in topics if _matches(t)]
    if not topics:
        st.info("No topics match your search.")
        return
    topic = st.selectbox("Topic", topics, key="explainer_topic")
    if topic:
        st.markdown(explain(topic).to_markdown())


# The page specs in display order: (title, icon, url_path, function, gate). The Explore pages
# follow the case-study workflow of the docs (know the panel -> describe -> split its variation
# -> trends -> groups -> relationships -> dynamics); the Analyze and Learn pages follow. A gate
# is ``None`` (always shown), a list of component names (shown when any are active), or
# ``_is_panel`` (a callable requiring a full panel).
_PAGE_SPECS: list[PageSpec] = [
    # Explore
    ("Overview & Data", "🏠", "overview", page_overview, None),
    (
        "Describe variables",
        "📊",
        "describe",
        page_describe,
        ["descriptive_table", "histogram", "bar_chart", "ext_obs"],
    ),
    ("Within & between", "🔀", "within_between", page_within_between, _is_panel),
    ("Trends", "📈", "trends", page_trends, ["trend_graph", "quantile_trend_graph"]),
    (
        "By group",
        "👥",
        "by_group",
        page_by_group,
        ["by_group_bar_graph", "by_group_violin_graph", "by_group_trend_graph"],
    ),
    (
        "Relationships",
        "🔗",
        "correlations",
        page_correlations,
        ["corrplot", "scatter_plot"],
    ),
    ("Dynamics", "🔁", "dynamics", page_dynamics, _is_panel),
    # Analyze (in the docs case-study order: fit -> read it -> choose the estimator ->
    # the flagship curve -> a related convergence question -> a causal design)
    ("Regression", "🧮", "regression", page_regression, ["regression"]),
    ("Post-estimation", "🔎", "postestimation", page_postestimation, ["regression"]),
    ("Panel models", "🪧", "panel_models", page_panel_models, _is_panel),
    ("Kuznets waves", "🌊", "kuznets_waves", page_kuznets_waves, _is_panel),
    ("Convergence", "📉", "convergence", page_convergence, _is_panel),
    ("Event study & DiD", "🎯", "event_study", page_event_study, _is_panel),
    # Learn
    ("Concept sandboxes", "🧪", "sandboxes", page_sandboxes, None),
    ("Concept explainers", "📚", "explainers", page_explainers, None),
]

# Which module each page belongs to. The three apps (Explore / Analyze / Learn) each render
# only the pages mapped to their module; the combined nav (module=None) shows them all.
_MODULE: dict[str, str] = {
    "overview": "explore",
    "describe": "explore",
    "within_between": "explore",
    "trends": "explore",
    "by_group": "explore",
    "correlations": "explore",
    "dynamics": "explore",
    "regression": "analyze",
    "postestimation": "analyze",
    "panel_models": "analyze",
    "kuznets_waves": "analyze",
    "convergence": "analyze",
    "event_study": "analyze",
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
