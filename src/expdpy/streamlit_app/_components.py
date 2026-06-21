"""Per-component compute helpers for the ExPdPy app.

Each helper turns the current analysis sample plus the user's selections into a Plotly
figure or a Great Tables HTML string, reusing the library's ``explore_*`` / ``analyze_*``
functions. They return ``None`` when the selection is incomplete so renderers can no-op
gracefully.
"""

from __future__ import annotations

import pandas as pd

from expdpy import (
    analyze_event_study,
    analyze_fwl_plot,
    analyze_hausman_test,
    analyze_panel_table,
    analyze_regression_table,
    explore_bar_plot,
    explore_bar_plot_by_group,
    explore_correlation_plot,
    explore_descriptive_table,
    explore_ext_obs_table,
    explore_histogram,
    explore_missing_values_plot,
    explore_quantile_trend_plot,
    explore_scatter_plot,
    explore_trend_plot,
    explore_trend_plot_by_group,
    explore_violin_plot_by_group,
)

# Components in their canonical order, with the kind of output they render.
COMPONENT_ORDER = [
    "sample_selection",
    "subset_factor",
    "grouping",
    "bar_chart",
    "missing_values",
    "udvars",
    "descriptive_table",
    "histogram",
    "ext_obs",
    "by_group_bar_graph",
    "by_group_violin_graph",
    "trend_graph",
    "quantile_trend_graph",
    "by_group_trend_graph",
    "corrplot",
    "scatter_plot",
    "regression",
    "fwl_plot",
    "event_study",
    "panel_models",
]

# Components that only need the panel's time dimension.
TS_COMPONENTS = {
    "missing_values",
    "trend_graph",
    "quantile_trend_graph",
    "by_group_trend_graph",
    "event_study",
    "panel_models",
}

# Components rendered as a card with controls + output ("plotly" or "gt"); others are
# control-only (handled by the sidebar / pipeline).
COMPONENT_KIND = {
    "bar_chart": "plotly",
    "missing_values": "plotly",
    "descriptive_table": "gt",
    "histogram": "plotly",
    "ext_obs": "gt",
    "by_group_bar_graph": "plotly",
    "by_group_violin_graph": "plotly",
    "trend_graph": "plotly",
    "quantile_trend_graph": "plotly",
    "by_group_trend_graph": "plotly",
    "corrplot": "plotly",
    "scatter_plot": "plotly",
    "regression": "gt",
    "fwl_plot": "plotly",
    "event_study": "plotly",
    "panel_models": "gt",
}


def _ok(value: str | None) -> bool:
    return value not in (None, "", "None", "All", "Full Sample")


def descriptive(sample: pd.DataFrame):
    """Return descriptive-statistics HTML for the numeric columns."""
    num = sample.select_dtypes("number")
    if num.shape[1] < 1:
        return None
    return explore_descriptive_table(num).gt.as_raw_html()


def ext_obs(sample: pd.DataFrame, var: str | None):
    if not _ok(var) or var not in sample.columns:
        return None
    n = min(5, len(sample) // 2)
    if n < 1:
        return None
    return explore_ext_obs_table(sample, n=n, var=var).gt.as_raw_html()


def corrplot(sample: pd.DataFrame):
    num = sample.select_dtypes("number")
    if num.shape[1] < 2 or len(num) < 5:
        return None
    return explore_correlation_plot(num).fig


def histogram(sample: pd.DataFrame, var: str | None, bins: int):
    if not _ok(var) or var not in sample.columns:
        return None
    assert var is not None
    return explore_histogram(sample, var, bins=bins).fig


def bar_chart(sample: pd.DataFrame, var: str | None):
    if not _ok(var) or var not in sample.columns:
        return None
    assert var is not None
    return explore_bar_plot(sample, var).fig


def missing(sample: pd.DataFrame, time: str | None):
    if not _ok(time) or time not in sample.columns:
        return None
    assert time is not None
    return explore_missing_values_plot(sample, time=time).fig


def scatter(sample, x, y, color, size, loess):
    if not (_ok(x) and _ok(y)):
        return None
    return explore_scatter_plot(
        sample,
        x,
        y,
        color=color if _ok(color) else None,
        size=size if _ok(size) else None,
        loess=1 if loess else 0,
    ).fig


def trend(sample: pd.DataFrame, time: str | None, variables: list[str]):
    variables = [v for v in variables if _ok(v) and v in sample.columns]
    if not _ok(time) or not variables:
        return None
    assert time is not None
    return explore_trend_plot(sample, var=variables, time=time).fig


def quantile_trend(sample, time, var):
    if not (_ok(time) and _ok(var)) or var not in sample.columns:
        return None
    return explore_quantile_trend_plot(sample, var=var, time=time).fig


def by_group_bar(sample, byvar, var):
    if not (_ok(byvar) and _ok(var)):
        return None
    return explore_bar_plot_by_group(sample, byvar, var).fig


def by_group_violin(sample, byvar, var):
    if not (_ok(byvar) and _ok(var)):
        return None
    return explore_violin_plot_by_group(sample, byvar, var).fig


def by_group_trend(sample, time, group, var):
    if not (_ok(time) and _ok(group) and _ok(var)):
        return None
    return explore_trend_plot_by_group(sample, group, var, time=time).fig


def regression(sample, y, xs, fes, clusters):
    xs = [x for x in xs if _ok(x)]
    if not (_ok(y) and xs):
        return None
    fes = [f for f in fes if _ok(f)]
    clusters = [c for c in clusters if _ok(c)]
    res = analyze_regression_table(
        sample, dvs=y, idvs=xs, feffects=fes, clusters=clusters, format="gt"
    )
    return res.etable.as_raw_html()


def regression_notes(sample, y, xs, fes, clusters):
    """Return ``(interpretation, method_explainer)`` markdown for the regression card.

    Returns ``None`` when the selection is incomplete (mirroring :func:`regression`).
    """
    xs = [x for x in xs if _ok(x)]
    if not (_ok(y) and xs):
        return None
    fes = [f for f in fes if _ok(f)]
    clusters = [c for c in clusters if _ok(c)]
    res = analyze_regression_table(
        sample, dvs=y, idvs=xs, feffects=fes, clusters=clusters, format="gt"
    )
    return res.interpret(), res.explain().to_markdown()


def fwl_plot(sample, y, xs, focal, fes, clusters):
    """Frisch-Waugh-Lovell plot for the focal regressor, reusing the regression inputs."""
    xs = [x for x in xs if _ok(x)]
    if not (_ok(y) and _ok(focal) and focal in xs):
        return None
    controls = [x for x in xs if x != focal]
    fes = [f for f in fes if _ok(f)]
    clusters = [c for c in clusters if _ok(c)]
    return analyze_fwl_plot(
        sample, dv=y, var=focal, controls=controls, feffects=fes, clusters=clusters
    ).fig


def event_study(sample, outcome, unit, time, cohort, estimator):
    """Event-study / staggered-DiD plot for a treated panel (Plotly figure)."""
    if not (_ok(outcome) and _ok(unit) and _ok(time) and _ok(cohort)):
        return None
    return analyze_event_study(
        sample,
        outcome=outcome,
        unit=unit,
        time=time,
        cohort=cohort,
        estimator=estimator or "did2s",
    ).fig


def event_study_notes(sample, outcome, unit, time, cohort, estimator):
    """Return (interpretation, method-explainer) markdown for the event-study card."""
    if not (_ok(outcome) and _ok(unit) and _ok(time) and _ok(cohort)):
        return None
    res = analyze_event_study(
        sample,
        outcome=outcome,
        unit=unit,
        time=time,
        cohort=cohort,
        estimator=estimator or "did2s",
    )
    return res.interpret(), res.explain().to_markdown()


def panel_models(sample, dv, idvs, entity, time):
    """Pooled/between/FE/RE comparison HTML, or a friendly message if linearmodels is absent."""
    idvs = [x for x in idvs if _ok(x)]
    if not (_ok(dv) and idvs and _ok(entity) and _ok(time)):
        return None
    try:
        res = analyze_panel_table(
            sample, dv=dv, idvs=idvs, entity=entity, time=time, format="html"
        )
    except ImportError as exc:
        return f"<p><em>{exc}</em></p>"
    return res.etable


def panel_models_notes(sample, dv, idvs, entity, time):
    """Return (panel-table interpretation, Hausman-test markdown), or ``None``."""
    idvs = [x for x in idvs if _ok(x)]
    if not (_ok(dv) and idvs and _ok(entity) and _ok(time)):
        return None
    try:
        panel = analyze_panel_table(
            sample, dv=dv, idvs=idvs, entity=entity, time=time, format="df"
        )
        hausman = analyze_hausman_test(
            sample, dv=dv, idvs=idvs, entity=entity, time=time
        )
    except ImportError:
        return None
    return panel.interpret(), hausman.interpret()
