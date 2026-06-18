"""Per-component compute helpers for the ExPdPy app.

Each helper turns the current analysis sample plus the user's selections into a Plotly
figure or a Great Tables HTML string, reusing the library's ``prepare_*`` functions. They
return ``None`` when the selection is incomplete so renderers can no-op gracefully.
"""

from __future__ import annotations

import pandas as pd

from expdpy import (
    prepare_bar_chart,
    prepare_by_group_bar_graph,
    prepare_by_group_trend_graph,
    prepare_by_group_violin_graph,
    prepare_correlation_graph,
    prepare_descriptive_table,
    prepare_ext_obs_table,
    prepare_fwl_plot,
    prepare_histogram,
    prepare_missing_values_graph,
    prepare_quantile_trend_graph,
    prepare_regression_table,
    prepare_scatter_plot,
    prepare_trend_graph,
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
]

# Components that only need the panel's time dimension.
TS_COMPONENTS = {
    "missing_values",
    "trend_graph",
    "quantile_trend_graph",
    "by_group_trend_graph",
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
}


def _ok(value: str | None) -> bool:
    return value not in (None, "", "None", "All", "Full Sample")


def descriptive(sample: pd.DataFrame):
    """Return descriptive-statistics HTML for the numeric columns."""
    num = sample.select_dtypes("number")
    if num.shape[1] < 1:
        return None
    return prepare_descriptive_table(num).gt.as_raw_html()


def ext_obs(sample: pd.DataFrame, var: str | None):
    if not _ok(var) or var not in sample.columns:
        return None
    n = min(5, len(sample) // 2)
    if n < 1:
        return None
    return prepare_ext_obs_table(sample, n=n, var=var).gt.as_raw_html()


def corrplot(sample: pd.DataFrame):
    num = sample.select_dtypes("number")
    if num.shape[1] < 2 or len(num) < 5:
        return None
    return prepare_correlation_graph(num).fig


def histogram(sample: pd.DataFrame, var: str | None, bins: int):
    if not _ok(var) or var not in sample.columns:
        return None
    assert var is not None
    return prepare_histogram(sample, var, bins=bins).fig


def bar_chart(sample: pd.DataFrame, var: str | None):
    if not _ok(var) or var not in sample.columns:
        return None
    assert var is not None
    return prepare_bar_chart(sample, var).fig


def missing(sample: pd.DataFrame, ts_id: str | None):
    if not _ok(ts_id) or ts_id not in sample.columns:
        return None
    assert ts_id is not None
    return prepare_missing_values_graph(sample, ts_id=ts_id)


def scatter(sample, x, y, color, size, loess):
    if not (_ok(x) and _ok(y)):
        return None
    return prepare_scatter_plot(
        sample,
        x,
        y,
        color=color if _ok(color) else None,
        size=size if _ok(size) else None,
        loess=1 if loess else 0,
    )


def trend(sample: pd.DataFrame, ts_id: str | None, variables: list[str]):
    variables = [v for v in variables if _ok(v) and v in sample.columns]
    if not _ok(ts_id) or not variables:
        return None
    assert ts_id is not None
    return prepare_trend_graph(sample, ts_id=ts_id, var=variables).fig


def quantile_trend(sample, ts_id, var):
    if not (_ok(ts_id) and _ok(var)) or var not in sample.columns:
        return None
    return prepare_quantile_trend_graph(sample, ts_id=ts_id, var=var).fig


def by_group_bar(sample, byvar, var):
    if not (_ok(byvar) and _ok(var)):
        return None
    return prepare_by_group_bar_graph(sample, byvar, var).fig


def by_group_violin(sample, byvar, var):
    if not (_ok(byvar) and _ok(var)):
        return None
    return prepare_by_group_violin_graph(sample, byvar, var)


def by_group_trend(sample, ts_id, group, var):
    if not (_ok(ts_id) and _ok(group) and _ok(var)):
        return None
    return prepare_by_group_trend_graph(
        sample, ts_id=ts_id, group_var=group, var=var
    ).fig


def regression(sample, y, xs, fes, clusters):
    xs = [x for x in xs if _ok(x)]
    if not (_ok(y) and xs):
        return None
    fes = [f for f in fes if _ok(f)]
    clusters = [c for c in clusters if _ok(c)]
    res = prepare_regression_table(
        sample, dvs=y, idvs=xs, feffects=fes, clusters=clusters, format="gt"
    )
    return res.etable.as_raw_html()


def fwl_plot(sample, y, xs, focal, fes, clusters):
    """Frisch-Waugh-Lovell plot for the focal regressor, reusing the regression inputs."""
    xs = [x for x in xs if _ok(x)]
    if not (_ok(y) and _ok(focal) and focal in xs):
        return None
    controls = [x for x in xs if x != focal]
    fes = [f for f in fes if _ok(f)]
    clusters = [c for c in clusters if _ok(c)]
    return prepare_fwl_plot(
        sample, dv=y, var=focal, controls=controls, feffects=fes, clusters=clusters
    ).fig
