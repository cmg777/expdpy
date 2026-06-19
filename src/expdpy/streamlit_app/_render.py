"""Native Streamlit renderers for the analysis components.

Tables are rendered natively with :func:`streamlit.dataframe` (sortable, theme-aware) from
the plain ``DataFrame`` each ``prepare_*`` function returns; the publication-quality Great
Tables / pyfixest ``etable`` output is offered as an optional secondary view. Plotly figures
are passed straight to :func:`streamlit.plotly_chart`.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from expdpy import (
    prepare_correlation_graph,
    prepare_correlation_table,
    prepare_descriptive_table,
    prepare_ext_obs_table,
    prepare_regression_table,
)
from expdpy._theme import PLOTLY_CONFIG

__all__ = [
    "render_plotly",
    "render_descriptive",
    "render_ext_obs",
    "render_correlation",
    "render_regression",
]

_SELECT_HINT = (
    "Select the required variable(s) in the controls above to render this view."
)


def render_plotly(thunk, *, hint: str = _SELECT_HINT) -> None:
    """Render a Plotly figure produced by ``thunk`` (a figure or a no-arg callable).

    ``thunk`` is usually a ``lambda`` deferring the compute so that any error in building
    the figure surfaces as a friendly message instead of crashing the page. A ``None`` result
    means the selection is incomplete.
    """
    try:
        fig = thunk() if callable(thunk) else thunk
    except Exception as exc:
        st.warning(f"Could not render this chart: {exc}")
        return
    if fig is None:
        st.info(hint)
    else:
        st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG)


def _number_config(df: pd.DataFrame, decimals: int = 3) -> dict:
    """Build a ``column_config`` formatting every numeric column to ``decimals`` places."""
    cfg: dict = {}
    for col in df.columns:
        if col == "N":
            cfg[col] = st.column_config.NumberColumn("N", format="%d")
        elif pd.api.types.is_numeric_dtype(df[col]):
            cfg[col] = st.column_config.NumberColumn(col, format=f"%.{decimals}f")
    return cfg


def render_descriptive(sample: pd.DataFrame) -> None:
    """Descriptive statistics for the numeric/logical columns (native table)."""
    num = sample.select_dtypes("number")
    if num.shape[1] < 1 or len(sample) < 2:
        st.info("Need at least one numeric variable and two observations.")
        return
    try:
        res = prepare_descriptive_table(num)
    except Exception as exc:
        st.info(str(exc))
        return
    st.dataframe(
        res.df,
        width="stretch",
        column_config=_number_config(res.df),
    )
    with st.expander("Publication-quality table (Great Tables)"):
        st.html(res.gt.as_raw_html())


def render_ext_obs(sample: pd.DataFrame, var: str | None) -> None:
    """Top/bottom extreme observations sorted by ``var`` (native table)."""
    if not var or var in ("None", "") or var not in sample.columns:
        st.info("Choose a variable to list its extreme observations.")
        return
    n = min(5, len(sample) // 2)
    if n < 1:
        st.info("Not enough observations for an extreme-observations table.")
        return
    try:
        res = prepare_ext_obs_table(sample, n=n, var=var)
    except Exception as exc:
        st.info(str(exc))
        return
    st.dataframe(
        res.df,
        width="stretch",
        hide_index=True,
        column_config=_number_config(res.df),
    )
    st.caption(f"Top and bottom {n} observations, sorted by **{var}**.")


def render_correlation(sample: pd.DataFrame) -> None:
    """Correlation table (Pearson above / Spearman below) plus the Plotly heatmap."""
    num = sample.select_dtypes("number")
    if num.shape[1] < 2 or len(num) < 5:
        st.info("Need at least two numeric variables and five observations.")
        return
    try:
        res = prepare_correlation_table(num)
        graph = prepare_correlation_graph(num)
    except Exception as exc:
        st.info(str(exc))
        return
    st.dataframe(
        res.df_corr,
        width="stretch",
        column_config={
            c: st.column_config.NumberColumn(c, format="%.2f")
            for c in res.df_corr.columns
        },
    )
    st.caption(
        "Pearson correlations above the diagonal, Spearman correlations below "
        "(self-correlations on the diagonal)."
    )
    st.plotly_chart(graph.fig, width="stretch", config=PLOTLY_CONFIG)


def _stars(p: float) -> str:
    """Significance stars for a p-value (``***`` <1%, ``**`` <5%, ``*`` <10%)."""
    if pd.isna(p):
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


def render_regression(
    sample: pd.DataFrame,
    y: str | None,
    xs: list[str],
    fes: list[str],
    clusters: list[str],
) -> None:
    """OLS regression with fixed effects + clustered SEs (native coefficient table)."""

    def _ok(v: str | None) -> bool:
        return v not in (None, "", "None")

    xs = [x for x in xs if _ok(x)]
    if not (_ok(y) and xs):
        st.info("Choose a dependent variable and at least one independent variable.")
        return
    assert y is not None  # narrowed by the _ok(y) guard above
    fes = [f for f in fes if _ok(f)]
    clusters = [c for c in clusters if _ok(c)]
    try:
        res = prepare_regression_table(
            sample, dvs=y, idvs=xs, feffects=fes, clusters=clusters, format="gt"
        )
    except Exception as exc:
        st.info(str(exc))
        return

    tidy = res.df.copy()
    p_col = "Pr(>|t|)" if "Pr(>|t|)" in tidy.columns else tidy.columns[-1]
    tidy[""] = tidy[p_col].apply(_stars)
    drop = [c for c in ("model", "byvalue") if c in tidy.columns]
    display = tidy.drop(columns=drop)
    st.dataframe(
        display,
        width="stretch",
        hide_index=True,
        column_config=_number_config(display, decimals=4),
    )

    model = res.models[0]
    cols = st.columns(3)
    n_obs = getattr(model, "_N", None)
    r2 = getattr(model, "_r2", None)
    r2_within = getattr(model, "_r2_within", None)
    if n_obs is not None:
        cols[0].metric("Observations", f"{int(n_obs):,}")
    if r2 is not None:
        cols[1].metric("R²", f"{float(r2):.3f}")
    if r2_within is not None and fes:
        cols[2].metric("Within R²", f"{float(r2_within):.3f}")
    fe_note = ", ".join(fes) if fes else "none"
    cl_note = ", ".join(clusters) if clusters else "iid"
    st.caption(
        f"Fixed effects: {fe_note}. Standard errors: {cl_note}. "
        "Significance: *** p<0.01, ** p<0.05, * p<0.1."
    )
    if hasattr(res.etable, "as_raw_html"):
        with st.expander("Publication-quality table (pyfixest etable)"):
            st.html(res.etable.as_raw_html())
