"""The analysis-sample pipeline for the Streamlit app.

Runs the chain ``base_df → apply_user_vars → build_analysis_sample →
create_var_categories``. Results are memoised in ``st.session_state`` keyed on small,
hashable tokens (the data id, the user-defined-variable expressions, and the pipeline config)
so the (potentially large) DataFrame is never re-fingerprinted on every rerun and the sample
is only recomputed when an input that actually affects it changes.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from expdpy._validation import drop_required, required_columns
from expdpy.streamlit_app._sample import apply_user_vars, build_analysis_sample
from expdpy.streamlit_app._varcat import VarCats, create_var_categories


def _types_map(df_def: pd.DataFrame | None) -> dict[str, str]:
    """Return ``{var_name: type}`` from a data dictionary (empty when unavailable)."""
    cols = getattr(df_def, "columns", [])
    if df_def is None or "var_name" not in cols or "type" not in cols:
        return {}
    return {
        str(n): str(t) for n, t in zip(df_def["var_name"], df_def["type"], strict=True)
    }


__all__ = ["OUTLIER_CHOICES", "pipeline_cfg", "filter_spec", "udv_records", "analysis"]

#: Outlier-treatment radio value → label.
OUTLIER_CHOICES = {
    "1": "None",
    "2": "Winsorize 1%",
    "3": "Winsorize 5%",
    "4": "Truncate 1%",
    "5": "Truncate 5%",
}


def udv_records() -> tuple[tuple[str, str], ...]:
    """Return the user-defined-variable rows as a hashable ``(name, expr)`` tuple."""
    rows = st.session_state.get("udvars") or []
    out: list[tuple[str, str]] = []
    for r in rows:
        name = str(r.get("var_name") or "").strip()
        expr = str(r.get("var_def") or "").strip()
        if name and expr:
            out.append((name, expr))
    return tuple(out)


def filter_spec() -> tuple:
    """Reconstruct the active sample filters from the sidebar widget keys.

    Returns ``(period_range, cat_filters, range_filters)`` in **hashable** form (so it can go
    into the pipeline memo key). The widget values live under their own keys (``period_slider``,
    ``cat_filter_vars`` + ``catf::<var>``, ``range_filter_vars`` + ``rangef::<var>``) and
    Streamlit updates them *before* the script reruns, so reading them here applies a selection
    on the same run (no lag). An empty category selection means "no filter" (keeps all).
    """
    period = st.session_state.get("period_slider")
    period_range = tuple(period) if period is not None else None

    cat: list[tuple[str, tuple]] = []
    for var in st.session_state.get("cat_filter_vars") or []:
        vals = st.session_state.get(f"catf::{var}")
        if vals:
            cat.append((var, tuple(str(v) for v in vals)))

    rng: list[tuple[str, tuple]] = []
    for var in st.session_state.get("range_filter_vars") or []:
        bounds = st.session_state.get(f"rangef::{var}")
        if bounds is not None:
            rng.append((var, (float(bounds[0]), float(bounds[1]))))

    return period_range, tuple(cat), tuple(rng)


def pipeline_cfg() -> dict:
    """Return the subset / balanced-panel / outlier config from ``session_state``."""
    period_range, cat_filters, range_filters = filter_spec()
    return {
        "period_range": period_range,
        "cat_filters": cat_filters,
        "range_filters": range_filters,
        "outlier_treatment": st.session_state.get("outlier_treatment", "1"),
        "balanced_panel": st.session_state.get("balanced_panel", False),
        "outlier_factor": st.session_state.get("outlier_factor", "None"),
    }


def _udv_frame(
    data_id: str, base_df: pd.DataFrame, entities: list[str], time: str | None
) -> tuple[pd.DataFrame, str | None]:
    """Apply user-defined variables (advanced mode), memoised on ``(data_id, records)``."""
    records = udv_records()
    key = (data_id, records)
    memo = st.session_state.get("_udv_memo")
    if memo is None or memo["key"] != key:
        frame: pd.DataFrame = base_df
        error: str | None = None
        if records:
            try:
                var_def = pd.DataFrame(list(records), columns=["var_name", "var_def"])
                frame = apply_user_vars(base_df, var_def, entities, time)
            except Exception as exc:
                error = str(exc)
                frame = base_df
        memo = {"key": key, "frame": frame, "error": error}
        st.session_state["_udv_memo"] = memo
    return memo["frame"], memo["error"]


def analysis(
    data_id: str,
    base_df: pd.DataFrame,
    entities: list[str],
    time: str | None,
    factor_cutoff: int,
    df_def: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, VarCats, pd.DataFrame, str | None]:
    """Build the analysis sample + variable categories for the current selections.

    Returns ``(analysis_sample, var_cats, pre_subset_frame, udv_error)``. The
    ``pre_subset_frame`` (post user-defined-variables, pre subsetting) is used to populate
    the subset-value choices without collapsing them once a subset is applied. The data
    dictionary (``df_def``) makes its declared ``type`` authoritative for the variable
    categories and enforces ``can_be_na`` (rows missing a required variable are dropped).
    """
    frame, udv_error = _udv_frame(data_id, base_df, entities, time)
    frame = drop_required(frame, df_def)  # enforce the dictionary's can_be_na contract
    types = _types_map(df_def)
    cfg = pipeline_cfg()
    key = (
        data_id,
        udv_records(),
        tuple(sorted(cfg.items())),
        int(factor_cutoff),
        tuple(sorted(types.items())),
        tuple(required_columns(df_def)),
    )
    memo = st.session_state.get("_sample_memo")
    if memo is None or memo["key"] != key:
        sample = build_analysis_sample(frame, entities, time, cfg)
        var_cats = create_var_categories(
            sample, entities, time, factor_cutoff=factor_cutoff, types=types
        )
        memo = {"key": key, "sample": sample, "var_cats": var_cats}
        st.session_state["_sample_memo"] = memo
    return memo["sample"], memo["var_cats"], frame, udv_error
