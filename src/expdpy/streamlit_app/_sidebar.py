"""The global sidebar: data source, sample pipeline, configuration I/O and export.

Rendered once per run (before :func:`streamlit.navigation`) so it appears on every page. It
resolves the active data source (built-in dataset, launch sample, or upload), drives the
subset / outlier / user-defined-variable pipeline, and offers config save/load + reproducible
export. The resolved :class:`Active` bundle is stashed in ``st.session_state`` for the pages.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass

import pandas as pd
import streamlit as st

from expdpy.streamlit_app import _handoff as handoff
from expdpy.streamlit_app import _pipeline as pipeline
from expdpy.streamlit_app import _widgets as w
from expdpy.streamlit_app._appcore import _CONFIG_INPUT_KEYS
from expdpy.streamlit_app._config_io import dump_config, load_config
from expdpy.streamlit_app._context import DATASETS, AppContext
from expdpy.streamlit_app._export_nb import build_export_zip
from expdpy.streamlit_app._state import parse_config
from expdpy.streamlit_app._upload import read_uploaded
from expdpy.streamlit_app._varcat import VarCats

__all__ = [
    "Active",
    "render_sidebar",
    "get_active",
    "current_config",
    "apply_pending_config",
]


@dataclass
class Active:
    """The active dataset + prepared analysis sample for the current run."""

    source_name: str
    data_id: str
    base_df: pd.DataFrame
    df_def: pd.DataFrame | None
    cs_list: list[str]
    ts: str | None
    sample: pd.DataFrame
    var_cats: VarCats
    active_components: list[str]


def get_active() -> Active | None:
    """Return the :class:`Active` bundle stashed by :func:`render_sidebar` (for pages)."""
    return st.session_state.get("_active")


# --------------------------------------------------------------------------- data source ---
@st.cache_data(show_spinner=False)
def _load_dataset(name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    loader, def_loader = DATASETS[name]
    return loader(), def_loader()


def _parse_upload(up) -> tuple[pd.DataFrame, str]:
    """Parse an uploaded file once, caching by its Streamlit ``file_id``."""
    cache = st.session_state.get("_upload_cache", {})
    if cache.get("id") != up.file_id:
        suffix = os.path.splitext(up.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(up.getvalue())
            path = tmp.name
        try:
            df = read_uploaded(path, up.name)
        finally:
            os.unlink(path)
        cache = {"id": up.file_id, "name": up.name, "df": df}
        st.session_state["_upload_cache"] = cache
    return cache["df"], cache["name"]


def _resolve_source(ctx: AppContext):
    """Return ``(source_name, data_id, base_df, df_def, cs_list, ts)`` for the selection."""
    st.subheader("Data")
    built_in = list(ctx.samples) if ctx.samples else list(DATASETS)
    selected = w.selectbox("Dataset", built_in, key="sample") if built_in else None

    up = st.file_uploader(
        "…or upload your own",
        type=["csv", "xlsx", "xls", "parquet"],
        key="data_upload",
        help="CSV, Excel or Parquet. An upload overrides the dataset above.",
    )
    if up is not None:
        df, name = _parse_upload(up)
        st.caption(f"Using uploaded file **{name}** ({len(df):,} rows).")
        return f"upload:{up.file_id}", f"upload:{up.file_id}", df, None, [], None

    name = str(selected)
    if ctx.samples:  # launch bundle: all samples share the fixed panel definition
        return (
            name,
            f"bundle:{name}",
            ctx.samples[name],
            ctx.df_def,
            ctx.cs_list,
            ctx.ts,
        )
    base_df, df_def = _load_dataset(name)
    cs_list, ts = handoff.resolve_ids(df_def, None, None)
    return name, f"ds:{name}", base_df, df_def, cs_list, ts


# ------------------------------------------------------------------------------ pipeline ---
def _render_pipeline(active: Active, pre_subset: pd.DataFrame) -> None:
    st.subheader("Sample")
    factors = ["Full Sample", *active.var_cats.grouping]
    sf = w.selectbox("Subset by", factors, key="subset_factor")
    if sf not in (None, "Full Sample") and sf in pre_subset.columns:
        levels = [str(v) for v in sorted(pre_subset[sf].dropna().unique(), key=str)]
        w.selectbox("Value", ["All", *levels], key="subset_value")
    else:
        st.session_state["subset_value"] = "All"

    opts = list(pipeline.OUTLIER_CHOICES)
    if st.session_state.get("outlier_treatment") not in opts:
        st.session_state["outlier_treatment"] = "1"
    st.selectbox(
        "Outlier treatment",
        opts,
        key="outlier_treatment",
        format_func=lambda k: pipeline.OUTLIER_CHOICES[k],
    )


def _render_udv(active: Active, udv_error: str | None) -> None:
    with st.expander("Advanced: user-defined variables"):
        st.caption(
            "Define new variables with safe expressions (columns and "
            "`isna`/`exp`/`log`/`lag`/`lead`). When any row is filled, the analysis uses "
            "only the defined variables."
        )
        rows = st.session_state.get("udvars") or []
        edited = st.data_editor(
            pd.DataFrame(rows or [{"var_name": "", "var_def": ""}]),
            num_rows="dynamic",
            width="stretch",
            key="udvars_editor",
            column_config={
                "var_name": st.column_config.TextColumn("Variable"),
                "var_def": st.column_config.TextColumn("Expression"),
            },
        )
        st.session_state["udvars"] = edited.to_dict("records")
        if udv_error:
            st.error(f"User-defined variable error: {udv_error}")


# -------------------------------------------------------------------------- config & export ---
def current_config(ctx: AppContext) -> dict:
    """Snapshot the current selections as a config dict."""
    cfg = parse_config(ctx.base_cfg)
    for key in _CONFIG_INPUT_KEYS:
        if key in st.session_state and st.session_state[key] is not None:
            value = st.session_state[key]
            cfg[key] = list(value) if isinstance(value, tuple) else value
    cfg["sample"] = st.session_state.get("sample")
    cfg["udvars"] = [
        r for r in (st.session_state.get("udvars") or []) if r.get("var_name")
    ]
    return cfg


def apply_pending_config(ctx: AppContext) -> None:
    """Apply a just-loaded configuration to ``session_state`` before widgets render."""
    cfg = st.session_state.pop("_pending_cfg", None)
    if not cfg:
        return
    for key in [*_CONFIG_INPUT_KEYS, "sample"]:
        if key not in cfg:
            continue
        value = cfg[key]
        if key == "reg_x" and isinstance(value, str):
            value = [] if value in ("None", "") else [value]
        if key == "hist_nr_of_breaks":
            value = max(5, min(100, int(value)))
        st.session_state[key] = value
    if isinstance(cfg.get("udvars"), list):
        st.session_state["udvars"] = cfg["udvars"]


def _render_io(ctx: AppContext, active: Active) -> None:
    if ctx.save_settings_option:
        st.divider()
        st.download_button(
            "💾 Save config",
            data=dump_config(current_config(ctx), None),
            file_name="expdpy_config.json",
            mime="application/json",
            width="stretch",
        )
        cfg_file = st.file_uploader(
            "Load config", type=["json", "cfg"], key="config_upload"
        )
        if (
            cfg_file is not None
            and st.session_state.get("_cfg_file_id") != cfg_file.file_id
        ):
            st.session_state["_cfg_file_id"] = cfg_file.file_id
            try:
                st.session_state["_pending_cfg"] = parse_config(
                    load_config(cfg_file.getvalue(), None)
                )
                st.rerun()
            except Exception as exc:
                st.error(f"Could not load config: {exc}")

    if ctx.export_nb_option:
        st.divider()
        st.download_button(
            "📓 Export notebook + data",
            data=build_export_zip(
                current_config(ctx), active.active_components, active.sample, active.ts
            ),
            file_name="ExPdPy_analysis.zip",
            mime="application/zip",
            width="stretch",
        )


# --------------------------------------------------------------------------------- entry ---
def render_sidebar(ctx: AppContext) -> Active:
    """Render the global sidebar and return the active dataset + prepared sample."""
    with st.sidebar:
        st.title("ExPdPy")
        source_name, data_id, base_df, df_def, cs_list, ts = _resolve_source(ctx)

        sample, var_cats, pre_subset, udv_error = pipeline.analysis(
            data_id, base_df, cs_list, ts, ctx.factor_cutoff
        )
        active = Active(
            source_name=source_name,
            data_id=data_id,
            base_df=base_df,
            df_def=df_def,
            cs_list=cs_list,
            ts=ts,
            sample=sample,
            var_cats=var_cats,
            active_components=handoff.active_components(ctx.components, ts),
        )

        _render_pipeline(active, pre_subset)
        _render_udv(active, udv_error)
        _render_io(ctx, active)

    st.session_state["_active"] = active
    return active
