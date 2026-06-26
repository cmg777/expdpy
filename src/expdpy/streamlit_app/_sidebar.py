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
from pandas.api import types as pdt

from expdpy import build_data_def, resolve_label, set_labels, set_panel
from expdpy.data import _normalize_def
from expdpy.streamlit_app import _handoff as handoff
from expdpy.streamlit_app import _pipeline as pipeline
from expdpy.streamlit_app import _widgets as w
from expdpy.streamlit_app._appcore import _CONFIG_INPUT_KEYS
from expdpy.streamlit_app._config_io import dump_config, load_config
from expdpy.streamlit_app._context import DATASETS, AppContext
from expdpy.streamlit_app._export_nb import build_export_zip
from expdpy.streamlit_app._state import parse_config
from expdpy.streamlit_app._upload import read_uploaded
from expdpy.streamlit_app._varcat import VarCats, create_var_categories

#: The five columns of a data dictionary (df_def), in order.
_DDEF_COLUMNS = ["var_name", "var_def", "label", "type", "can_be_na"]

#: The allowed ``type`` values for a data dictionary (df_def).
_DDEF_TYPES = ["entity", "time", "factor", "logical", "numeric"]

#: Stored-widget values that are *not* column names and so survive a dataset switch.
_RESET_SKIP_VALUES = {"None", "All", "Full Sample", "1", "2", "3", "4", "5"}

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
    entities: list[str]
    #: The *effective* time id: the declared time column, or ``None`` when the active sample
    #: spans a single period (so panel/over-time views fall back to cross-sectional gating).
    time: str | None
    sample: pd.DataFrame
    var_cats: VarCats
    active_components: list[str]
    #: The post-UDV, pre-subset working frame (the full data the filters narrow) — used by the
    #: reproducible export so the notebook can rebuild the sample from the unfiltered data.
    working: pd.DataFrame | None = None
    #: The declared time column even when the sample is a single-period cross-section.
    panel_time: str | None = None


def get_active() -> Active | None:
    """Return the :class:`Active` bundle stashed by :func:`render_sidebar` (for pages)."""
    return st.session_state.get("_active")


# --------------------------------------------------------------------------- data source ---
@st.cache_data(show_spinner=False)
def _load_dataset(name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    loader, def_loader = DATASETS[name]
    return loader(), def_loader()


def _parse_file(up, cache_key: str) -> pd.DataFrame:
    """Parse an uploaded file once, caching it under ``cache_key`` by Streamlit ``file_id``."""
    cache = st.session_state.get(cache_key, {})
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
        st.session_state[cache_key] = cache
    return cache["df"]


def _editable_data_def(
    df: pd.DataFrame, data_id: str, factor_cutoff: int
) -> pd.DataFrame:
    """Show an editable auto-built data dictionary; return the *applied* df_def.

    The dictionary is inferred once per ``data_id`` (so a new upload re-guesses), shown in an
    ``st.data_editor``, and only drives the analysis once the user clicks **Apply** — so edits
    don't churn every figure mid-typing.
    """
    if st.session_state.get("_ddef_seed_id") != data_id:
        guess = build_data_def(df, factor_cutoff=factor_cutoff).to_dict("records")
        st.session_state["_ddef_rows"] = guess
        st.session_state["_ddef_applied"] = guess
        st.session_state["_ddef_seed_id"] = data_id
        st.session_state.pop("ddef_editor", None)  # clear stale editor deltas

    with st.expander(
        "Data dictionary (auto-detected — edit, then Apply)", expanded=True
    ):
        st.caption(
            "Auto-detected from your data. Set one row's type to **entity** and one to "
            "**time** to unlock the panel views, tidy the labels, then **Apply**."
        )
        edited = st.data_editor(
            pd.DataFrame(st.session_state["_ddef_rows"], columns=_DDEF_COLUMNS),
            key="ddef_editor",
            num_rows="fixed",
            width="stretch",
            hide_index=True,
            column_config={
                "var_name": st.column_config.TextColumn("Variable", disabled=True),
                "label": st.column_config.TextColumn("Label"),
                "var_def": st.column_config.TextColumn("Description"),
                "type": st.column_config.SelectboxColumn("Type", options=_DDEF_TYPES),
                "can_be_na": st.column_config.CheckboxColumn("Can be NA"),
            },
        )
        c1, c2 = st.columns(2)
        if c1.button("Apply", key="ddef_apply", width="stretch"):
            rows = edited.to_dict("records")
            bad = sorted({str(r.get("type")) for r in rows} - set(_DDEF_TYPES))
            if bad:
                st.warning(
                    f"Unknown type(s) {', '.join(bad)} — those rows stay untyped."
                )
            st.session_state["_ddef_rows"] = rows
            st.session_state["_ddef_applied"] = rows
            st.rerun()
        c2.download_button(
            "Download .csv",
            data=pd.DataFrame(st.session_state["_ddef_applied"], columns=_DDEF_COLUMNS)
            .to_csv(index=False)
            .encode(),
            file_name="expdpy_data_def.csv",
            mime="text/csv",
            width="stretch",
        )

    applied = pd.DataFrame(st.session_state["_ddef_applied"], columns=_DDEF_COLUMNS)
    return _normalize_def(applied)


def _apply_labels_panel(df: pd.DataFrame, df_def: pd.DataFrame | None) -> pd.DataFrame:
    """Attach the dictionary's labels and (when the id columns survive) declare the panel.

    Mutates ``df.attrs`` in place (idempotent) and returns ``df``. A malformed dictionary
    degrades gracefully to bare column names rather than crashing the app.
    """
    if df_def is None or "var_name" not in getattr(df_def, "columns", []):
        return df
    try:
        set_labels(df, df_def)
        entities, time = handoff.resolve_ids(df_def, None, None)
        kw: dict[str, str] = {}
        entity = entities[0] if entities else None
        if entity and entity in df.columns:
            kw["entity"] = entity
        if time and time in df.columns:
            kw["time"] = time
        if kw:
            set_panel(df, **kw)
    except Exception as exc:  # malformed dictionary → keep raw names, never crash
        st.warning(f"Could not apply the data dictionary: {exc}")
    return df


def _reset_stale_inputs(data_id: str, columns) -> None:
    """On a genuine data switch, drop stored selections that name a now-absent column.

    Layers on top of the per-widget coercion in :mod:`expdpy.streamlit_app._widgets` so the
    raw ``st.selectbox`` call sites (cluster, event-study, panel-model selectors) also re-seed
    cleanly when the dataset changes.
    """
    if st.session_state.get("_data_id") == data_id:
        return
    st.session_state["_data_id"] = data_id
    cols = {str(c) for c in columns}
    for key in _CONFIG_INPUT_KEYS:
        val = st.session_state.get(key)
        if isinstance(val, str):
            if val and val not in cols and val not in _RESET_SKIP_VALUES:
                del st.session_state[key]
        elif isinstance(val, (list, tuple)):
            kept = [v for v in val if not isinstance(v, str) or v in cols]
            if list(kept) != list(val):
                st.session_state[key] = kept
    # The sample filters reference this dataset's columns/levels/ranges — clear them outright
    # (period slider bounds and category levels differ across datasets).
    _clear_filter_state()


def _clear_filter_state() -> None:
    """Remove all sample-filter widget state (period / category / range)."""
    for key in (
        "period_slider",
        "cat_filter_vars",
        "range_filter_vars",
    ):
        st.session_state.pop(key, None)
    for key in [
        k
        for k in st.session_state
        if isinstance(k, str) and (k.startswith("catf::") or k.startswith("rangef::"))
    ]:
        st.session_state.pop(key, None)


def _resolve_source(ctx: AppContext):
    """Return ``(source_name, data_id, base_df, df_def, entities, time)`` for the selection."""
    st.subheader("Data")
    built_in = list(ctx.samples) if ctx.samples else list(DATASETS)
    selected = w.selectbox("Dataset", built_in, key="sample") if built_in else None

    up = st.file_uploader(
        "…or upload your own data",
        type=["csv", "xlsx", "xls", "parquet"],
        key="data_upload",
        help="CSV, Excel or Parquet. An upload overrides the dataset above.",
    )
    if up is not None:
        df = _parse_file(up, "_upload_cache")
        st.caption(f"Using uploaded file **{up.name}** ({len(df):,} rows).")
        dict_up = st.file_uploader(
            "…and its data dictionary (optional)",
            type=["csv", "xlsx", "xls", "parquet"],
            key="dict_upload",
            help=(
                "A df_def with columns var_name / var_def / label / type / can_be_na. "
                "Leave empty to auto-build an editable one below."
            ),
        )
        if dict_up is not None:
            ddef = _normalize_def(_parse_file(dict_up, "_dict_upload_cache"))
            st.caption(f"Using dictionary **{dict_up.name}**.")
            entities, time = handoff.resolve_ids(ddef, None, None)
            data_id = f"upload:{up.file_id}|dict:{dict_up.file_id}"
            return up.name, data_id, df, ddef, entities, time
        data_id = f"upload:{up.file_id}"
        ddef = _editable_data_def(df, data_id, ctx.factor_cutoff)
        entities, time = handoff.resolve_ids(ddef, None, None)
        return up.name, data_id, df, ddef, entities, time

    name = str(selected)
    if ctx.samples:  # launch bundle: all samples share the fixed panel definition
        return (
            name,
            f"bundle:{name}",
            ctx.samples[name],
            ctx.df_def,
            ctx.entities,
            ctx.time,
        )
    base_df, df_def = _load_dataset(name)
    entities, time = handoff.resolve_ids(df_def, None, None)
    return name, f"ds:{name}", base_df, df_def, entities, time


# ------------------------------------------------------------------------------ pipeline ---
def _range_slider(label: str, lo, hi, key: str, *, as_int: bool):
    """Render a range ``st.slider`` over ``[lo, hi]`` respecting a stored ``session_state`` value."""
    lo, hi = (int(lo), int(hi)) if as_int else (float(lo), float(hi))
    if key in st.session_state:
        cur = st.session_state[key]  # clamp a stale stored value into the new bounds
        st.session_state[key] = (max(lo, min(hi, cur[0])), max(lo, min(hi, cur[1])))
        return st.slider(label, lo, hi, key=key)
    return st.slider(label, lo, hi, value=(lo, hi), key=key)


def _candidate_cats(pre_subset, entities, time, factor_cutoff) -> tuple[list, list]:
    """Return ``(category_vars, range_vars)`` from the *pre-subset* frame.

    Computing candidates from the unfiltered frame is what keeps a selected factor from
    vanishing once the sample collapses it to a single value.
    """
    vc = create_var_categories(pre_subset, entities, time, factor_cutoff=factor_cutoff)
    cat_vars = list(vc.grouping)
    ids = {*entities, time}
    range_vars = [
        c
        for c in pre_subset.columns
        if c not in ids
        and pdt.is_numeric_dtype(pre_subset[c])
        and not pdt.is_bool_dtype(pre_subset[c])
        and pre_subset[c].dropna().nunique() > 1
    ]
    return cat_vars, range_vars


def _render_sample(
    active: Active,
    pre_subset: pd.DataFrame,
    time: str | None,
    factor_cutoff: int,
) -> None:
    """Render the sample-selection menu: period, category and range filters + outliers."""
    st.subheader("Sample")
    cat_vars, range_vars = _candidate_cats(
        pre_subset, active.entities, time, factor_cutoff
    )

    # --- Period (time) sub-sampling -------------------------------------------------------
    if time and time in pre_subset.columns:
        periods = pre_subset[time].dropna()
        if pdt.is_numeric_dtype(periods) and periods.nunique() > 1:
            lo, hi = _range_slider(
                "Period",
                periods.min(),
                periods.max(),
                key="period_slider",
                as_int=bool(pdt.is_integer_dtype(periods)),
            )
            st.caption(
                f"Single period ({lo}) — cross-sectional analysis."
                if lo == hi
                else "Drag the handles together for a single-year cross-section."
            )

    # --- Filter by category ----------------------------------------------------------------
    chosen_cats = w.multiselect(
        "Filter by category",
        cat_vars,
        key="cat_filter_vars",
        help="Keep rows whose value is one of the categories you pick for each variable.",
    )
    for var in chosen_cats:
        levels = [str(v) for v in sorted(pre_subset[var].dropna().unique(), key=str)]
        w.multiselect(
            f"{_label(active, var)} is any of",
            levels,
            key=f"catf::{var}",
            default=levels,
        )

    # --- Filter by range -------------------------------------------------------------------
    chosen_ranges = w.multiselect(
        "Filter by range",
        range_vars,
        key="range_filter_vars",
        help="Keep rows whose value falls inside the range you set for each variable.",
    )
    for var in chosen_ranges:
        col = pre_subset[var].dropna()
        _range_slider(
            f"{_label(active, var)} range",
            col.min(),
            col.max(),
            key=f"rangef::{var}",
            as_int=bool(pdt.is_integer_dtype(col)),
        )

    # --- Outlier treatment -----------------------------------------------------------------
    opts = list(pipeline.OUTLIER_CHOICES)
    if st.session_state.get("outlier_treatment") not in opts:
        st.session_state["outlier_treatment"] = "1"
    st.selectbox(
        "Outlier treatment",
        opts,
        key="outlier_treatment",
        format_func=lambda k: pipeline.OUTLIER_CHOICES[k],
    )


def _label(active: Active, name: str) -> str:
    """Human-readable label for ``name`` from the active sample's stored labels."""
    return resolve_label(active.sample, name)


def _active_filter_summary() -> tuple[list[str], list[str]]:
    """Return ``(category_lines, range_lines)`` describing the active filters for the summary."""
    cat_lines = []
    for var in st.session_state.get("cat_filter_vars") or []:
        vals = st.session_state.get(f"catf::{var}")
        if vals:
            shown = ", ".join(str(v) for v in vals[:3]) + ("…" if len(vals) > 3 else "")
            cat_lines.append(f"{var} ∈ {{{shown}}}")
    range_lines = []
    for var in st.session_state.get("range_filter_vars") or []:
        b = st.session_state.get(f"rangef::{var}")
        if b is not None:
            range_lines.append(f"{var} ∈ [{b[0]:g}, {b[1]:g}]")
    return cat_lines, range_lines


def _render_active_summary(active: Active, pre_subset: pd.DataFrame) -> None:
    """Always-visible summary of the sample being analyzed (dataset, period, filters, n/N)."""
    n, total = len(active.sample), len(pre_subset)
    period = st.session_state.get("period_slider")
    with st.container(border=True):
        st.markdown(f"**Active sample** · {active.source_name}")
        if active.panel_time and period:
            tag = period[0] if period[0] == period[1] else f"{period[0]}-{period[1]}"
            st.caption(f"{active.panel_time}: {tag}")
        cat_lines, range_lines = _active_filter_summary()
        for line in (*cat_lines, *range_lines):
            st.caption(line)
        st.caption(f"Rows kept: **{n:,} / {total:,}**")
        narrowed = bool(
            n < total or cat_lines or range_lines or (period and period[0] != period[1])
        )
        if narrowed and st.button(
            "Reset filters", key="reset_filters", width="stretch"
        ):
            _clear_filter_state()
            st.rerun()


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
    # Sample filters (JSON-serializable) — also consumed by the reproducible export.
    period = st.session_state.get("period_slider")
    cfg["period_range"] = list(period) if period is not None else None
    cat_filters: dict[str, list] = {}
    for var in st.session_state.get("cat_filter_vars") or []:
        vals = st.session_state.get(f"catf::{var}")
        if vals:
            cat_filters[var] = list(vals)
    cfg["cat_filters"] = cat_filters
    range_filters: dict[str, list] = {}
    for var in st.session_state.get("range_filter_vars") or []:
        bounds = st.session_state.get(f"rangef::{var}")
        if bounds is not None:
            range_filters[var] = [float(b) for b in bounds]
    cfg["range_filters"] = range_filters
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
    _apply_filter_config(cfg)


def _apply_filter_config(cfg: dict) -> None:
    """Seed the sample-filter widgets from a loaded config (migrating the legacy subset)."""
    _clear_filter_state()
    period = cfg.get("period_range")
    if period:
        st.session_state["period_slider"] = tuple(period)

    cat = dict(cfg.get("cat_filters") or {})
    if not cat:  # migrate a legacy single-factor subset (subset_factor == subset_value)
        sf, sv = cfg.get("subset_factor"), cfg.get("subset_value")
        if sf not in (None, "Full Sample") and sv not in (None, "All"):
            cat = {sf: [sv]}
    if cat:
        st.session_state["cat_filter_vars"] = list(cat)
        for var, vals in cat.items():
            st.session_state[f"catf::{var}"] = list(vals)

    rng = dict(cfg.get("range_filters") or {})
    if rng:
        st.session_state["range_filter_vars"] = list(rng)
        for var, bounds in rng.items():
            st.session_state[f"rangef::{var}"] = tuple(bounds)


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
        # Export the *working* frame (pre-subset) so the notebook reproduces the filters as
        # code; fall back to the prepared sample if the working frame is unavailable.
        export_df = active.working if active.working is not None else active.sample
        st.download_button(
            "📓 Export notebook + data",
            data=build_export_zip(
                current_config(ctx),
                active.active_components,
                export_df,
                active.panel_time,
                active.df_def,
                active.entities,
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
        source_name, data_id, base_df, df_def, entities, time = _resolve_source(ctx)
        _reset_stale_inputs(data_id, base_df.columns)
        base_df = _apply_labels_panel(base_df, df_def)

        sample, var_cats, pre_subset, udv_error = pipeline.analysis(
            data_id, base_df, entities, time, ctx.factor_cutoff
        )
        # Re-attach labels/panel to the prepared sample: the subset/outlier steps preserve
        # ``attrs``, but the user-defined-variables path builds a fresh frame that drops them.
        sample = _apply_labels_panel(sample, df_def)

        # When the period filter collapses the sample to a single period it is no longer a
        # panel — fall back to the cross-sectional gating (``effective_time = None``) so the
        # over-time / panel views hide rather than error.
        n_periods = (
            int(sample[time].nunique()) if time and time in sample.columns else 0
        )
        effective_time = time if n_periods >= 2 else None

        active = Active(
            source_name=source_name,
            data_id=data_id,
            base_df=base_df,
            df_def=df_def,
            entities=entities,
            time=effective_time,
            sample=sample,
            var_cats=var_cats,
            active_components=handoff.active_components(ctx.components, effective_time),
            working=pre_subset,
            panel_time=time,
        )

        _render_active_summary(active, pre_subset)
        if time and effective_time is None and n_periods == 1:
            st.warning(
                "Single period selected → cross-sectional analysis (panel views hidden)."
            )
        _render_sample(active, pre_subset, time, ctx.factor_cutoff)
        _render_udv(active, udv_error)
        _render_io(ctx, active)

    st.session_state["_active"] = active
    return active
