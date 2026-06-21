"""Framework-agnostic helpers shared across the Streamlit app's modules.

These small pure helpers (sample normalization, id resolution, component selection, cluster
translation, and the list of config-bearing input keys) were previously defined alongside the
interactive app; they carry no UI-framework dependency and are reused throughout
:mod:`expdpy.streamlit_app`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from expdpy.streamlit_app._components import (
    COMPONENT_KIND,
    COMPONENT_ORDER,
    TS_COMPONENTS,
)

__all__ = [
    "_CONFIG_INPUT_KEYS",
    "_active_components",
    "_cluster_vars",
    "_normalize_samples",
    "_resolve_ids",
]


def _normalize_samples(
    df: Any, df_name: str | Sequence[str] | None
) -> dict[str, pd.DataFrame]:
    if df is None:
        return {}
    if isinstance(df, pd.DataFrame):
        name = df_name if isinstance(df_name, str) else "Sample"
        return {name: df}
    if isinstance(df, Mapping):
        return dict(df)
    return {f"Sample {i + 1}": d for i, d in enumerate(df)}


def _resolve_ids(
    df_def: pd.DataFrame | None, entity: Sequence[str] | str | None, time: str | None
) -> tuple[list[str], str | None]:
    if df_def is not None:
        entities = list(df_def.loc[df_def["type"] == "entity", "var_name"])
        time_rows = list(df_def.loc[df_def["type"] == "time", "var_name"])
        return entities, (time_rows[0] if time_rows else None)
    if isinstance(entity, str):
        entity = [entity]
    return (list(entity) if entity else []), time


def _active_components(components: Any, time: str | None) -> list[str]:
    if isinstance(components, Mapping):
        selected = [c for c in COMPONENT_ORDER if components.get(c)]
    elif isinstance(components, (list, tuple)):
        selected = [c for c in components if c in COMPONENT_ORDER]
    else:
        selected = list(COMPONENT_ORDER)
    renderable = [c for c in selected if c in COMPONENT_KIND]
    if not time:
        renderable = [c for c in renderable if c not in TS_COMPONENTS]
    return renderable


def _cluster_vars(choice: Any, fe1: str | None, fe2: str | None) -> list[str]:
    """Translate the cluster radio (1-4) into a list of cluster variables."""
    fes = [f for f in (fe1, fe2) if f and f != "None"]
    try:
        choice = int(choice)
    except (TypeError, ValueError):
        choice = 1
    if choice <= 1:
        return []
    return fes[: choice - 1]


_CONFIG_INPUT_KEYS = [
    "subset_factor",
    "subset_value",
    "outlier_treatment",
    "hist_var",
    "hist_nr_of_breaks",
    "ext_obs_var",
    "bar_chart_var1",
    "scatter_x",
    "scatter_y",
    "scatter_color",
    "scatter_size",
    "scatter_loess",
    "trend_graph_var1",
    "trend_graph_var2",
    "trend_graph_var3",
    "quantile_trend_graph_var",
    "bgbg_byvar",
    "bgbg_var",
    "bgvg_byvar",
    "bgvg_var",
    "bgtg_byvar",
    "bgtg_var",
    "reg_y",
    "reg_x",
    "reg_fe1",
    "reg_fe2",
    "cluster",
    "fwl_focal",
    "es_outcome",
    "es_cohort",
    "es_estimator",
    "pm_dv",
    "pm_idvs",
]
