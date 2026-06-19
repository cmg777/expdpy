"""Resolve the app's data source and launch options.

Three startup paths are supported, in priority order:

1. **Launch bundle** — :func:`expdpy.streamlit_app.ExPdPy` wrote a bundle and pointed
   us at it via :data:`expdpy.streamlit_app._handoff.EXPDPY_BUNDLE_ENV`. The bundle is read
   exactly once (its temp directory may be cleaned up afterwards).
2. **Bundled-dataset picker** — no bundle (e.g. on Streamlit Community Cloud): the user picks
   one of the example datasets shipped with the package.
3. **Upload** — the user uploads their own file, which overrides the selected dataset.

The resolved :class:`AppContext` is cached in ``st.session_state`` so the bundle is never
re-read on rerun.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import streamlit as st

from expdpy.app._state import parse_config
from expdpy.data import (
    load_gapminder,
    load_gapminder_data_def,
    load_kuznets,
    load_kuznets_data_def,
    load_staggered_did,
    load_staggered_did_data_def,
)
from expdpy.streamlit_app import _handoff as handoff

__all__ = ["AppContext", "DATASETS", "resolve_context"]

#: Example datasets offered by the picker: name → (data loader, data-def loader).
#: Kuznets is listed first so it is the picker's default selection.
DATASETS: dict[str, tuple[Callable[[], pd.DataFrame], Callable[[], pd.DataFrame]]] = {
    "Kuznets": (load_kuznets, load_kuznets_data_def),
    "Gapminder": (load_gapminder, load_gapminder_data_def),
    "Staggered DiD": (load_staggered_did, load_staggered_did_data_def),
}

_DEFAULT_TITLE = "ExPdPy — Explore your data!"


@dataclass
class AppContext:
    """Static, per-session configuration resolved once at startup."""

    samples: dict[str, pd.DataFrame] = field(default_factory=dict)
    df_def: pd.DataFrame | None = None
    var_def: pd.DataFrame | None = None
    cs_list: list[str] = field(default_factory=list)
    ts: str | None = None
    components: Any = None
    factor_cutoff: int = 10
    title: str = _DEFAULT_TITLE
    export_nb_option: bool = True
    save_settings_option: bool = True
    base_cfg: dict = field(default_factory=dict)
    #: True when no samples were provided at launch (offer the dataset picker).
    allow_dataset_picker: bool = True


def _from_bundle(path: str) -> AppContext:
    bundle = handoff.read_bundle(path)
    return AppContext(
        samples=bundle.samples,
        df_def=bundle.df_def,
        var_def=bundle.var_def,
        cs_list=bundle.cs_list,
        ts=bundle.ts,
        components=bundle.components,
        factor_cutoff=bundle.factor_cutoff,
        title=bundle.title,
        export_nb_option=bundle.export_nb_option,
        save_settings_option=bundle.save_settings_option,
        base_cfg=parse_config(bundle.base_cfg),
        allow_dataset_picker=False,
    )


def resolve_context() -> AppContext:
    """Resolve (and cache) the :class:`AppContext` for this session."""
    if "_ctx" in st.session_state:
        return st.session_state["_ctx"]

    bundle_path = os.environ.get(handoff.EXPDPY_BUNDLE_ENV)
    if bundle_path and os.path.isdir(bundle_path):
        ctx = _from_bundle(bundle_path)
    else:
        ctx = AppContext(base_cfg=parse_config(None), allow_dataset_picker=True)

    st.session_state["_ctx"] = ctx
    return ctx
