"""The ExPdPy interactive apps, built with Streamlit.

expdpy ships **three** no-code apps — one per module — that build a multipage UI on top of
the library's ``prepare_*`` / ``sandbox_*`` functions:

* :func:`ExploreApp` — exploratory data analysis (tables, distributions, trends, scatter…),
* :func:`AnalyzeApp` — panel estimators (regression, FWL, panel models, CRE, event study…),
* :func:`LearnApp` — the teaching layer (concept sandboxes and explainers).

All three share the same shell (data upload, the analysis-sample pipeline, config save/load
and reproducible notebook export) and differ only in which pages they expose. Because
Streamlit runs as its own process, each launcher serializes the data you pass to a temporary
bundle and starts ``streamlit run`` in a subprocess (see
:mod:`expdpy.streamlit_app._launcher`), tagging it with the module via the
``EXPDPY_MODULE`` environment variable. The app body is
:func:`expdpy.streamlit_app._entry.run_app`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd

__all__ = [
    "ExploreApp",
    "AnalyzeApp",
    "LearnApp",
    "main_explore",
    "main_analyze",
    "main_learn",
]

_DEFAULT_TITLES = {
    "explore": "ExPdPy — Explore your data!",
    "analyze": "ExPdPy — Analyze your panel!",
    "learn": "ExPdPy — Learn the methods!",
}


def _launch(
    module: str,
    df: Any = None,
    cs_id: Sequence[str] | str | None = None,
    ts_id: str | None = None,
    df_def: pd.DataFrame | None = None,
    var_def: pd.DataFrame | None = None,
    config_list: dict | None = None,
    *,
    title: str | None = None,
    df_name: str | Sequence[str] | None = None,
    components: Any = None,
    factor_cutoff: int = 10,
    export_nb_option: bool = True,
    save_settings_option: bool = True,
    run: bool = True,
    **run_kwargs: Any,
) -> Any:
    """Launch (or describe) one module's ExPdPy app — the shared engine behind the three."""
    from expdpy.streamlit_app import _handoff as handoff
    from expdpy.streamlit_app import _launcher

    samples = handoff.normalize_samples(df, df_name)
    cs_list, ts = handoff.resolve_ids(df_def, cs_id, ts_id)
    return _launcher.launch(
        samples,
        df_def=df_def,
        var_def=var_def,
        cs_list=cs_list,
        ts=ts,
        components=components,
        factor_cutoff=factor_cutoff,
        title=title or _DEFAULT_TITLES[module],
        export_nb_option=export_nb_option,
        save_settings_option=save_settings_option,
        base_cfg=dict(config_list) if config_list else {},
        module=module,
        run=run,
        **run_kwargs,
    )


def ExploreApp(df: Any = None, *args: Any, **kwargs: Any) -> Any:
    """Launch the **Explore** app — exploratory analysis of panel / cross-sectional data.

    Same parameters as the other launchers (``df``, ``cs_id``, ``ts_id``, ``df_def``,
    ``var_def``, ``config_list``, ``title``, ``components``, server ``run_kwargs`` …); see
    the module docstring. Pass a :class:`pandas.DataFrame` (or ``None`` for the dataset
    picker / upload dialog).
    """
    return _launch("explore", df, *args, **kwargs)


def AnalyzeApp(df: Any = None, *args: Any, **kwargs: Any) -> Any:
    """Launch the **Analyze** app — panel estimators, panel models, CRE and event studies."""
    return _launch("analyze", df, *args, **kwargs)


def LearnApp(df: Any = None, *args: Any, **kwargs: Any) -> Any:
    """Launch the **Learn** app — concept sandboxes and the explainer index."""
    return _launch("learn", df, *args, **kwargs)


def main_explore() -> None:
    """Console entry point (``expdpy-explore``): launch the Explore app."""
    ExploreApp(run=True)


def main_analyze() -> None:
    """Console entry point (``expdpy-analyze``): launch the Analyze app."""
    AnalyzeApp(run=True)


def main_learn() -> None:
    """Console entry point (``expdpy-learn``): launch the Learn app."""
    LearnApp(run=True)
