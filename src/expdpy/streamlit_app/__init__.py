"""The ExPdPy interactive app, built with Streamlit.

``ExPdPy`` is the Streamlit counterpart of :func:`expdpy.app.ExPdPy`: it builds the
same no-code, multipage exploration UI on top of the library's ``prepare_*`` functions, with
native Streamlit tables, an analysis-sample pipeline (subset / outlier treatment /
user-defined variables), config save/load, and reproducible notebook export — and runs both
locally and on Streamlit Community Cloud.

Because Streamlit runs as its own process, ``ExPdPy`` serializes the data you pass to
a temporary bundle and starts ``streamlit run`` in a subprocess (see
:mod:`expdpy.streamlit_app._launcher`). The app body itself is
:func:`expdpy.streamlit_app._entry.run_app`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd

__all__ = ["ExPdPy", "main"]

_DEFAULT_TITLE = "ExPdPy — Explore your data!"


def ExPdPy(
    df: Any = None,
    cs_id: Sequence[str] | str | None = None,
    ts_id: str | None = None,
    df_def: pd.DataFrame | None = None,
    var_def: pd.DataFrame | None = None,
    config_list: dict | None = None,
    *,
    title: str = _DEFAULT_TITLE,
    df_name: str | Sequence[str] | None = None,
    components: Any = None,
    factor_cutoff: int = 10,
    export_nb_option: bool = True,
    save_settings_option: bool = True,
    run: bool = True,
    **run_kwargs: Any,
) -> Any:
    """Launch (or describe) the interactive Streamlit ExPdPy app.

    Parameters
    ----------
    df
        A :class:`pandas.DataFrame`, a mapping of name->DataFrame, a list of DataFrames, or
        ``None`` to start with the bundled-dataset picker / upload dialog.
    cs_id, ts_id
        Cross-sectional / time-series identifier column name(s). Overridden by ``df_def``.
    df_def
        Optional variable-definition frame (columns ``var_name``, ``var_def``, ``type``) used
        to identify the panel dimensions.
    var_def
        Optional analysis-sample variable definitions (advanced mode).
    config_list
        Optional startup configuration (see :func:`expdpy.data.get_config`).
    title
        App / browser-tab title.
    df_name
        Display name(s) for the provided sample(s).
    components
        Ordered list (or ``{name: bool}`` mapping) selecting which components to show.
    factor_cutoff
        Numeric columns with at most this many unique values are treated as factors.
    export_nb_option, save_settings_option
        Enable the notebook-export and config save/load controls.
    run
        If ``True`` (default), start ``streamlit run`` in a subprocess and block until it
        exits; if ``False``, return the command list without launching (used for testing).
    **run_kwargs
        Streamlit server options, e.g. ``port``, ``host``, ``headless``, ``launch_browser``,
        ``max_upload_size`` (mapped to the corresponding ``--server.*`` flags).

    Returns
    -------
    subprocess.Popen | list[str]
        The launched process (``run=True``) or the command line (``run=False``).
    """
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
        title=title,
        export_nb_option=export_nb_option,
        save_settings_option=save_settings_option,
        base_cfg=dict(config_list) if config_list else {},
        run=run,
        **run_kwargs,
    )


def main() -> None:
    """Console entry point (``expdpy-streamlit``): launch with the dataset picker."""
    ExPdPy(run=True)
