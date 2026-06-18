"""Launch the Streamlit app in a subprocess, handing over in-memory data via a bundle.

A subprocess (``python -m streamlit run``) is used rather than Streamlit's in-process
bootstrap because the latter installs signal handlers and calls :func:`sys.exit`, which would
hijack the caller's Python session/REPL.
"""

from __future__ import annotations

import atexit
import os
import subprocess
import sys
from importlib import resources
from typing import Any

import pandas as pd

from expdpy.streamlit_app import _handoff as handoff

__all__ = ["launch", "build_command"]

# run_kwarg name → Streamlit CLI flag.
_FLAG_MAP = {
    "port": "--server.port",
    "host": "--server.address",
    "address": "--server.address",
    "base_url_path": "--server.baseUrlPath",
    "max_upload_size": "--server.maxUploadSize",
}


def _entry_script() -> str:
    """Filesystem path to the packaged runnable script Streamlit should execute."""
    return str(resources.files("expdpy.streamlit_app") / "_run.py")


def build_command(entry: str, run_kwargs: dict[str, Any]) -> list[str]:
    """Build the ``streamlit run`` command line from ``run_kwargs``."""
    cmd = [sys.executable, "-m", "streamlit", "run", entry]
    headless = run_kwargs.get("headless")
    if headless is None and run_kwargs.get("launch_browser") is False:
        headless = True
    if headless is not None:
        cmd += ["--server.headless", "true" if headless else "false"]
    for key, flag in _FLAG_MAP.items():
        value = run_kwargs.get(key)
        if value is not None:
            cmd += [flag, str(value)]
    return cmd


def launch(
    samples: dict[str, pd.DataFrame],
    *,
    df_def: pd.DataFrame | None,
    var_def: pd.DataFrame | None,
    cs_list: list[str],
    ts: str | None,
    components: Any,
    factor_cutoff: int,
    title: str,
    export_nb_option: bool,
    save_settings_option: bool,
    base_cfg: dict,
    run: bool = True,
    **run_kwargs: Any,
) -> subprocess.Popen | list[str]:
    """Write the data bundle (if any) and start (or describe) the Streamlit process.

    Returns the :class:`subprocess.Popen` when ``run`` is true, or the command list when
    ``run`` is false (used by the test-suite to assert the invocation without spawning).
    """
    bundle: str | None = None
    if samples:
        bundle = handoff.write_bundle(
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
            base_cfg=base_cfg,
        )
        os.environ[handoff.EXPDPY_BUNDLE_ENV] = bundle
        atexit.register(handoff.cleanup_bundle, bundle)

    cmd = build_command(_entry_script(), run_kwargs)
    if not run:
        return cmd

    proc = subprocess.Popen(cmd)
    try:
        proc.wait()
    except KeyboardInterrupt:  # pragma: no cover - interactive only
        proc.terminate()
    finally:
        handoff.cleanup_bundle(bundle)
    return proc
