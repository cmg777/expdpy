"""In-memory → subprocess data handoff for the Streamlit ExPdPy app.

Streamlit runs as a separate process, so :func:`expdpy.streamlit_app.ExPdPy`
cannot hand a live DataFrame to the running app directly. Instead the launcher writes a
small *bundle* (the sample(s) as parquet plus a JSON manifest of options) to a temporary
directory and points the app at it via the :data:`EXPDPY_BUNDLE_ENV` environment variable.
The app reads the bundle exactly once on startup (see :mod:`expdpy.streamlit_app._context`).

This module imports neither ``shiny`` nor ``streamlit`` — it is pure and reusable by the
launcher (parent process) and the app (child process) alike.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

# The pure (framework-agnostic) sample-normalisation helpers from the Shiny app are reused
# directly — importing them does not pull in shiny (the import is lazy there).
from expdpy.app import (
    _active_components,
    _normalize_samples,
    _resolve_ids,
)

__all__ = [
    "EXPDPY_BUNDLE_ENV",
    "Bundle",
    "write_bundle",
    "read_bundle",
    "cleanup_bundle",
    "normalize_samples",
    "resolve_ids",
    "active_components",
]

#: Environment variable holding the path to a serialized :class:`Bundle`.
EXPDPY_BUNDLE_ENV = "EXPDPY_STREAMLIT_BUNDLE"

_MANIFEST = "manifest.json"

# Re-export the reused pure helpers under public names so the rest of the package never has
# to reach into the Shiny app's private API.
normalize_samples = _normalize_samples
resolve_ids = _resolve_ids
active_components = _active_components


@dataclass
class Bundle:
    """A deserialized launch bundle: the data plus the resolved app options."""

    samples: dict[str, pd.DataFrame]
    df_def: pd.DataFrame | None = None
    var_def: pd.DataFrame | None = None
    cs_list: list[str] = field(default_factory=list)
    ts: str | None = None
    components: Any = None
    factor_cutoff: int = 10
    title: str = "ExPdPy — Explore your data!"
    export_nb_option: bool = True
    save_settings_option: bool = True
    base_cfg: dict = field(default_factory=dict)


def write_bundle(
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
) -> str:
    """Serialize ``samples`` + options to a fresh temp directory and return its path.

    Samples and the definition frames are written as parquet (preserving dtypes such as
    ordered categoricals); everything else goes into ``manifest.json``.
    """
    path = tempfile.mkdtemp(prefix="expdpy_st_")
    sample_files: dict[str, str] = {}
    for i, (name, frame) in enumerate(samples.items()):
        fname = f"sample_{i}.parquet"
        frame.to_parquet(f"{path}/{fname}", index=False)
        sample_files[name] = fname
    if df_def is not None:
        df_def.to_parquet(f"{path}/df_def.parquet", index=False)
    if var_def is not None:
        var_def.to_parquet(f"{path}/var_def.parquet", index=False)

    manifest = {
        "sample_files": sample_files,
        "has_df_def": df_def is not None,
        "has_var_def": var_def is not None,
        "cs_list": list(cs_list),
        "ts": ts,
        "components": components,
        "factor_cutoff": int(factor_cutoff),
        "title": title,
        "export_nb_option": bool(export_nb_option),
        "save_settings_option": bool(save_settings_option),
        "base_cfg": base_cfg,
    }
    with open(f"{path}/{_MANIFEST}", "w") as fh:
        json.dump(manifest, fh)
    return path


def read_bundle(path: str) -> Bundle:
    """Read a bundle previously written by :func:`write_bundle`."""
    with open(f"{path}/{_MANIFEST}") as fh:
        m = json.load(fh)
    samples = {
        name: pd.read_parquet(f"{path}/{fname}")
        for name, fname in m["sample_files"].items()
    }
    df_def = pd.read_parquet(f"{path}/df_def.parquet") if m.get("has_df_def") else None
    var_def = (
        pd.read_parquet(f"{path}/var_def.parquet") if m.get("has_var_def") else None
    )
    return Bundle(
        samples=samples,
        df_def=df_def,
        var_def=var_def,
        cs_list=list(m.get("cs_list", [])),
        ts=m.get("ts"),
        components=m.get("components"),
        factor_cutoff=int(m.get("factor_cutoff", 10)),
        title=m.get("title", "ExPdPy — Explore your data!"),
        export_nb_option=bool(m.get("export_nb_option", True)),
        save_settings_option=bool(m.get("save_settings_option", True)),
        base_cfg=m.get("base_cfg", {}),
    )


def cleanup_bundle(path: str | None) -> None:
    """Remove a bundle directory, ignoring errors (best-effort)."""
    if path:
        shutil.rmtree(path, ignore_errors=True)
