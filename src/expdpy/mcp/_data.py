"""Resolve an agent-supplied data handle into a :class:`pandas.DataFrame`.

The MCP tools advertise the shared ``DATA_HANDLE_SCHEMA`` (a ``oneOf`` of a bundled
dataset name, an absolute file path, or inline records). This module turns that handle
into a frame:

* ``{"dataset": "kuznets"}`` loads a bundled dataset and (with ``with_labels``) applies
  its data dictionary and declares the panel, so panel-aware tools work without the agent
  knowing the entity/time column names.
* ``{"path": "/abs/file.parquet"}`` reads a local ``.csv``/``.parquet`` (suffix-allowlisted,
  optionally confined to ``EXPDPY_MCP_DATA_ROOT`` to guard against path traversal).
* ``{"records": [...]}`` builds a frame from inline rows, capped by ``EXPDPY_MCP_MAXROWS``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd

import expdpy
from expdpy import data as _datasets
from expdpy._meta import DATASET_NAMES

_ALLOWED_SUFFIXES = {".csv", ".parquet", ".pq"}


def _max_rows() -> int:
    return int(os.environ.get("EXPDPY_MCP_MAXROWS", "5000"))


def _load_dataset(name: str, with_labels: bool) -> pd.DataFrame:
    if name not in DATASET_NAMES:
        raise ValueError(
            f"unknown dataset {name!r}; choose one of: {', '.join(DATASET_NAMES)}"
        )
    df = getattr(_datasets, f"load_{name}")()
    if with_labels:
        def_loader = getattr(_datasets, f"load_{name}_data_def", None)
        if def_loader is not None:
            # Applies human-readable labels AND declares the panel (entity/time) from the
            # data dictionary's `type` column, so panel tools resolve the panel for free.
            df = expdpy.set_labels(df, def_loader(), set_panel=True)
    return df


def _load_path(path_str: str) -> pd.DataFrame:
    path = Path(path_str).expanduser().resolve()
    root = os.environ.get("EXPDPY_MCP_DATA_ROOT")
    if root:
        allowed = Path(root).expanduser().resolve()
        if not path.is_relative_to(allowed):
            raise ValueError(f"path {path} is outside EXPDPY_MCP_DATA_ROOT ({allowed})")
    if path.suffix.lower() not in _ALLOWED_SUFFIXES:
        raise ValueError(
            f"unsupported file type {path.suffix!r}; use one of {sorted(_ALLOWED_SUFFIXES)}"
        )
    if not path.is_file():
        raise FileNotFoundError(f"no such file: {path}")
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def _load_records(records: list[dict[str, Any]]) -> pd.DataFrame:
    cap = _max_rows()
    if len(records) > cap:
        raise ValueError(
            f"received {len(records)} inline records; the limit is {cap}. "
            "Pass a file path instead for larger datasets."
        )
    return pd.DataFrame.from_records(records)


def resolve_data(handle: dict[str, Any]) -> pd.DataFrame:
    """Resolve a data handle (dataset / path / records) into a DataFrame."""
    if not isinstance(handle, dict):
        raise TypeError(
            "the 'data' argument must be an object (dataset, path or records)"
        )
    if "dataset" in handle:
        return _load_dataset(handle["dataset"], handle.get("with_labels", True))
    if "path" in handle:
        return _load_path(handle["path"])
    if "records" in handle:
        return _load_records(handle["records"])
    raise ValueError("'data' must provide one of: dataset, path, records")
