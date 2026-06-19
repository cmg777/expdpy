"""Bundled example datasets ported from ExPanDaR.

All loaders return :class:`pandas.DataFrame` objects read from parquet files shipped with
the package; :func:`get_config` returns a startup configuration ``dict`` for the ``ExPdPy``
app. The data is provided for didactic purposes only (see the ExPanDaR documentation).
"""

from __future__ import annotations

import json
from functools import cache
from importlib import resources

import pandas as pd

__all__ = [
    "available_configs",
    "get_config",
    "load_gapminder",
    "load_gapminder_data_def",
    "load_kuznets",
    "load_kuznets_data_def",
    "load_staggered_did",
    "load_staggered_did_data_def",
]

_CONFIGS = {
    "kuznets": "expdpy_config_kuznets.json",
}


def _read_parquet(name: str) -> pd.DataFrame:
    with resources.as_file(resources.files("expdpy.data") / f"{name}.parquet") as path:
        return pd.read_parquet(path)


def _normalize_def(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce a data/var-definition frame's ``can_be_na`` column to boolean."""
    if "can_be_na" in df.columns:
        df = df.copy()
        df["can_be_na"] = df["can_be_na"].astype(float).astype(bool)
    return df


def load_gapminder() -> pd.DataFrame:
    """Load the gapminder dataset (life expectancy, population, GDP per capita)."""
    return _read_parquet("gapminder")


def load_gapminder_data_def() -> pd.DataFrame:
    """Return derived variable definitions for :func:`load_gapminder`."""
    return _normalize_def(_read_parquet("gapminder_data_def"))


def load_kuznets() -> pd.DataFrame:
    """Load the synthetic kuznets dataset (country-year, N-shaped regional Kuznets curve)."""
    return _read_parquet("kuznets")


def load_kuznets_data_def() -> pd.DataFrame:
    """Return variable definitions for :func:`load_kuznets`."""
    return _normalize_def(_read_parquet("kuznets_data_def"))


def load_staggered_did() -> pd.DataFrame:
    """Load the synthetic staggered difference-in-differences dataset.

    A balanced unit-year panel with several treatment cohorts (and a never-treated control
    group, ``cohort == 0``) and a known *dynamic* treatment effect — for teaching event-study
    and staggered-DiD methods via :func:`expdpy.prepare_event_study`.
    """
    return _read_parquet("staggered_did")


def load_staggered_did_data_def() -> pd.DataFrame:
    """Return variable definitions for :func:`load_staggered_did`."""
    return _normalize_def(_read_parquet("staggered_did_data_def"))


@cache
def get_config(name: str) -> dict:
    """Return a startup configuration for ``ExPdPy``.

    Parameters
    ----------
    name
        The configuration name (currently only ``"kuznets"``; see
        :func:`available_configs`).

    Returns
    -------
    dict
        The configuration mapping.
    """
    if name not in _CONFIGS:
        raise KeyError(f"unknown config '{name}'; available: {sorted(_CONFIGS)}")
    text = (resources.files("expdpy.data") / _CONFIGS[name]).read_text()
    return json.loads(text)


def available_configs() -> list[str]:
    """Return the names of the bundled ``ExPdPy`` configurations."""
    return sorted(_CONFIGS)
