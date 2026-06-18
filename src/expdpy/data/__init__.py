"""Bundled example datasets ported from ExPanDaR.

All loaders return :class:`pandas.DataFrame` objects read from parquet files shipped with
the package; :func:`get_config` returns a startup configuration ``dict`` for the ``ExPanD``
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
    "load_russell_3000",
    "load_russell_3000_data_def",
    "load_worldbank",
    "load_worldbank_data_def",
    "load_worldbank_var_def",
]

_CONFIGS = {
    "russell_3000": "ExPanD_config_russell_3000.json",
    "worldbank": "ExPanD_config_worldbank.json",
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


def load_russell_3000() -> pd.DataFrame:
    """Annual financial/stock data for a sample of Russell 3000 firms (2013-2016)."""
    df = _read_parquet("russell_3000")
    if "period" in df.columns:
        cats = sorted(df["period"].dropna().astype(str).unique())
        df["period"] = (
            df["period"].astype(str).astype(pd.CategoricalDtype(cats, ordered=True))
        )
    return df


def load_russell_3000_data_def() -> pd.DataFrame:
    """Variable definitions for :func:`load_russell_3000`."""
    return _normalize_def(_read_parquet("russell_3000_data_def"))


def load_worldbank() -> pd.DataFrame:
    """Load a snapshot of World Bank macroeconomic indicators (1960-2018)."""
    return _read_parquet("worldbank")


def load_worldbank_data_def() -> pd.DataFrame:
    """Variable definitions for :func:`load_worldbank`."""
    return _normalize_def(_read_parquet("worldbank_data_def"))


def load_worldbank_var_def() -> pd.DataFrame:
    """Analysis-sample variable definitions for the World Bank data."""
    return _normalize_def(_read_parquet("worldbank_var_def"))


def load_gapminder() -> pd.DataFrame:
    """Load the gapminder dataset (life expectancy, population, GDP per capita)."""
    return _read_parquet("gapminder")


def load_gapminder_data_def() -> pd.DataFrame:
    """Return derived variable definitions for :func:`load_gapminder`."""
    return _normalize_def(_read_parquet("gapminder_data_def"))


@cache
def get_config(name: str) -> dict:
    """Return a startup configuration for ``ExPanD``.

    Parameters
    ----------
    name
        Either ``"russell_3000"`` or ``"worldbank"``.

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
    """Return the names of the bundled ``ExPanD`` configurations."""
    return sorted(_CONFIGS)
