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
    "load_bolivia112_gdppc",
    "load_bolivia112_gdppc_data_def",
    "load_colonial_origins",
    "load_colonial_origins_data_def",
    "load_firms",
    "load_firms_data_def",
    "load_gapminder",
    "load_gapminder_data_def",
    "load_kuznets",
    "load_kuznets_data_def",
    "load_productivity",
    "load_productivity_data_def",
    "load_regional_conflict",
    "load_regional_conflict_data_def",
    "load_staggered_did",
    "load_staggered_did_data_def",
]

_CONFIGS = {
    "kuznets": "expdpy_config_kuznets.json",
}


def _read_parquet(name: str) -> pd.DataFrame:
    with resources.as_file(resources.files("expdpy.data") / f"{name}.parquet") as path:
        return pd.read_parquet(path)


#: Allowed values for a data dictionary's ``role`` column.
_ROLE_VALUES = {"outcome", "covariate", "entity_name"}


def _normalize_def(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce a data/var-definition frame's ``can_be_na`` to bool and clean any ``role`` column."""
    if "can_be_na" in df.columns or "role" in df.columns:
        df = df.copy()
    if "can_be_na" in df.columns:
        df["can_be_na"] = df["can_be_na"].astype(float).astype(bool)
    if "role" in df.columns:
        cleaned = df["role"].fillna("").astype(str).str.strip()
        df["role"] = cleaned.where(cleaned.isin(_ROLE_VALUES), "")
    return df


def load_bolivia112_gdppc() -> pd.DataFrame:
    """Load the Bolivian subnational GDP-per-capita panel (112 provinces, 1990-2024).

    A real-world balanced annual panel of 112 Bolivian provinces (nested within 9 departments)
    over 35 years with GDP per capita and its natural log. The empirical counterpart to the
    synthetic :func:`load_productivity` panel — built for the convergence workflows
    (:func:`expdpy.analyze_beta_convergence` / :func:`expdpy.analyze_sigma_convergence` /
    :func:`expdpy.analyze_convergence_clubs`) and for general subnational exploration (spaghetti,
    by-department group views, panel structure, trends). Source: Kummu, Kosonen & Masoumzadeh
    Sayyar, *Sci Data* 12, 178 (2025), https://doi.org/10.1038/s41597-025-04487-x.
    """
    return _read_parquet("bolivia112_gdppc")


def load_bolivia112_gdppc_data_def() -> pd.DataFrame:
    """Return variable definitions for :func:`load_bolivia112_gdppc`."""
    return _normalize_def(_read_parquet("bolivia112_gdppc_data_def"))


def load_colonial_origins() -> pd.DataFrame:
    """Load the Colonial Origins cross-section (Acemoglu, Johnson & Robinson 2001).

    A country-level **cross-section** (163 countries, no time dimension) carrying the variables
    of the famous settler-mortality IV example: log GDP per capita in 1995, average protection
    against expropriation risk (the endogenous regressor), and log settler mortality (the
    instrument), plus geography, colonizer and religion covariates and a ``base_sample`` flag
    for the 64-country base sample. The canonical teaching dataset for
    :func:`expdpy.analyze_iv_regression`. Source: Acemoglu, Johnson & Robinson (2001), "The
    Colonial Origins of Comparative Development", *American Economic Review* 91(5).
    """
    return _read_parquet("colonial_origins")


def load_colonial_origins_data_def() -> pd.DataFrame:
    """Return variable definitions for :func:`load_colonial_origins`."""
    return _normalize_def(_read_parquet("colonial_origins_data_def"))


def load_firms() -> pd.DataFrame:
    """Load the synthetic *unbalanced* firms panel (staggered entry/exit, gaps, AR-1 revenue).

    A small firm-year panel built to exercise the panel-structure diagnostics (it is
    deliberately unbalanced, with varying numbers of periods per firm and a few interior
    gaps), the transition matrix (the discrete ``size_class`` moves over time) and the
    within-unit persistence view (``log_revenue`` follows a persistent AR-1 process).
    """
    return _read_parquet("firms")


def load_firms_data_def() -> pd.DataFrame:
    """Return variable definitions for :func:`load_firms`."""
    return _normalize_def(_read_parquet("firms_data_def"))


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


def load_productivity() -> pd.DataFrame:
    """Load the cross-country labor-productivity convergence panel (PWT9.0, 1990-2014).

    A balanced annual panel of 108 countries over 25 years with log GDP per capita and log
    labor productivity (raw, before HP filtering), their levels, and region / income grouping
    factors. Built for :func:`expdpy.analyze_convergence_clubs` — the flagship dataset for the
    Phillips-Sul log(t) club-convergence workflow (mendez2020-convergence-clubs).
    """
    return _read_parquet("productivity")


def load_productivity_data_def() -> pd.DataFrame:
    """Return variable definitions for :func:`load_productivity`."""
    return _normalize_def(_read_parquet("productivity_data_def"))


def load_regional_conflict() -> pd.DataFrame:
    """Load the African regional-conflict panel (a focused teaching subset).

    An unbalanced region-year panel (5,689 African regions across 43 countries, 1994-2010)
    with conflict indicators, night-time lights (a proxy for local economic activity) and
    lagged rainfall and drought instruments — all region-detrended (``*_dt``). The canonical
    teaching dataset for :func:`expdpy.analyze_panel_iv_regression`: it instruments night-lights
    with lagged weather while absorbing region and year fixed effects (a panel ``xtivreg2 fe``).
    Source: the regional-conflict / economic-shocks replication data (night-lights, rainfall and
    UCDP conflict events).
    """
    return _read_parquet("regional_conflict")


def load_regional_conflict_data_def() -> pd.DataFrame:
    """Return variable definitions for :func:`load_regional_conflict`."""
    return _normalize_def(_read_parquet("regional_conflict_data_def"))


def load_staggered_did() -> pd.DataFrame:
    """Load the synthetic staggered difference-in-differences dataset.

    A balanced unit-year panel with several treatment cohorts (and a never-treated control
    group, ``cohort == 0``) and a known *dynamic* treatment effect — for teaching event-study
    and staggered-DiD methods via :func:`expdpy.analyze_event_study`.
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
