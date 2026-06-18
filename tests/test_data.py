"""Tests for the bundled dataset loaders."""

from __future__ import annotations

import pandas as pd
import pytest

from expdpy import data


def test_load_russell():
    df = data.load_russell_3000()
    assert df.shape[0] > 8000
    assert {"coid", "period", "sector", "sales", "nioa"}.issubset(df.columns)
    assert isinstance(df["period"].dtype, pd.CategoricalDtype)
    assert df["period"].cat.ordered


def test_load_worldbank():
    df = data.load_worldbank()
    assert df.shape[0] > 9000
    assert "year" in df.columns


def test_load_gapminder():
    df = data.load_gapminder()
    assert list(df.columns) == [
        "country",
        "continent",
        "year",
        "lifeExp",
        "pop",
        "gdpPercap",
    ]


def test_load_kuznets():
    df = data.load_kuznets()
    assert df.shape == (880, 21)
    assert df["country"].nunique() == 80
    assert sorted(df["year"].unique()) == list(range(2015, 2026))
    assert list(df.columns) == [
        "country",
        "iso",
        "year",
        "continent",
        "gini_regional",
        "gdp_pc",
        "population",
        "resource_rents",
        "arable_land",
        "trade_share",
        "fdi_share",
        "area",
        "gasoline_price",
        "aid",
        "school_enrollment",
        "gini_lights",
        "polity2",
        "federal",
        "log_gdp_pc",
        "log_gdp_pc_sq",
        "log_gdp_pc_cu",
    ]


def test_load_kuznets_data_def():
    dd = data.load_kuznets_data_def()
    assert list(dd["var_name"]) == list(data.load_kuznets().columns)
    assert set(dd["type"]).issubset({"cs_id", "ts_id", "factor", "logical", "numeric"})
    assert dd["can_be_na"].dtype == bool
    # Panel identifiers are flagged and never missing.
    ids = dd.loc[dd["type"].isin(["cs_id", "ts_id"]), "var_name"]
    assert set(ids) == {"country", "iso", "year"}
    assert not data.load_kuznets()[list(ids)].isna().any().any()


def test_data_defs_have_can_be_na_bool():
    dd = data.load_worldbank_var_def()
    assert dd["can_be_na"].dtype == bool
    assert {"var_name", "var_def", "type"}.issubset(dd.columns)


def test_data_def_types_valid():
    dd = data.load_russell_3000_data_def()
    assert set(dd["type"]).issubset({"cs_id", "ts_id", "factor", "logical", "numeric"})


def test_get_config():
    cfg = data.get_config("worldbank")
    assert isinstance(cfg, dict)
    assert "reg_x" in cfg
    assert data.available_configs() == ["kuznets", "russell_3000", "worldbank"]


def test_get_config_kuznets():
    cfg = data.get_config("kuznets")
    assert isinstance(cfg, dict)
    assert cfg["sample"] == "kuznets"
    assert cfg["reg_x"] == ["log_gdp_pc", "log_gdp_pc_sq", "log_gdp_pc_cu"]
    # The preset opens on the panel's two-way (country + year) fixed effects, clustered
    # by country (cluster choice "2" == FE 1).
    assert cfg["reg_fe1"] == "country"
    assert cfg["reg_fe2"] == "year"
    assert cfg["cluster"] == "2"
    assert cfg["scatter_x"] == "log_gdp_pc"
    assert cfg["scatter_y"] == "gini_regional"


def test_get_config_unknown():
    with pytest.raises(KeyError):
        data.get_config("nope")
