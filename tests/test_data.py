"""Tests for the bundled dataset loaders."""

from __future__ import annotations

import pytest

from expdpy import data


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
    assert set(dd["type"]).issubset({"entity", "time", "factor", "logical", "numeric"})
    assert dd["can_be_na"].dtype == bool
    # Panel identifiers are flagged and never missing.
    ids = dd.loc[dd["type"].isin(["entity", "time"]), "var_name"]
    assert set(ids) == {"country", "iso", "year"}
    assert not data.load_kuznets()[list(ids)].isna().any().any()


def test_load_firms_is_unbalanced():
    df = data.load_firms()
    assert {"firm", "year", "sector", "revenue", "size_class"}.issubset(df.columns)
    obs_per_firm = df.groupby("firm")["year"].nunique()
    # the dataset is deliberately unbalanced (varying periods per firm)
    assert obs_per_firm.min() < obs_per_firm.max()
    assert set(df["size_class"]) == {"small", "medium", "large"}


def test_load_firms_data_def():
    dd = data.load_firms_data_def()
    assert list(dd["var_name"]) == list(data.load_firms().columns)
    assert set(dd["type"]).issubset({"entity", "time", "factor", "logical", "numeric"})
    ids = dd.loc[dd["type"].isin(["entity", "time"]), "var_name"]
    assert set(ids) == {"firm", "year"}


def test_data_defs_have_can_be_na_bool():
    dd = data.load_gapminder_data_def()
    assert dd["can_be_na"].dtype == bool
    assert {"var_name", "var_def", "type"}.issubset(dd.columns)


def test_data_def_types_valid():
    dd = data.load_gapminder_data_def()
    assert set(dd["type"]).issubset({"entity", "time", "factor", "logical", "numeric"})


def test_get_config():
    cfg = data.get_config("kuznets")
    assert isinstance(cfg, dict)
    assert "reg_x" in cfg
    assert data.available_configs() == ["kuznets"]


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
