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
    assert data.available_configs() == ["russell_3000", "worldbank"]


def test_get_config_unknown():
    with pytest.raises(KeyError):
        data.get_config("nope")
