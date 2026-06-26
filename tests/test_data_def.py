"""Tests for :func:`expdpy.build_data_def` — data-dictionary inference."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import expdpy as ex
from expdpy.data import (
    load_bolivia112_gdppc,
    load_firms,
    load_gapminder,
    load_kuznets,
    load_productivity,
    load_staggered_did,
)

_DEF_COLUMNS = ["var_name", "var_def", "label", "type", "can_be_na"]
_TYPES = {"entity", "time", "factor", "logical", "numeric"}


def test_returns_five_column_contract():
    df = pd.DataFrame({"country": ["A", "B"], "year": [2000, 2001], "gdp": [1.0, 2.0]})
    ddef = ex.build_data_def(df)
    assert list(ddef.columns) == _DEF_COLUMNS
    assert ddef["can_be_na"].dtype == bool
    assert len(ddef) == df.shape[1]
    assert list(ddef["var_name"]) == list(df.columns)  # one row per column, in order
    assert set(ddef["type"]).issubset(_TYPES)


def test_type_inference_table():
    n = 12
    df = pd.DataFrame(
        {
            "country": (["A"] * 6) + (["B"] * 6),  # object name hint -> entity
            "year": list(range(2000, 2006)) * 2,  # int year -> time
            "continent": (["x", "y", "z"] * 4),  # >2 object -> factor
            "flag": [True, False] * 6,  # bool -> logical
            "binary": [0, 1] * 6,  # 2-valued -> logical
            "grade": ([1, 2, 3] * 4),  # low-card numeric -> factor
            "gdp": np.linspace(1.0, 50.0, n),  # high-card continuous -> numeric
        }
    )
    ddef = (
        ex.build_data_def(df, factor_cutoff=10).set_index("var_name")["type"].to_dict()
    )
    assert ddef["country"] == "entity"
    assert ddef["year"] == "time"
    assert ddef["continent"] == "factor"
    assert ddef["flag"] == "logical"
    assert ddef["binary"] == "logical"
    assert ddef["grade"] == "factor"
    assert ddef["gdp"] == "numeric"


def test_explicit_entity_time_win_over_detection():
    df = pd.DataFrame(
        {"country": ["A", "B"], "year": [2000, 2001], "wave": [1, 2], "x": [1.0, 2.0]}
    )
    ddef = ex.build_data_def(df, entity="x", time="wave").set_index("var_name")["type"]
    assert ddef["x"] == "entity"
    assert ddef["wave"] == "time"
    # the auto-detected country/year are demoted because the user pinned the ids
    assert ddef["country"] != "entity"
    assert ddef["year"] != "time"


def test_explicit_id_must_exist():
    df = pd.DataFrame({"a": [1, 2]})
    with pytest.raises(ValueError, match="not in df"):
        ex.build_data_def(df, entity="missing")


def test_all_numeric_frame_has_no_false_panel():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(20, 3)), columns=["m1", "m2", "m3"])
    ddef = ex.build_data_def(df)
    assert "entity" not in set(ddef["type"])
    assert "time" not in set(ddef["type"])


def test_year_detection_without_name_hint():
    df = pd.DataFrame(
        {"unit": ["A", "B", "A", "B"], "yr_col": [1998, 1998, 1999, 1999]}
    )
    # "yr_col" tokenizes to {yr, col}; "yr" is a time hint, so it is detected as time.
    ddef = ex.build_data_def(df).set_index("var_name")["type"]
    assert ddef["yr_col"] == "time"
    assert ddef["unit"] == "entity"


def test_humanized_labels():
    df = pd.DataFrame({"gini_regional": [0.1, 0.2], "log_gdp_pc": [1.0, 2.0]})
    ddef = ex.build_data_def(df).set_index("var_name")
    assert ddef.loc["gini_regional", "label"] == "Gini Regional"
    assert ddef.loc["log_gdp_pc", "label"] == "Log Gdp Pc"
    assert (
        ddef.loc["gini_regional", "var_def"] == "Gini Regional"
    )  # var_def defaults to label


@pytest.mark.parametrize(
    ("loader", "entity", "time"),
    [
        (load_kuznets, "country", "year"),
        (load_gapminder, "country", "year"),
        (load_staggered_did, "unit", "year"),
        (load_firms, "firm", "year"),
        (load_productivity, "country", "year"),
        (load_bolivia112_gdppc, "prov_id", "year"),
    ],
)
def test_roundtrip_recovers_panel(loader, entity, time):
    """build_data_def -> set_labels(set_panel=True) declares the right panel."""
    raw = loader()
    ddef = ex.build_data_def(raw)
    df = ex.set_labels(loader(), ddef, set_panel=True)
    assert ex.resolve_panel(df) == (entity, time)
    # the inferred dictionary satisfies the same contract the bundled ones do
    assert set(ddef["type"]).issubset(_TYPES)
    assert ddef["can_be_na"].dtype == bool


def test_empty_frame_is_safe():
    ddef = ex.build_data_def(pd.DataFrame())
    assert list(ddef.columns) == _DEF_COLUMNS
    assert len(ddef) == 0


def test_non_dataframe_raises():
    with pytest.raises(TypeError):
        ex.build_data_def([1, 2, 3])
