"""Tests for the df_def-driven metadata: analytical roles and entity names.

Covers the storage helpers (``set_roles`` / ``set_panel(entity_name=)``), their resolution from a
data dictionary (``set_labels(df_def, set_panel=True)``), the entity-name auto-detection in
``build_data_def``, the shared ``Name (id)`` display helper, the role-based argument defaults on
the entry functions, and the two newly-enforced df_def fields (``type`` authority and
``can_be_na`` completeness).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import expdpy as ex
from expdpy._common import entity_display_map, lead_columns
from expdpy._panel import resolve_entity_name, stored_entity_name
from expdpy._roles import resolve_roles, set_roles, stored_roles
from expdpy._validation import drop_required, required_columns


def _toy() -> pd.DataFrame:
    """A 2-unit panel with an id, a readable name, a code and two numeric variables."""
    return pd.DataFrame(
        {
            "id": [1, 1, 2, 2],
            "name": ["Foo", "Foo", "Bar", "Bar"],
            "code": ["F", "F", "B", "B"],
            "year": [2000, 2001, 2000, 2001],
            "y": [1.0, 2.0, 3.0, 4.0],
            "x": [0.5, 1.0, 1.5, 2.0],
        }
    )


# --------------------------------------------------------------------------- roles storage ---
def test_set_roles_roundtrip():
    df = set_roles(_toy(), outcome="y", covariates=["x"])
    assert stored_roles(df) == ("y", ["x"])
    assert resolve_roles(df) == ("y", ["x"])


def test_set_roles_single_covariate_string():
    df = set_roles(_toy(), outcome="y", covariates="x")
    assert stored_roles(df) == ("y", ["x"])


def test_set_roles_partial_update():
    df = set_roles(_toy(), outcome="y", covariates=["x"])
    set_roles(df, covariates=["x", "year"])  # leave outcome untouched
    assert stored_roles(df) == ("y", ["x", "year"])


def test_set_roles_validates_columns():
    with pytest.raises(ValueError):
        set_roles(_toy(), outcome="nope")
    with pytest.raises(ValueError):
        set_roles(_toy(), covariates=["x", "nope"])


def test_resolve_roles_explicit_wins():
    df = set_roles(_toy(), outcome="y", covariates=["x"])
    assert resolve_roles(df, outcome="x") == ("x", ["x"])
    assert resolve_roles(df, covariates=["year"]) == ("y", ["year"])


def test_stored_roles_empty_by_default():
    assert stored_roles(_toy()) == (None, [])


# ----------------------------------------------------------------------- entity name storage ---
def test_set_panel_entity_name_roundtrip():
    df = ex.set_panel(_toy(), entity="id", time="year", entity_name="name")
    assert stored_entity_name(df) == "name"
    assert resolve_entity_name(df) == "name"


def test_resolve_entity_name_absent_returns_none():
    df = _toy()
    assert resolve_entity_name(df) is None  # nothing declared
    df = ex.set_panel(df, entity="id", time="year", entity_name="name")
    assert resolve_entity_name(df.drop(columns=["name"])) is None  # column gone


# ------------------------------------------------------------------------- entity display map ---
def test_entity_display_map_name_id():
    df = _toy()
    disp = entity_display_map(df, "id", "name")
    assert disp == {"1": "Foo (1)", "2": "Bar (2)"}


def test_entity_display_map_identity_when_no_name():
    df = _toy()
    assert entity_display_map(df, "id", None) == {"1": "1", "2": "2"}
    assert entity_display_map(df, "id", "id") == {"1": "1", "2": "2"}  # no "X (X)"
    assert entity_display_map(df, "id", "missing") == {"1": "1", "2": "2"}


def test_entity_display_map_blank_name_falls_back():
    df = _toy()
    df.loc[df["id"] == 2, "name"] = np.nan
    disp = entity_display_map(df, "id", "name")
    assert disp == {"1": "Foo (1)", "2": "2"}


# ----------------------------------------------------------------------------- df_def loading ---
def test_set_labels_loads_roles_and_entity_name():
    df0 = _toy()
    ddef = ex.build_data_def(df0)
    ddef.loc[ddef["var_name"] == "y", "role"] = "outcome"
    ddef.loc[ddef["var_name"] == "x", "role"] = "covariate"
    df = ex.set_labels(df0, ddef, set_panel=True)
    assert stored_roles(df) == ("y", ["x"])
    # ``name`` is auto-detected as the entity-name column.
    assert stored_entity_name(df) == "name"


def test_set_labels_roleless_df_def_tolerated():
    df0 = _toy()
    ddef = ex.build_data_def(df0).drop(columns=["role"])
    df = ex.set_labels(df0, ddef, set_panel=True)
    assert stored_roles(df) == (None, [])
    assert stored_entity_name(df) is None


# --------------------------------------------------------------- build_data_def auto-detection ---
def test_build_data_def_emits_role_and_detects_name():
    ddef = ex.build_data_def(_toy())
    assert "role" in ddef.columns
    # ``name`` wins over ``code`` (name-like, longer) and over ``id`` (the entity itself).
    names = list(ddef.loc[ddef["role"] == "entity_name", "var_name"])
    assert names == ["name"]
    assert set(ddef.loc[ddef["role"] != "entity_name", "role"]) == {""}


def test_build_data_def_no_name_when_id_already_readable():
    # kuznets keys on ``country`` (a readable name) paired with ``iso`` — no backwards label.
    from expdpy.data import load_kuznets

    ddef = ex.build_data_def(load_kuznets())
    assert list(ddef.loc[ddef["role"] == "entity_name", "var_name"]) == []


def test_build_data_def_detects_name_on_bolivia():
    from expdpy.data import load_bolivia112_gdppc

    ddef = ex.build_data_def(load_bolivia112_gdppc())
    assert list(ddef.loc[ddef["role"] == "entity_name", "var_name"]) == ["prov"]


@pytest.mark.parametrize(
    ("loader", "outcome", "covariates", "entity_name"),
    [
        (
            "kuznets",
            "gini_regional",
            ["log_gdp_pc", "log_gdp_pc_sq", "log_gdp_pc_cu"],
            None,
        ),
        ("gapminder", "lifeExp", ["gdpPercap"], None),
        ("bolivia112_gdppc", "log_gdppc", [], "prov"),
        ("colonial_origins", "log_gdp_pc_1995", ["expropriation_risk"], None),
        ("productivity", "log_gdppc", [], None),
        ("staggered_did", "outcome", [], None),
        ("firms", "log_revenue", ["employees"], None),
        ("regional_conflict", "conflict", ["log_lights"], None),
    ],
)
def test_bundled_data_defs_carry_roles(loader, outcome, covariates, entity_name):
    """Each bundled dictionary ships the curated roles (regenerate via add_roles_to_data_defs)."""
    import expdpy.data as data

    df = getattr(data, f"load_{loader}")()
    ddef = getattr(data, f"load_{loader}_data_def")()
    assert "role" in ddef.columns
    df = ex.set_labels(df, ddef, set_panel=True)
    assert stored_roles(df) == (outcome, covariates)
    assert stored_entity_name(df) == entity_name


# -------------------------------------------------------------------------- role-based defaults ---
def test_scatter_defaults_to_roles():
    df = set_roles(_toy(), outcome="y", covariates=["x"])
    res = ex.explore_scatter_plot(df)  # x/y omitted
    assert res.fig.layout.xaxis.title.text == "x"
    assert res.fig.layout.yaxis.title.text == "y"


def test_scatter_without_roles_raises():
    with pytest.raises(ValueError, match="set_roles"):
        ex.explore_scatter_plot(_toy())


def test_regression_defaults_to_roles():
    df = set_roles(_toy(), outcome="y", covariates=["x"])
    res = ex.analyze_regression_table(df, format="df")  # dvs/idvs omitted
    assert "x" in set(res.df["term"])


def test_histogram_defaults_to_outcome():
    df = set_roles(_toy(), outcome="y")
    res = ex.explore_histogram(df)
    assert res.fig.layout.xaxis.title.text == "y"


def test_trend_defaults_to_outcome_only_when_set():
    df = ex.set_panel(_toy(), entity="id", time="year")
    res_all = ex.explore_trend_plot(df)  # no roles -> all numeric
    set_roles(df, outcome="y")
    res_one = ex.explore_trend_plot(df)  # one trace for the outcome
    assert len({t.name for t in res_one.fig.data}) < len(
        {t.name for t in res_all.fig.data}
    )


def test_lead_columns_orders_keys_first():
    assert lead_columns(["a", "x", "y", "b"], ["y", "x"]) == ["y", "x", "a", "b"]
    assert lead_columns(["a", "b"], [None, "missing"]) == ["a", "b"]  # no-op


# ----------------------------------------------------------------- maximize the inert df_def fields ---
def test_required_columns_and_drop_required():
    df0 = _toy()
    # Mark only ``x`` required (entity/time default to required too, but keep the toy explicit).
    ddef = ex.build_data_def(df0)
    ddef["can_be_na"] = True
    ddef.loc[ddef["var_name"] == "x", "can_be_na"] = False
    assert required_columns(ddef) == ["x"]
    df_missing = df0.copy()
    df_missing.loc[0, "x"] = np.nan
    kept = drop_required(df_missing, ddef)
    assert len(kept) == len(df0) - 1
    # No required columns -> a no-op.
    ddef["can_be_na"] = True
    assert len(drop_required(df_missing, ddef)) == len(df_missing)


def test_spaghetti_labels_units_with_name_id():
    df = ex.set_panel(_toy(), entity="id", time="year", entity_name="name")
    res = ex.explore_spaghetti_plot(df, "y", highlight=["1"])
    shown = [t.name for t in res.fig.data if t.showlegend]
    assert "Foo (1)" in shown


def test_create_var_categories_honours_declared_type():
    from expdpy.streamlit_app._varcat import create_var_categories

    df = _toy()
    # ``y`` is numeric by dtype; declaring it a factor must route it to grouping, not numeric.
    vc = create_var_categories(
        df, types={"y": "factor", "id": "entity", "year": "time"}
    )
    assert "y" in vc.factor
    assert "y" not in vc.numeric
    assert "id" not in vc.numeric and "year" not in vc.numeric  # ids skipped
