"""Tests for the app core: pipeline, varcats, UDV safety, config IO, export, components.

These exercise the framework-agnostic modules under :mod:`expdpy.streamlit_app` that power the
interactive Streamlit app (sample pipeline, variable categorization, the safe ``var_def``
evaluator, config save/load, notebook export, and the Plotly/Great-Tables component helpers).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from expdpy.streamlit_app import _components as comp
from expdpy.streamlit_app._config_io import dump_config, load_config
from expdpy.streamlit_app._export_nb import (
    build_export_zip,
    build_notebook,
    build_script,
)
from expdpy.streamlit_app._sample import build_analysis_sample
from expdpy.streamlit_app._state import DEFAULT_CONFIG, parse_config
from expdpy.streamlit_app._udv import UDVError, evaluate_var_def
from expdpy.streamlit_app._upload import read_uploaded
from expdpy.streamlit_app._varcat import create_var_categories


# --- variable categorization --------------------------------------------------
def test_varcats(kuznets):
    vc = create_var_categories(kuznets, ["country"], "year", factor_cutoff=10)
    assert "gdp_pc" in vc.numeric
    assert "continent" in vc.factor
    assert "year" in vc.times
    assert "country" in vc.entities
    # Panel identifiers are usable as fixed effects, but not as grouping factors.
    assert {"country", "year"}.issubset(vc.fe_choices)
    assert "country" not in vc.grouping and "year" not in vc.grouping
    assert "continent" in vc.fe_choices  # grouping factors remain available too


# --- config state -------------------------------------------------------------
def test_parse_config_fills_defaults():
    cfg = parse_config({"reg_y": "x"})
    assert cfg["reg_y"] == "x"
    assert set(DEFAULT_CONFIG).issubset(cfg)


# --- sample pipeline ----------------------------------------------------------
def test_pipeline_winsorize(kuznets):
    out = build_analysis_sample(
        kuznets,
        ["country"],
        "year",
        {
            "outlier_treatment": "3",
            "subset_factor": "Full Sample",
            "subset_value": "All",
        },
    )
    assert out["gdp_pc"].max() <= kuznets["gdp_pc"].quantile(0.95) + 1e-6


def test_pipeline_subset(kuznets):
    level = kuznets["continent"].dropna().iloc[0]
    out = build_analysis_sample(
        kuznets,
        ["country"],
        "year",
        {"subset_factor": "continent", "subset_value": level, "outlier_treatment": "1"},
    )
    assert len(out) < len(kuznets)
    assert set(out["continent"].astype(str)) == {str(level)}


# --- safe UDV evaluator -------------------------------------------------------
def test_udv_arithmetic_and_log():
    df = pd.DataFrame({"a": [1.0, 2, 3], "b": [10.0, 20, 30]})
    assert list(evaluate_var_def("a + b", df)) == [11, 22, 33]
    assert evaluate_var_def("log(b)", df).iloc[0] == pytest.approx(np.log(10))


def test_udv_grouped_lag():
    df = pd.DataFrame(
        {"a": [1.0, 2, 3, 4], "cs": [1, 1, 2, 2], "yr": [2010, 2011, 2010, 2011]}
    )
    out = evaluate_var_def("lag(a, 1)", df, entities=["cs"], time="yr")
    assert np.isnan(out.iloc[0]) and out.iloc[1] == 1.0


@pytest.mark.parametrize(
    "expr", ["__import__('os')", "a.values", "a[0]", "lambda: 1", "exec('x')"]
)
def test_udv_rejects_dangerous(expr):
    df = pd.DataFrame({"a": [1.0, 2, 3]})
    with pytest.raises(UDVError):
        evaluate_var_def(expr, df)


# --- config IO ----------------------------------------------------------------
def test_config_roundtrip_plain():
    cfg = parse_config({"reg_x": ["a", "b"]})
    assert load_config(dump_config(cfg))["reg_x"] == ["a", "b"]


def test_config_roundtrip_encrypted():
    # Optional config encryption (only when the `cryptography` package is installed).
    pytest.importorskip("cryptography")
    from cryptography.fernet import InvalidToken

    cfg = parse_config({"sample": "x"})
    enc = dump_config(cfg, "secret")
    assert load_config(enc, "secret")["sample"] == "x"
    with pytest.raises(InvalidToken):
        load_config(enc, "wrong-key")


# --- notebook export ----------------------------------------------------------
def test_export_script_and_zip(kuznets):
    comps = ["descriptive_table", "histogram", "scatter_plot", "regression"]
    cfg = {
        "hist_var": "gdp_pc",
        "scatter_x": "log_gdp_pc",
        "scatter_y": "gini_regional",
        "reg_y": "gini_regional",
        "reg_x": ["log_gdp_pc"],
    }
    script = build_script(cfg, comps, time="year")
    assert "ex.explore_descriptive_table" in script
    assert "ex.analyze_regression_table" in script
    assert len(build_notebook(cfg, comps)) > 100
    assert len(build_export_zip(cfg, comps, kuznets.head(50), "year")) > 200


# --- upload -------------------------------------------------------------------
def test_read_uploaded(tmp_path, kuznets):
    p = tmp_path / "d.csv"
    kuznets.head(20).to_csv(p, index=False)
    df = read_uploaded(str(p), "d.csv")
    assert len(df) == 20
    with pytest.raises(ValueError):
        read_uploaded(str(p), "d.txt")


# --- component compute helpers ------------------------------------------------
def test_component_helpers(kuznets):
    sample = build_analysis_sample(
        kuznets, ["country"], "year", {"outlier_treatment": "3"}
    )
    assert comp.descriptive(sample) is not None
    assert comp.histogram(sample, "gdp_pc", 20) is not None
    assert (
        comp.scatter(sample, "log_gdp_pc", "gini_regional", "continent", None, True)
        is not None
    )
    assert comp.corrplot(sample.select_dtypes("number")) is not None
    assert (
        comp.regression(
            sample, "gini_regional", ["log_gdp_pc"], ["continent"], ["continent"]
        )
        is not None
    )
    # incomplete selections no-op
    assert comp.histogram(sample, "None", 20) is None
    assert comp.scatter(sample, "None", "None", None, None, False) is None


def test_event_study_and_panel_helpers():
    import importlib.util

    from expdpy.data import load_staggered_did

    has_lm = importlib.util.find_spec("linearmodels") is not None
    df = load_staggered_did()
    # Event study needs only pyfixest, which is always present.
    fig = comp.event_study(df, "outcome", "unit", "year", "cohort", "did2s")
    assert fig is not None
    notes = comp.event_study_notes(df, "outcome", "unit", "year", "cohort", "did2s")
    assert notes is not None and len(notes) == 2
    # Panel models need the optional linearmodels extra; without it the card shows a
    # friendly HTML message and the notes helper returns None (graceful degradation).
    html = comp.panel_models(df, "outcome", ["treated"], "unit", "year")
    assert html and "<" in html
    pnotes = comp.panel_models_notes(df, "outcome", ["treated"], "unit", "year")
    if has_lm:
        assert pnotes is not None and len(pnotes) == 2
    else:
        assert pnotes is None
    # incomplete selections no-op
    assert comp.event_study(df, "None", "unit", "year", "cohort", "did2s") is None
    assert comp.panel_models(df, "outcome", [], "unit", "year") is None
