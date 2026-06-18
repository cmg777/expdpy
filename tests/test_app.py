"""Tests for the ExPdPy app: pipeline, varcats, UDV safety, config IO, export, build."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("shiny")

from shiny import App

from expdpy.app import ExPdPy
from expdpy.app import _components as comp
from expdpy.app._config_io import dump_config, load_config
from expdpy.app._export_nb import (
    build_export_zip,
    build_notebook,
    build_script,
)
from expdpy.app._sample import build_analysis_sample
from expdpy.app._state import DEFAULT_CONFIG, parse_config
from expdpy.app._udv import UDVError, evaluate_var_def
from expdpy.app._upload import read_uploaded
from expdpy.app._varcat import create_var_categories

pytestmark = pytest.mark.app


# --- variable categorization --------------------------------------------------
def test_varcats(russell):
    vc = create_var_categories(russell, ["coid", "coname"], "period", factor_cutoff=10)
    assert "sales" in vc.numeric
    assert "sector" in vc.factor
    assert "period" in vc.ts_id
    assert "coid" in vc.cs_id
    # Panel identifiers are usable as fixed effects, but not as grouping factors.
    assert {"coid", "period"}.issubset(vc.fe_choices)
    assert "coid" not in vc.grouping and "period" not in vc.grouping
    assert "sector" in vc.fe_choices  # grouping factors remain available too


# --- config state -------------------------------------------------------------
def test_parse_config_fills_defaults():
    cfg = parse_config({"reg_y": "x"})
    assert cfg["reg_y"] == "x"
    assert set(DEFAULT_CONFIG).issubset(cfg)


# --- sample pipeline ----------------------------------------------------------
def test_pipeline_winsorize(russell):
    out = build_analysis_sample(
        russell,
        ["coid", "coname"],
        "period",
        {
            "outlier_treatment": "3",
            "subset_factor": "Full Sample",
            "subset_value": "All",
        },
    )
    assert out["sales"].max() <= russell["sales"].quantile(0.95) + 1e-6


def test_pipeline_subset(russell):
    level = russell["sector"].dropna().iloc[0]
    out = build_analysis_sample(
        russell,
        ["coid", "coname"],
        "period",
        {"subset_factor": "sector", "subset_value": level, "outlier_treatment": "1"},
    )
    assert len(out) < len(russell)
    assert set(out["sector"].astype(str)) == {str(level)}


# --- safe UDV evaluator -------------------------------------------------------
def test_udv_arithmetic_and_log():
    df = pd.DataFrame({"a": [1.0, 2, 3], "b": [10.0, 20, 30]})
    assert list(evaluate_var_def("a + b", df)) == [11, 22, 33]
    assert evaluate_var_def("log(b)", df).iloc[0] == pytest.approx(np.log(10))


def test_udv_grouped_lag():
    df = pd.DataFrame(
        {"a": [1.0, 2, 3, 4], "cs": [1, 1, 2, 2], "yr": [2010, 2011, 2010, 2011]}
    )
    out = evaluate_var_def("lag(a, 1)", df, cs_id=["cs"], ts_id="yr")
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
    from cryptography.fernet import InvalidToken

    cfg = parse_config({"sample": "x"})
    enc = dump_config(cfg, "secret")
    assert load_config(enc, "secret")["sample"] == "x"
    with pytest.raises(InvalidToken):
        load_config(enc, "wrong-key")


# --- notebook export ----------------------------------------------------------
def test_export_script_and_zip(russell):
    comps = ["descriptive_table", "histogram", "scatter_plot", "regression"]
    cfg = {
        "hist_var": "sales",
        "scatter_x": "sales",
        "scatter_y": "nioa",
        "reg_y": "nioa",
        "reg_x": ["ni_sales"],
    }
    script = build_script(cfg, comps, ts_id="period")
    assert "ex.prepare_descriptive_table" in script
    assert "ex.prepare_regression_table" in script
    assert len(build_notebook(cfg, comps)) > 100
    assert len(build_export_zip(cfg, comps, russell.head(50), "period")) > 200


# --- upload -------------------------------------------------------------------
def test_read_uploaded(tmp_path, russell):
    p = tmp_path / "d.csv"
    russell.head(20).to_csv(p, index=False)
    df = read_uploaded(str(p), "d.csv")
    assert len(df) == 20
    with pytest.raises(ValueError):
        read_uploaded(str(p), "d.txt")


# --- component compute helpers ------------------------------------------------
def test_component_helpers(russell):
    sample = build_analysis_sample(
        russell, ["coid", "coname"], "period", {"outlier_treatment": "3"}
    )
    assert comp.descriptive(sample) is not None
    assert comp.histogram(sample, "sales", 20) is not None
    assert comp.scatter(sample, "ni_sales", "nioa", "sector", None, True) is not None
    assert comp.corrplot(sample.select_dtypes("number")) is not None
    assert (
        comp.regression(sample, "nioa", ["ni_sales"], ["sector"], ["sector"])
        is not None
    )
    # incomplete selections no-op
    assert comp.histogram(sample, "None", 20) is None
    assert comp.scatter(sample, "None", "None", None, None, False) is None


# --- app construction ---------------------------------------------------------
def test_expand_builds_app(russell):
    app = ExPdPy(russell, cs_id=["coid", "coname"], ts_id="period", run=False)
    assert isinstance(app, App)


def test_expand_upload_mode_builds():
    assert isinstance(ExPdPy(run=False), App)


def test_expand_serves_http(russell):
    pytest.importorskip("httpx")
    starlette_test = pytest.importorskip("starlette.testclient")
    app = ExPdPy(russell, cs_id=["coid", "coname"], ts_id="period", run=False)
    client = starlette_test.TestClient(app)
    resp = client.get("/")
    assert resp.status_code == 200
    assert "ExPdPy" in resp.text
