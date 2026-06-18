"""Tests for the Streamlit ExPdPy app: launch handoff, pipeline, pages, config IO, export.

The interactive pages are exercised with ``streamlit.testing.v1.AppTest`` (skipped when
streamlit is not installed); the framework-agnostic pieces (bundle round-trip, launcher
command, config round-trip, export zip, TS-component hiding) are tested directly.
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest

from expdpy.app._config_io import dump_config, load_config
from expdpy.app._export_nb import build_export_zip
from expdpy.app._state import parse_config
from expdpy.streamlit_app import ExPdPy
from expdpy.streamlit_app import _handoff as handoff
from expdpy.streamlit_app._pages import selected_specs
from expdpy.streamlit_app._sidebar import Active

pytestmark = pytest.mark.streamlit

_RUN = os.path.join(
    os.path.dirname(__file__), "..", "src", "expdpy", "streamlit_app", "_run.py"
)

# A from_string script that renders the sidebar then a single requested page, so individual
# pages can be tested without driving st.navigation (unsupported by AppTest for st.Page).
_PAGE_SCRIPT = """
import streamlit as st
from expdpy.streamlit_app._context import resolve_context
from expdpy.streamlit_app._sidebar import render_sidebar
from expdpy.streamlit_app import _pages
ctx = resolve_context()
active = render_sidebar(ctx)
getattr(_pages, "page_" + st.session_state.get("_test_page", "overview"))()
"""


@pytest.fixture(autouse=True)
def _no_bundle(monkeypatch):
    """Default every test to standalone mode (bundled-dataset picker)."""
    monkeypatch.delenv(handoff.EXPDPY_BUNDLE_ENV, raising=False)


def _page(name: str, **timeout) -> AppTest:
    at = AppTest.from_string(_PAGE_SCRIPT, default_timeout=timeout.get("t", 90))
    at.session_state["_test_page"] = name
    return at.run()


# --- full-app smoke -----------------------------------------------------------
def test_app_runs_standalone():
    at = AppTest.from_file(_RUN, default_timeout=90).run()
    assert not at.exception
    assert len(at.dataframe) >= 1  # descriptive table + preview
    assert any(s.key == "sample" for s in at.selectbox)  # dataset picker


def test_app_runs_bundle_russell(monkeypatch):
    from expdpy.data import load_russell_3000

    bundle = handoff.write_bundle(
        {"Russell": load_russell_3000()},
        df_def=None,
        var_def=None,
        cs_list=["coid", "coname"],
        ts="period",
        components=None,
        factor_cutoff=10,
        title="t",
        export_nb_option=True,
        save_settings_option=True,
        base_cfg={},
    )
    try:
        monkeypatch.setenv(handoff.EXPDPY_BUNDLE_ENV, bundle)
        at = AppTest.from_file(_RUN, default_timeout=120).run()
        assert not at.exception
    finally:
        handoff.cleanup_bundle(bundle)


# --- individual pages ---------------------------------------------------------
def test_distributions_histogram():
    at = _page("distributions")
    assert not at.exception
    at.selectbox(key="hist_var").set_value("lifeExp")
    at.run()
    assert not at.exception


def test_correlations_and_scatter():
    at = _page("correlations")
    assert not at.exception
    at.selectbox(key="scatter_x").set_value("gdpPercap")
    at.selectbox(key="scatter_y").set_value("lifeExp")
    at.run()
    assert not at.exception


def test_regression_renders_table():
    at = _page("regression")
    assert not at.exception
    at.selectbox(key="reg_y").set_value("lifeExp")
    at.multiselect(key="reg_x").set_value(["gdpPercap"])
    at.run()
    assert not at.exception
    assert len(at.dataframe) >= 1  # coefficient table


# --- time-series component hiding --------------------------------------------
def _active(ts: str | None) -> Active:
    return Active(
        source_name="x",
        data_id="x",
        base_df=None,
        df_def=None,
        cs_list=[],
        ts=ts,
        sample=None,
        var_cats=None,
        active_components=handoff.active_components(None, ts),
    )


def test_trends_page_only_for_panel():
    panel = [spec[0] for spec in selected_specs(_active("year"))]
    cross = [spec[0] for spec in selected_specs(_active(None))]
    assert "Trends" in panel
    assert "Trends" not in cross
    assert "Overview & Data" in panel and "Overview & Data" in cross


# --- config round-trip (interchangeable with the Shiny app) -------------------
def test_config_roundtrip_plain():
    cfg = parse_config({"reg_y": "lifeExp", "reg_x": ["gdpPercap"], "hist_var": "pop"})
    payload = dump_config(cfg, None)  # same serializer the Shiny app uses
    loaded = load_config(payload, None)
    assert loaded["reg_y"] == "lifeExp"
    assert loaded["reg_x"] == ["gdpPercap"]
    assert loaded["hist_var"] == "pop"


# --- reproducible export ------------------------------------------------------
def test_export_zip_contents():
    from expdpy.data import load_russell_3000

    cfg = {"hist_var": "sales", "reg_y": "nioa", "reg_x": ["ni_sales"]}
    comps = ["descriptive_table", "histogram", "regression"]
    payload = build_export_zip(cfg, comps, load_russell_3000().head(50), "period")
    import io
    import zipfile

    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        names = zf.namelist()
    assert "expdpy_sample.parquet" in names
    assert "ExPdPy_analysis.ipynb" in names
    assert "ExPdPy_analysis.py" in names


# --- launcher (no subprocess) -------------------------------------------------
def test_launcher_command_and_bundle(monkeypatch):
    from expdpy.data import load_russell_3000

    cmd = ExPdPy(
        load_russell_3000(),
        cs_id=["coid", "coname"],
        ts_id="period",
        run=False,
        headless=True,
        port=8765,
    )
    assert "streamlit" in cmd and "run" in cmd
    assert cmd[-3].endswith("_run.py") or any(c.endswith("_run.py") for c in cmd)
    assert "--server.port" in cmd and "8765" in cmd

    bundle = os.environ[handoff.EXPDPY_BUNDLE_ENV]
    try:
        loaded = handoff.read_bundle(bundle)
        assert list(loaded.samples) == ["Sample"]
        assert loaded.ts == "period"
        assert loaded.cs_list == ["coid", "coname"]
    finally:
        handoff.cleanup_bundle(bundle)
        monkeypatch.delenv(handoff.EXPDPY_BUNDLE_ENV, raising=False)
