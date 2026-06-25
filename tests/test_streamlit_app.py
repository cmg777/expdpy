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

from expdpy.streamlit_app import AnalyzeApp, ExploreApp, LearnApp
from expdpy.streamlit_app import _handoff as handoff
from expdpy.streamlit_app._config_io import dump_config, load_config
from expdpy.streamlit_app._export_nb import build_export_zip
from expdpy.streamlit_app._pages import selected_specs
from expdpy.streamlit_app._sidebar import Active
from expdpy.streamlit_app._state import parse_config

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
    """Default every test to standalone mode (bundled-dataset picker, combined nav)."""
    monkeypatch.delenv(handoff.EXPDPY_BUNDLE_ENV, raising=False)
    monkeypatch.delenv(handoff.EXPDPY_MODULE_ENV, raising=False)


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


def test_app_runs_bundle_kuznets(monkeypatch):
    from expdpy.data import load_kuznets

    bundle = handoff.write_bundle(
        {"Kuznets": load_kuznets()},
        df_def=None,
        var_def=None,
        entities=["country"],
        time="year",
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
# Variables are picked from the widgets' actual options so these stay valid regardless
# of which bundled dataset is the picker default.
def test_describe_histogram():
    at = _page("describe")
    assert not at.exception
    var = at.selectbox(key="hist_var")
    var.set_value(var.options[0])
    at.run()
    assert not at.exception


def test_correlations_and_scatter():
    at = _page("correlations")
    assert not at.exception
    x, y = at.selectbox(key="scatter_x"), at.selectbox(key="scatter_y")
    x.set_value(x.options[0])
    y.set_value(y.options[1] if len(y.options) > 1 else y.options[0])
    at.run()
    assert not at.exception


def test_regression_renders_table():
    at = _page("regression")
    assert not at.exception
    y = at.selectbox(key="reg_y")
    y.set_value(y.options[0])
    rx = at.multiselect(key="reg_x")
    x = rx.options[1] if len(rx.options) > 1 else rx.options[0]
    rx.set_value([x])
    at.run()
    assert not at.exception
    assert len(at.dataframe) >= 1  # coefficient table


def test_postestimation_page_renders():
    at = _page("postestimation")  # predictions / fixef / joint test / robust inference
    assert not at.exception


def test_sandboxes_page_renders_all_tabs():
    at = _page("sandboxes")
    assert not at.exception
    assert len(at.tabs) == 9  # first differences, within-vs-LSDV, pooled-vs-FE, OVB,
    # clustering, beta / sigma / clubs convergence, Kuznets waves


def test_explainers_page_lists_topics():
    at = _page("explainers")
    assert not at.exception
    topic = at.selectbox(key="explainer_topic")
    assert "fixed_effects" in topic.options
    assert "correlated_random_effects" in topic.options


# --- time-series component hiding --------------------------------------------
def _active(time: str | None, entities: list[str] | None = None) -> Active:
    return Active(
        source_name="x",
        data_id="x",
        base_df=None,
        df_def=None,
        entities=entities or [],
        time=time,
        sample=None,
        var_cats=None,
        active_components=handoff.active_components(None, time),
    )


def test_trends_page_only_for_panel():
    panel = [spec[0] for spec in selected_specs(_active("year"))]
    cross = [spec[0] for spec in selected_specs(_active(None))]
    assert "Trends" in panel
    assert "Trends" not in cross
    assert "Overview & Data" in panel and "Overview & Data" in cross


def test_panel_pages_need_full_panel():
    full = [spec[0] for spec in selected_specs(_active("year", entities=["country"]))]
    ts_only = [spec[0] for spec in selected_specs(_active("year"))]  # no entity id
    cross = [spec[0] for spec in selected_specs(_active(None))]
    for page in ("Within & between", "Dynamics"):
        assert page in full
        assert page not in ts_only
        assert page not in cross


def test_within_between_page_renders():
    at = _page("within_between")  # default dataset (Kuznets) is a full panel
    assert not at.exception


def test_dynamics_page_renders():
    at = _page("dynamics")  # default dataset (Kuznets) is a full panel
    assert not at.exception


def test_convergence_page_needs_full_panel():
    full = [spec[0] for spec in selected_specs(_active("year", entities=["country"]))]
    ts_only = [spec[0] for spec in selected_specs(_active("year"))]  # no entity id
    cross = [spec[0] for spec in selected_specs(_active(None))]
    assert "Convergence" in full
    assert "Convergence" not in ts_only
    assert "Convergence" not in cross


def test_convergence_page_renders():
    at = _page("convergence")  # default dataset (Kuznets) is a balanced panel
    assert not at.exception


def test_kuznets_waves_page_needs_full_panel():
    full = [spec[0] for spec in selected_specs(_active("year", entities=["country"]))]
    ts_only = [spec[0] for spec in selected_specs(_active("year"))]  # no entity id
    cross = [spec[0] for spec in selected_specs(_active(None))]
    assert "Kuznets waves" in full
    assert "Kuznets waves" not in ts_only
    assert "Kuznets waves" not in cross


def test_kuznets_waves_page_renders():
    at = _page("kuznets_waves")  # default dataset (Kuznets) is a balanced panel
    assert not at.exception


# --- per-module page filtering ------------------------------------------------
def test_each_module_shows_only_its_pages():
    active = _active("year", entities=["country"])  # a true panel: all gates pass
    explore = {spec[0] for spec in selected_specs(active, module="explore")}
    analyze = {spec[0] for spec in selected_specs(active, module="analyze")}
    learn = {spec[0] for spec in selected_specs(active, module="learn")}

    assert "Overview & Data" in explore and "Trends" in explore
    assert "Regression" in analyze and "Panel models" in analyze
    assert "Post-estimation" in analyze
    assert "Convergence" in analyze
    assert "Concept sandboxes" in learn and "Concept explainers" in learn
    # The three modules partition the pages — no page appears in two of them.
    assert explore.isdisjoint(analyze) and analyze.isdisjoint(learn)
    assert explore.isdisjoint(learn)
    # Learn pages are data-free, so they show even on cross-sectional data.
    cross_learn = {spec[0] for spec in selected_specs(_active(None), module="learn")}
    assert cross_learn == learn


# --- config round-trip (framework-agnostic serializer) ------------------------
def test_config_roundtrip_plain():
    cfg = parse_config({"reg_y": "lifeExp", "reg_x": ["gdpPercap"], "hist_var": "pop"})
    payload = dump_config(cfg, None)  # the shared config serializer
    loaded = load_config(payload, None)
    assert loaded["reg_y"] == "lifeExp"
    assert loaded["reg_x"] == ["gdpPercap"]
    assert loaded["hist_var"] == "pop"


# --- reproducible export ------------------------------------------------------
def test_export_zip_contents():
    from expdpy.data import load_kuznets

    cfg = {"hist_var": "gdp_pc", "reg_y": "gini_regional", "reg_x": ["log_gdp_pc"]}
    comps = ["descriptive_table", "histogram", "regression"]
    payload = build_export_zip(cfg, comps, load_kuznets().head(50), "year")
    import io
    import zipfile

    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        names = zf.namelist()
    assert "expdpy_sample.parquet" in names
    assert "ExPdPy_analysis.ipynb" in names
    assert "ExPdPy_analysis.py" in names


# --- launcher (no subprocess) -------------------------------------------------
def test_launcher_command_and_bundle(monkeypatch):
    from expdpy.data import load_kuznets

    cmd = ExploreApp(
        load_kuznets(),
        entity=["country"],
        time="year",
        run=False,
        headless=True,
        port=8765,
    )
    assert "streamlit" in cmd and "run" in cmd
    assert cmd[-3].endswith("_run.py") or any(c.endswith("_run.py") for c in cmd)
    assert "--server.port" in cmd and "8765" in cmd
    # The Explore launcher tags the subprocess with its module via the environment.
    assert os.environ[handoff.EXPDPY_MODULE_ENV] == "explore"

    bundle = os.environ[handoff.EXPDPY_BUNDLE_ENV]
    try:
        loaded = handoff.read_bundle(bundle)
        assert list(loaded.samples) == ["Sample"]
        assert loaded.time == "year"
        assert loaded.entities == ["country"]
    finally:
        handoff.cleanup_bundle(bundle)
        monkeypatch.delenv(handoff.EXPDPY_BUNDLE_ENV, raising=False)


@pytest.mark.parametrize("legacy", ["cs_id", "ts_id"])
def test_launcher_rejects_legacy_id_kwargs(legacy):
    from expdpy.data import load_kuznets

    with pytest.raises(TypeError, match="no longer accepted"):
        ExploreApp(load_kuznets(), **{legacy: ["country"]}, run=False)


@pytest.mark.parametrize(
    ("launch", "module"),
    [(ExploreApp, "explore"), (AnalyzeApp, "analyze"), (LearnApp, "learn")],
)
def test_module_launchers_tag_the_module(monkeypatch, launch, module):
    from expdpy.data import load_kuznets

    monkeypatch.delenv(handoff.EXPDPY_MODULE_ENV, raising=False)
    launch(load_kuznets(), entity=["country"], time="year", run=False)
    assert os.environ[handoff.EXPDPY_MODULE_ENV] == module
    monkeypatch.delenv(handoff.EXPDPY_BUNDLE_ENV, raising=False)
