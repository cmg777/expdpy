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

import expdpy as ex
from expdpy.streamlit_app import AnalyzeApp, ExploreApp, LearnApp
from expdpy.streamlit_app import _handoff as handoff
from expdpy.streamlit_app._config_io import dump_config, load_config
from expdpy.streamlit_app._context import DATASETS
from expdpy.streamlit_app._export_nb import build_export_zip
from expdpy.streamlit_app._pages import selected_specs
from expdpy.streamlit_app._sidebar import Active, _apply_labels_panel
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


# Sidebar controls (some use a format_func, so their AppTest .options are display labels,
# not raw values) — skipped so the driver only touches a page's analysis selectors.
_CONTROL_KEYS = {
    "sample",
    "subset_factor",
    "subset_value",
    "outlier_treatment",
    "cluster",
}

_EXPLORE_PAGES = [
    "overview",
    "describe",
    "within_between",
    "trends",
    "by_group",
    "correlations",
    "dynamics",
]
_ANALYZE_PAGES = [
    "regression",
    "postestimation",
    "panel_models",
    "kuznets_waves",
    "convergence",
    "event_study",
]


def _drive(at: AppTest) -> AppTest:
    """Set every page selector to a valid value and rerun — exercises the page's functions.

    ``*_y`` selectboxes and multiselects take a *different* option from the ``*_x`` / dv
    choices so paired selectors (scatter x/y, regression dv/idvs) don't collide.
    """
    for sb in at.selectbox:
        key = sb.key or ""
        opts = list(sb.options or [])
        if not opts or key in _CONTROL_KEYS:
            continue
        pick = opts[-1] if (key.endswith("_y") and len(opts) > 1) else opts[0]
        sb.set_value(pick)
    for ms in at.multiselect:
        if ms.key in _CONTROL_KEYS:
            continue
        opts = list(ms.options or [])
        if opts:
            ms.set_value([opts[-1]])
    return at.run()


def _sweep(page: str, dataset: str) -> None:
    at = AppTest.from_string(_PAGE_SCRIPT, default_timeout=120)
    at.session_state["_test_page"] = page
    at.session_state["sample"] = dataset
    at.run()
    assert not at.exception, f"{page} x {dataset} (render): {at.exception}"
    _drive(at)
    assert not at.exception, f"{page} x {dataset} (driven): {at.exception}"


@pytest.mark.parametrize("dataset", list(DATASETS))
def test_explore_pages_drive_clean_on_every_dataset(dataset):
    """Every Explore page renders *and* runs its functions on each bundled dataset."""
    for page in _EXPLORE_PAGES:
        _sweep(page, dataset)


@pytest.mark.parametrize("dataset", list(DATASETS))
def test_analyze_pages_drive_clean_on_every_dataset(dataset):
    """Every Analyze page renders *and* runs its functions on each bundled dataset."""
    for page in _ANALYZE_PAGES:
        _sweep(page, dataset)


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
    from expdpy.data import load_kuznets, load_kuznets_data_def

    cfg = {"hist_var": "gdp_pc", "reg_y": "gini_regional", "reg_x": ["log_gdp_pc"]}
    comps = ["descriptive_table", "histogram", "regression"]
    payload = build_export_zip(
        cfg, comps, load_kuznets().head(50), "year", load_kuznets_data_def()
    )
    import io
    import json
    import zipfile

    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        names = zf.namelist()
        # The data dictionary is shipped alongside the data so labels/panel round-trip.
        assert "expdpy_data_def.csv" in names
        ddef_csv = zf.read("expdpy_data_def.csv").decode()
        assert ddef_csv.splitlines()[0] == "var_name,var_def,label,type,can_be_na"
        nb = json.loads(zf.read("ExPdPy_analysis.ipynb"))
        cells = ["".join(c["source"]) for c in nb["cells"]]
        # Colab-ready: a pinned install + restart cell, and a set_labels(set_panel) load.
        assert any(f"expdpy=={ex.__version__}" in c and "os.kill" in c for c in cells)
        assert any("ex.set_labels(df, data_def, set_panel=True)" in c for c in cells)
    assert "expdpy_sample.parquet" in names
    assert "ExPdPy_analysis.ipynb" in names
    assert "ExPdPy_analysis.py" in names


def test_export_dict_is_inferred_when_absent():
    """With no df_def, the export still ships an inferred dictionary."""
    from expdpy.data import load_gapminder

    payload = build_export_zip(
        {}, ["descriptive_table"], load_gapminder().head(40), "year"
    )
    import io
    import zipfile

    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        ddef = zf.read("expdpy_data_def.csv").decode()
    assert "country,Country,Country,entity" in ddef.replace(
        '"', ""
    )  # auto-built entity row


# --- two-file workflow: data + dictionary -------------------------------------
@pytest.mark.parametrize("name", list(DATASETS))
def test_autodict_recovers_panel_for_every_dataset(name):
    """A data-only upload (auto-built dict) declares the panel for any bundled dataset."""
    loader, _ = DATASETS[name]
    ddef = ex.build_data_def(loader())  # what the app auto-builds on a data-only upload
    entities, time = handoff.resolve_ids(ddef, None, None)
    assert entities and time  # the dictionary unlocks panel features


def test_autodict_unhides_panel_pages_and_missing_values():
    """Regression: uploaded data with an auto-dict shows panel pages + missing-values.

    Previously uploads resolved to ``entities=[], time=None`` so panel pages and the
    missing-values view were silently hidden — the reported bug.
    """
    from expdpy.data import load_gapminder

    ddef = ex.build_data_def(load_gapminder())
    entities, time = handoff.resolve_ids(ddef, None, None)
    pages = [spec[0] for spec in selected_specs(_active(time, entities=entities))]
    assert "Within & between" in pages and "Dynamics" in pages
    assert "missing_values" in handoff.active_components(None, time)


def test_apply_labels_panel_attaches_labels_and_panel():
    from expdpy.data import load_kuznets, load_kuznets_data_def

    df = _apply_labels_panel(load_kuznets(), load_kuznets_data_def())
    assert ex.resolve_panel(df) == ("country", "year")
    assert (
        ex.resolve_label(df, "gini_regional") != "gini_regional"
    )  # a label was attached


def test_apply_labels_panel_tolerates_missing_dict():
    from expdpy.data import load_kuznets

    df = load_kuznets()
    assert _apply_labels_panel(df, None) is df  # no dictionary → unchanged, no crash


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
