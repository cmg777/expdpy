"""Tests for prepare_event_study and prepare_panel_view (event study / staggered DiD)."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import pytest

import expdpy as ex
from expdpy import EventStudyResult, PanelViewResult
from expdpy.data import load_staggered_did

_EVENT_COLS = {"event_time", "estimate", "se", "ci_lower", "ci_upper", "cohort"}


@pytest.fixture(scope="module")
def did_df():
    return load_staggered_did()


def test_staggered_did_dataset_loads(did_df):
    assert {"unit", "year", "cohort", "treated", "outcome"} <= set(did_df.columns)
    assert 0 in set(did_df["cohort"])  # never-treated group present


@pytest.mark.parametrize("estimator", ["did2s", "twfe", "saturated", "lpdid"])
def test_event_study_estimators(did_df, estimator):
    kw = {"pre_window": 5, "post_window": 5} if estimator == "lpdid" else {}
    res = ex.prepare_event_study(
        did_df,
        outcome="outcome",
        unit="unit",
        time="year",
        cohort="cohort",
        estimator=estimator,
        **kw,
    )
    assert isinstance(res, EventStudyResult)
    assert set(res.df.columns) == _EVENT_COLS
    assert isinstance(res.fig, go.Figure)
    # the true dynamic effect is positive, so the mean post-treatment estimate > 0
    post = res.df[res.df["event_time"] >= 0]["estimate"].mean()
    assert post > 0
    # no non-finite event times leak through (the -inf never-treated row is dropped)
    assert np.isfinite(res.df["event_time"]).all()


def test_event_study_has_reference_period(did_df):
    res = ex.prepare_event_study(
        did_df, outcome="outcome", unit="unit", time="year", cohort="cohort"
    )
    ref = res.df[res.df["event_time"] == -1.0]
    assert len(ref) == 1
    assert float(ref["estimate"].iloc[0]) == 0.0


def test_saturated_has_per_cohort_curves(did_df):
    res = ex.prepare_event_study(
        did_df,
        outcome="outcome",
        unit="unit",
        time="year",
        cohort="cohort",
        estimator="saturated",
    )
    assert res.df["cohort"].nunique() >= 2
    assert len(res.fig.data) == res.df["cohort"].nunique()


def test_event_study_zero_and_reference_lines(did_df):
    res = ex.prepare_event_study(
        did_df, outcome="outcome", unit="unit", time="year", cohort="cohort"
    )
    # a dotted zero line and a dashed t=-1 reference line
    assert len(res.fig.layout.shapes) == 2


def test_event_study_interpret_and_explain(did_df):
    res = ex.prepare_event_study(
        did_df, outcome="outcome", unit="unit", time="year", cohort="cohort"
    )
    text = res.interpret()
    assert "event study" in text.lower()
    assert "parallel" in text.lower() or "pre-treatment" in text.lower()
    assert "causes" not in text.lower()
    assert res.explain().topic == "event_study"


def test_event_study_missing_column_raises(did_df):
    with pytest.raises(KeyError, match="nope"):
        ex.prepare_event_study(
            did_df, outcome="nope", unit="unit", time="year", cohort="cohort"
        )


# --- panel view ----------------------------------------------------------------------


def test_panel_view_quilt_from_cohort(did_df):
    res = ex.prepare_panel_view(did_df, unit="unit", time="year", cohort="cohort")
    assert isinstance(res, PanelViewResult)
    assert res.df.shape == (did_df["unit"].nunique(), did_df["year"].nunique())
    assert len(res.fig.data) == 1  # a single heatmap


def test_panel_view_quilt_from_binary_treat(did_df):
    res = ex.prepare_panel_view(did_df, unit="unit", time="year", treat="treated")
    assert res.df.shape[0] == did_df["unit"].nunique()


def test_panel_view_outcome_lines(did_df):
    res = ex.prepare_panel_view(
        did_df, unit="unit", time="year", cohort="cohort", outcome="outcome"
    )
    assert list(res.df.columns) == ["unit", "year", "outcome"]
    assert len(res.fig.data) == did_df["unit"].nunique()


def test_panel_view_requires_treat_or_cohort(did_df):
    with pytest.raises(ValueError, match=r"treat.*cohort"):
        ex.prepare_panel_view(did_df, unit="unit", time="year")


def test_panel_view_non_binary_treat_raises(did_df):
    with pytest.raises(ValueError, match="binary"):
        ex.prepare_panel_view(did_df, unit="unit", time="year", treat="outcome")


def test_panel_view_max_units(did_df):
    res = ex.prepare_panel_view(
        did_df, unit="unit", time="year", cohort="cohort", max_units=20
    )
    assert res.df.shape[0] == 20
