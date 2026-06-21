"""Tests for the variable-label helper and label-aware figures/tables."""

from __future__ import annotations

import pandas as pd

from expdpy import (
    explore_descriptive_table,
    explore_scatter_plot,
    explore_trend_plot,
    resolve_label,
    resolve_panel,
    set_labels,
)
from expdpy._labels import resolve_labels
from expdpy.data import load_kuznets, load_kuznets_data_def


def _toy_def() -> pd.DataFrame:
    """A small df_def exercising the label -> var_def -> var_name fallback chain."""
    return pd.DataFrame(
        {
            "var_name": ["firm", "year", "x1", "x2", "x3"],
            "var_def": ["Firm id", "Year", "First X", "", "Third X"],
            "label": ["Firm", "Year", "Variable one", "", "   "],
            "type": ["entity", "time", "numeric", "numeric", "numeric"],
        }
    )


# ------------------------------------------------------------------ set_labels / resolve ---
def test_dict_mapping_and_precedence(sample_df):
    df = set_labels(sample_df.copy(), {"x1": "Alpha"})
    assert resolve_label(df, "x1") == "Alpha"  # from attrs
    assert resolve_label(df, "x2") == "x2"  # no label -> bare name
    assert resolve_label(df, "x1", label="Override") == "Override"  # explicit wins


def test_unknown_name_never_raises(sample_df):
    df = set_labels(sample_df.copy(), {"x1": "Alpha"})
    # regression terms are not columns; resolve must fall back, not raise
    assert resolve_label(df, "log_gdp_pc_sq") == "log_gdp_pc_sq"


def test_resolve_labels_vectorized_with_override(sample_df):
    df = set_labels(sample_df.copy(), {"x1": "Alpha", "x2": "Beta"})
    out = resolve_labels(df, ["x1", "x2", "x3"], labels={"x1": "Over"})
    assert out == ["Over", "Beta", "x3"]


def test_df_def_label_then_var_def_then_name(sample_df):
    df = set_labels(sample_df.copy(), _toy_def())
    assert resolve_label(df, "x1") == "Variable one"  # label column
    assert resolve_label(df, "x2") == "x2"  # blank label + blank var_def -> name
    assert resolve_label(df, "x3") == "Third X"  # whitespace label -> var_def


def test_df_def_without_label_column_uses_var_def(sample_df):
    dd = pd.DataFrame({"var_name": ["x1"], "var_def": ["First X"], "type": ["numeric"]})
    df = set_labels(sample_df.copy(), dd)
    assert resolve_label(df, "x1") == "First X"


def test_set_panel_flag_declares_panel(sample_df):
    df = set_labels(sample_df.copy(), _toy_def(), set_panel=True)
    assert resolve_panel(df) == ("firm", "year")


def test_set_labels_merges_partial_updates(sample_df):
    df = set_labels(sample_df.copy(), {"x1": "Alpha"})
    set_labels(df, {"x2": "Beta"})
    assert resolve_label(df, "x1") == "Alpha"  # first mapping survives
    assert resolve_label(df, "x2") == "Beta"


# ------------------------------------------------------------------- the trend y-axis rule ---
def test_trend_yaxis_single_multiple_and_fallback():
    df = set_labels(load_kuznets(), load_kuznets_data_def(), set_panel=True)
    one = explore_trend_plot(df, var=["gini_regional"]).fig
    assert one.layout.yaxis.title.text == "Regional inequality (Gini)"
    many = explore_trend_plot(df, var=["gini_regional", "trade_share"]).fig
    assert many.layout.yaxis.title.text == "Value"
    # No labels declared -> the y-axis falls back to the bare variable name.
    raw = explore_trend_plot(load_kuznets(), var=["gini_regional"], time="year").fig
    assert raw.layout.yaxis.title.text == "gini_regional"


def test_figure_axes_use_labels():
    df = set_labels(load_kuznets(), load_kuznets_data_def(), set_panel=True)
    fig = explore_scatter_plot(df, x="log_gdp_pc", y="gini_regional").fig
    assert fig.layout.xaxis.title.text == "Log GDP per capita"
    assert fig.layout.yaxis.title.text == "Regional inequality (Gini)"


# ----------------------------------------------------------------------- tables: display only ---
def test_table_relabels_display_but_keeps_raw_df():
    df = set_labels(load_kuznets(), load_kuznets_data_def(), set_panel=True)
    res = explore_descriptive_table(df)
    # the returned frame keeps the raw variable names (data contract / goldens)
    assert "gini_regional" in res.df.index
    # the rendered Great Table shows the human-readable label
    assert "Regional inequality (Gini)" in res.gt.as_raw_html()
