"""Tests for the shared validation helpers: NaN-drop reporting + column checks."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import expdpy as ex
from expdpy._validation import ExpdpyWarning, drop_missing, require_columns


def test_expdpy_warning_is_userwarning():
    # Subclassing UserWarning keeps existing ``pytest.warns(UserWarning)`` callers matching.
    assert issubclass(ExpdpyWarning, UserWarning)


def test_drop_missing_warns_with_count():
    df = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0, np.nan], "b": [1, 2, 3, 4, 5]})
    with pytest.warns(
        ExpdpyWarning,
        match=r"myfunc: dropped 2 of 5 row\(s\) \(40%\) with missing values in \['a'\]",
    ):
        out = drop_missing(df, ["a"], func="myfunc")
    assert len(out) == 3
    assert not out["a"].isna().any()


def test_drop_missing_silent_when_clean():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [1, 2, 3]})
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning would raise here
        out = drop_missing(df, ["a", "b"], func="f")
    assert out.equals(df)


def test_drop_missing_empty_frame_no_warning_no_zerodiv():
    df = pd.DataFrame({"a": []})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = drop_missing(df, ["a"], func="f")
    assert len(out) == 0


def test_drop_missing_stacklevel_points_at_caller():
    df = pd.DataFrame({"a": [1.0, np.nan]})

    def public_like(frame):  # simulates the public function that calls the helper
        return drop_missing(frame, ["a"], func="f")  # default stacklevel=3

    with pytest.warns(ExpdpyWarning) as record:
        public_like(df)
    # stacklevel=3: warn -> drop_missing -> public_like -> this test => the test's frame.
    assert record[0].filename.endswith("test_validation.py")


def test_require_columns_names_missing():
    df = pd.DataFrame({"a": [1], "b": [2]})
    with pytest.raises(
        ValueError, match=r"xtsum: column\(s\) not found in df: \['nope'\]"
    ):
        require_columns(df, ["a", "nope"], where="xtsum")


def test_require_columns_passes_when_present():
    df = pd.DataFrame({"a": [1], "b": [2]})
    require_columns(df, ["a", "b"], where="x")  # no raise


# --- integration: a converted public function warns through the public API ----------


def test_public_function_warns_on_dropped_rows(kuznets):
    # gasoline_price carries NaNs in the bundled kuznets panel.
    nan_col = next(c for c in kuznets.columns if kuznets[c].isna().any())
    with pytest.warns(ExpdpyWarning, match="dropped"):
        ex.explore_histogram(kuznets, var=nan_col)


def test_public_function_silent_on_complete_data(kuznets):
    complete = kuznets[["gini_regional", "log_gdp_pc"]].dropna()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ex.explore_scatter_plot(complete, x="log_gdp_pc", y="gini_regional")


def test_clearer_xtsum_error_names_missing_columns(kuznets):
    with pytest.raises(ValueError, match=r"xtsum: column\(s\) not found in df"):
        ex.explore_xtsum_table(
            kuznets, var=["definitely_not_a_column"], entity="country"
        )
