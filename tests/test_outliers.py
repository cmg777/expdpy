"""Tests for treat_outliers, including R golden parity and property-based checks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from expdpy import treat_outliers


def test_winsorize_matches_r_goldens(sample_df, goldens):
    for var, gold in goldens["winsorize_p05"].items():
        treated = treat_outliers(sample_df[var], 0.05)
        assert treated.min() == pytest.approx(gold["lo"], rel=1e-9)
        assert treated.max() == pytest.approx(gold["hi"], rel=1e-9)
        assert treated.sum() == pytest.approx(gold["sum"], rel=1e-9)


def test_vector_quantile_type7():
    # R: quantile(1:100, c(.05,.95), type=7) == 5.95, 95.05
    out = treat_outliers(np.arange(1, 101, dtype=float), 0.05)
    assert out[0] == pytest.approx(5.95)
    assert out[-1] == pytest.approx(95.05)


def test_truncate_sets_nan():
    out = treat_outliers(np.arange(1, 101, dtype=float), 0.05, truncate=True)
    assert np.isnan(out[0])
    assert np.isnan(out[-1])
    assert out[50] == 51


def test_nonfinite_becomes_nan(messy_series):
    out = treat_outliers(messy_series, 0.01)
    assert np.isnan(out.iloc[3])  # was +inf
    assert np.isnan(out.iloc[7])  # was -inf
    assert np.isnan(out.iloc[11])  # was nan


def test_dataframe_preserves_non_numeric_and_order():
    df = pd.DataFrame(
        {"a": np.arange(1, 101, dtype=float), "g": ["x"] * 100, "b": np.arange(100.0)}
    )
    out = treat_outliers(df, 0.05)
    assert list(out.columns) == ["a", "g", "b"]
    assert (out["g"] == "x").all()
    assert out["a"].min() == pytest.approx(5.95)


def test_grouped_treatment_preserves_index():
    rng = np.random.default_rng(1)
    df = pd.DataFrame(
        {"v": rng.normal(size=300), "g": rng.choice(["a", "b", "c"], 300)}
    )
    out = treat_outliers(df, 0.1, by="g")
    assert list(out.index) == list(df.index)
    # within each group, values are bounded by that group's cut-offs
    for g, part in out.groupby(df["g"]):
        lo, hi = np.nanquantile(df.loc[df["g"] == g, "v"], [0.1, 0.9])
        assert part["v"].min() >= lo - 1e-9
        assert part["v"].max() <= hi + 1e-9


def test_by_with_nan_raises():
    df = pd.DataFrame({"v": [1.0, 2, 3], "g": ["a", None, "b"]})
    with pytest.raises(ValueError, match="NA"):
        treat_outliers(df, 0.1, by="g")


@pytest.mark.parametrize("p", [-0.1, 0.0, 0.5, 0.7])
def test_bad_percentile_raises(p):
    with pytest.raises(ValueError):
        treat_outliers(np.arange(10.0), p)


@settings(max_examples=40, deadline=None)
@given(
    data=st.lists(
        st.floats(min_value=-1e6, max_value=1e6, allow_nan=False),
        min_size=20,
        max_size=200,
    ),
    pct=st.floats(min_value=0.01, max_value=0.45),
)
def test_property_bounds_and_length(data, pct):
    x = np.array(data, dtype=float)
    out = treat_outliers(x, pct)
    assert len(out) == len(x)
    lo, hi = np.nanquantile(x, [pct, 1 - pct])
    assert np.nanmin(out) >= lo - 1e-6
    assert np.nanmax(out) <= hi + 1e-6
