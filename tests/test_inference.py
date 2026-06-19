"""Tests for prepare_robust_inference (randomization inference, wild bootstrap)."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

import expdpy as ex


@pytest.fixture(scope="module")
def model(kuznets):
    return ex.prepare_regression_table(
        kuznets,
        dvs="gini_regional",
        idvs=["log_gdp_pc"],
        feffects=["country"],
        clusters=["country"],
    )


def test_ritest_runs(model):
    res = ex.prepare_robust_inference(
        model, "log_gdp_pc", method="ritest", reps=100, seed=1
    )
    assert res.method == "ritest"
    assert res.param == "log_gdp_pc"
    assert 0.0 <= res.p_value <= 1.0
    assert np.isfinite(res.estimate)
    assert len(res.conf_int) == 2


def test_ritest_is_reproducible(model):
    a = ex.prepare_robust_inference(model, "log_gdp_pc", reps=100, seed=7)
    b = ex.prepare_robust_inference(model, "log_gdp_pc", reps=100, seed=7)
    assert a.p_value == b.p_value


def test_unknown_method_raises(model):
    with pytest.raises(ValueError, match="unknown method"):
        ex.prepare_robust_inference(model, "log_gdp_pc", method="bogus")


@pytest.mark.skipif(
    importlib.util.find_spec("wildboottest") is not None,
    reason="wildboottest is installed, so the import-guard path is not exercised",
)
def test_wildboot_missing_package_message(model):
    with pytest.raises(ImportError, match="wildboottest"):
        ex.prepare_robust_inference(model, "log_gdp_pc", method="wildboot")
