"""Numerical-parity tests for :func:`expdpy.analyze_kuznets_waves` against R's ``fixest``.

These run only when ``rpy2`` and the ``fixest`` R package are installed (the pixi ``r``
environment). They are marked ``against_r`` and skipped otherwise. pyfixest is the Python port
of ``fixest``, so the three estimators' polynomial coefficients should match to machine
precision on identical data. The between estimator is replicated exactly as the Python function
defines it: collapse to entity means, then form the polynomial from those means.
"""

from __future__ import annotations

import numpy as np
import pytest

import expdpy as ex

pytestmark = pytest.mark.against_r

rpy2 = pytest.importorskip("rpy2")
from rpy2.robjects import (  # noqa: E402
    default_converter,
    numpy2ri,
    pandas2ri,
    r,
)
from rpy2.robjects.conversion import localconverter  # noqa: E402
from rpy2.robjects.packages import importr  # noqa: E402

_CONVERTER = default_converter + numpy2ri.converter + pandas2ri.converter

try:
    fixest = importr("fixest")
except Exception:  # pragma: no cover - depends on the R environment
    pytest.skip("fixest R package not available", allow_module_level=True)

_TERMS = ["x2", "x2_p2", "x2_p3", "x2_p4"]  # the quartic regressor names


def _to_r(obj):
    with localconverter(_CONVERTER):
        return _CONVERTER.py2rpy(obj)


def _to_py(obj):
    with localconverter(_CONVERTER):
        return _CONVERTER.rpy2py(obj)


def _add_powers(frame):
    """Augment a frame with the same ``x2_p2 .. x2_p4`` columns the function forms internally."""
    out = frame.copy()
    g = out["x2"].to_numpy(dtype=float)
    out["x2_p2"], out["x2_p3"], out["x2_p4"] = g**2, g**3, g**4
    return out


def _r_betas(rframe, formula):
    """Fit ``formula`` with fixest and return the quartic coefficients in ``_TERMS`` order."""
    model = fixest.feols(r(formula), data=_to_r(rframe))
    coefs = r["coef"](model)
    names = list(r["names"](coefs))
    named = dict(zip(names, np.asarray(_to_py(coefs), dtype=float), strict=True))
    return np.array([named[t] for t in _TERMS])


def _py_betas(res, estimator):
    coef = res.models[estimator][-1].coef()
    return np.array([float(coef[t]) for t in _TERMS])


@pytest.fixture(scope="module")
def fitted(sample_df):
    return ex.analyze_kuznets_waves(
        sample_df, "x1", "x2", entity="firm", time="year", degree=4
    )


def test_pooled_quartic_matches_fixest(sample_df, fitted):
    rframe = _add_powers(sample_df)
    r_betas = _r_betas(rframe, "x1 ~ x2 + x2_p2 + x2_p3 + x2_p4")
    np.testing.assert_allclose(
        _py_betas(fitted, "pooled"), r_betas, rtol=1e-6, atol=1e-8
    )


def test_within_two_way_fe_matches_fixest(sample_df, fitted):
    rframe = _add_powers(sample_df)
    r_betas = _r_betas(rframe, "x1 ~ x2 + x2_p2 + x2_p3 + x2_p4 | firm + year")
    np.testing.assert_allclose(
        _py_betas(fitted, "within"), r_betas, rtol=1e-6, atol=1e-8
    )


def test_between_entity_means_match_fixest(sample_df, fitted):
    # Replicate the function's between design: collapse to entity means, then powers of the mean.
    means = sample_df[["firm", "x1", "x2"]].groupby("firm", as_index=False).mean()
    rframe = _add_powers(means)
    r_betas = _r_betas(rframe, "x1 ~ x2 + x2_p2 + x2_p3 + x2_p4")
    np.testing.assert_allclose(
        _py_betas(fitted, "between"), r_betas, rtol=1e-6, atol=1e-8
    )
