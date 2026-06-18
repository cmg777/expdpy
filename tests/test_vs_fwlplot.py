"""Numerical-parity tests against the local fwlplot R package (Kyle Butts) via rpy2.

These run only when ``rpy2`` and the ``fwlplot``/``ggplot2`` R packages are installed (the
pixi ``r`` environment, with ``fwlplot`` installed from the bundled ``fwlplot-r/`` source).
They are marked ``against_r`` and skipped otherwise.

``fwl_plot(..., ggplot = TRUE)`` returns a ggplot whose ``$data`` is the residual frame
(columns ``x_resid``, ``y_resid``, ``fit``, ``lwr``, ``upr``); with ``n_sample = NULL`` it
keeps every row, so we can compare residuals and the fitted slope directly.
"""

from __future__ import annotations

import numpy as np
import pytest

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
_dollar = r["$"]  # R's ``$`` operator: extract a named element

try:
    fwlplot = importr("fwlplot")
    importr("ggplot2")  # required for ggplot = TRUE
except Exception:  # pragma: no cover - depends on the R environment
    pytest.skip("fwlplot/ggplot2 R packages not available", allow_module_level=True)


def _to_r(obj):
    with localconverter(_CONVERTER):
        return _CONVERTER.py2rpy(obj)


def _to_py(obj):
    with localconverter(_CONVERTER):
        return _CONVERTER.rpy2py(obj)


def _r_resids(formula: str, sample_df):
    """Residual frame from R's fwl_plot (ggplot=TRUE keeps the plotted data)."""
    gg = fwlplot.fwl_plot(r(formula), _to_r(sample_df), ggplot=True, n_sample=r("NULL"))
    return _to_py(_dollar(gg, "data")).sort_values("x_resid").reset_index(drop=True)


def test_fwl_residual_parity_with_fe(sample_df):
    """Residualized x/y and the fitted slope match fwlplot (controls + fixed effects)."""
    from expdpy import prepare_fwl_plot

    r_df = _r_resids("x2 ~ x1 + x3 | firm", sample_df)
    py = prepare_fwl_plot(
        sample_df, dv="x2", var="x1", controls=["x3"], feffects=["firm"], n_sample=None
    )
    assert len(py.df) == len(r_df)
    np.testing.assert_allclose(
        py.df["x_resid"].to_numpy(), r_df["x_resid"].to_numpy(), rtol=1e-4, atol=1e-6
    )
    np.testing.assert_allclose(
        py.df["y_resid"].to_numpy(), r_df["y_resid"].to_numpy(), rtol=1e-4, atol=1e-6
    )
    r_slope = float(np.polyfit(r_df["x_resid"], r_df["y_resid"], 1)[0])
    assert py.slope == pytest.approx(r_slope, rel=1e-5)


def test_fwl_slope_parity_no_fe(sample_df):
    """The no-fixed-effects (controls-only) path also matches fwlplot."""
    from expdpy import prepare_fwl_plot

    r_df = _r_resids("x2 ~ x1 + x3", sample_df)
    py = prepare_fwl_plot(sample_df, dv="x2", var="x1", controls=["x3"], n_sample=None)
    r_slope = float(np.polyfit(r_df["x_resid"], r_df["y_resid"], 1)[0])
    assert py.slope == pytest.approx(r_slope, rel=1e-5)
    np.testing.assert_allclose(
        py.df["x_resid"].to_numpy(), r_df["x_resid"].to_numpy(), rtol=1e-4, atol=1e-6
    )
