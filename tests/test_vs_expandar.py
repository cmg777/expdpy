"""Numerical-parity tests against the original ExPanDaR R package via rpy2.

These run only when ``rpy2`` and the ``ExPanDaR`` R package are installed (the pixi ``r``
environment). They are marked ``against_r`` and skipped otherwise.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.against_r

rpy2 = pytest.importorskip("rpy2")
from rpy2.robjects import pandas2ri, r  # noqa: E402
from rpy2.robjects.packages import importr  # noqa: E402

try:
    expandar = importr("ExPanDaR")
except Exception:  # pragma: no cover - depends on the R environment
    pytest.skip("ExPanDaR R package not available", allow_module_level=True)

pandas2ri.activate()


@pytest.fixture(scope="module")
def r_df(sample_df):
    return pandas2ri.py2rpy(sample_df[["x1", "x2", "x3"]])


def test_descriptive_parity(sample_df, r_df):
    from expdpy import prepare_descriptive_table

    r_res = expandar.prepare_descriptive_table(r_df)
    r_tab = pandas2ri.rpy2py(r.as_data_frame(r_res.rx2("df")))
    py = prepare_descriptive_table(sample_df[["x1", "x2", "x3"]]).df
    np.testing.assert_allclose(
        py["Mean"].to_numpy(), r_tab["Mean"].to_numpy(), rtol=1e-6
    )
    np.testing.assert_allclose(
        py["Std. dev."].to_numpy(), r_tab["Std.dev."].to_numpy(), rtol=1e-6
    )


def test_correlation_parity(sample_df, r_df):
    from expdpy import prepare_correlation_table

    r_res = expandar.prepare_correlation_table(r_df)
    r_corr = pandas2ri.rpy2py(r.as_data_frame(r_res.rx2("df_corr")))
    py = prepare_correlation_table(sample_df[["x1", "x2", "x3"]]).df_corr
    np.testing.assert_allclose(
        py.to_numpy(dtype=float), r_corr.to_numpy(dtype=float), rtol=1e-5, atol=1e-6
    )


def test_treat_outliers_parity(sample_df):
    from expdpy import treat_outliers

    r_vec = expandar.treat_outliers(sample_df["x3"].to_numpy(), 0.01)
    py = treat_outliers(sample_df["x3"], 0.01)
    np.testing.assert_allclose(np.asarray(r_vec), py.to_numpy(), rtol=1e-9)


def test_regression_parity(sample_df):
    """Clustered-SE parity vs lfe::felm (cmethod='reghdfe') via prepare_regression_table."""
    from expdpy import prepare_regression_table

    r_res = expandar.prepare_regression_table(
        pandas2ri.py2rpy(sample_df),
        dvs="x2",
        idvs=r.c("x1", "x3"),
        feffects="firm",
        clusters="firm",
        format="text",
    )
    py = prepare_regression_table(
        sample_df, dvs="x2", idvs=["x1", "x3"], feffects=["firm"], clusters=["firm"]
    )
    r_model = r_res.rx2("models").rx2(1).rx2("model")
    r_coef = dict(
        zip(list(r.names(r.coef(r_model))), list(r.coef(r_model)), strict=False)
    )
    for term in ("x1", "x3"):
        assert float(py.models[0].coef()[term]) == pytest.approx(r_coef[term], rel=1e-5)
