"""Correlated Random Effects (the Mundlak device).

A random-effects model augmented with the entity (unit) means of the time-varying
regressors. The Mundlak (1978) result is that adding each regressor's unit mean to a
random-effects model makes the coefficient on the original regressor equal the within
(fixed-effects) estimate, while the coefficient on the mean captures how far the
between-unit association is from the within-unit one. A joint test that the mean
coefficients are zero is algebraically the Hausman test — so CRE is the Hausman test "in
regression form", and a teaching bridge between random and fixed effects.

Built on the same ``linearmodels``-backed core as :mod:`expdpy.panel_models`, returned in a
:class:`~expdpy.CRETableResult` (a thin :class:`~expdpy.RegressionTableResult`) so ``.df``,
``.interpret()`` and the apps work the same way.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import stats

from expdpy._estimation import as_list
from expdpy._types import CRETableResult
from expdpy._validation import ensure_dataframe
from expdpy.panel_models import (
    _attach_attrs,
    _fit_cov,
    _panel_frame,
    _require_linearmodels,
    _tidy_lm,
)

__all__ = ["prepare_cre_table"]

_MEAN_SUFFIX = "_mean"


def prepare_cre_table(
    df: pd.DataFrame,
    dv: str,
    idvs: Sequence[str] | str,
    *,
    entity: str,
    time: str,
    cov_type: Literal["clustered", "robust", "unadjusted"] = "clustered",
    cluster_entity: bool = True,
    format: Literal["gt", "md", "df", "html"] = "gt",
) -> CRETableResult:
    """Estimate a Correlated Random Effects (Mundlak) model.

    Augments a random-effects model with the entity means of each time-varying regressor.
    By the Mundlak equivalence the coefficient on each original regressor equals its within
    (fixed-effects) estimate — so ``prepare_cre_table(df, "y", "x", ...)`` recovers the same
    slope on ``x`` as ``prepare_regression_table(df, "y", "x", feffects=[entity])`` — while
    the coefficient on ``x_mean`` measures the between-vs-within gap. A joint Wald test that
    all ``*_mean`` coefficients are zero is the regression-form Hausman test (reported on the
    fitted model as ``_cre_mundlak_stat`` / ``_cre_mundlak_df`` / ``_cre_mundlak_p``).

    Parameters
    ----------
    df
        Long panel data frame.
    dv
        Dependent variable name.
    idvs
        Time-varying independent variable name(s).
    entity, time
        The cross-section and time identifiers.
    cov_type
        Covariance estimator: ``"clustered"`` (default), ``"robust"`` or ``"unadjusted"``.
    cluster_entity
        Cluster by entity when ``cov_type="clustered"``.
    format
        Output format for the rendered table.

    Returns
    -------
    CRETableResult
        ``models`` (the fitted linearmodels result), ``etable`` (the rendered table) and
        ``df`` (tidy coefficients for each regressor, its ``*_mean`` and the intercept).

    Examples
    --------
    ```python
    import expdpy as ex
    from expdpy.data import load_kuznets

    df = load_kuznets()
    cre = ex.prepare_cre_table(
        df, dv="gini_regional", idvs=["log_gdp_pc"], entity="country", time="year"
    )
    cre.etable          # the log_gdp_pc coefficient == the within / fixed-effects estimate
    cre.df              # rows for log_gdp_pc, log_gdp_pc_mean and Intercept
    print(cre.interpret())
    ```
    """
    lmp = _require_linearmodels()
    df = ensure_dataframe(df)
    idv_list = as_list(idvs)
    if not idv_list:
        raise ValueError("at least one independent variable is required")

    y, x = _panel_frame(df, dv, idv_list, entity, time)
    # Entity-level means of each time-varying regressor, broadcast back to the panel index.
    entity_ids = x.index.get_level_values(entity)
    means = x.groupby(entity_ids).transform("mean")
    means.columns = pd.Index([f"{c}{_MEAN_SUFFIX}" for c in means.columns])
    x_aug = pd.concat([x, means], axis=1).assign(const=1.0)

    res = _fit_cov(lmp.RandomEffects(y, x_aug), cov_type, cluster_entity)
    _attach_attrs(res, dv, entity, kind="re", cov_type=cov_type)

    # Joint Wald test that all mean coefficients are zero — the regression-form Hausman test.
    mean_terms = list(means.columns)
    b = res.params[mean_terms].to_numpy().reshape(-1, 1)
    v = res.cov.loc[mean_terms, mean_terms].to_numpy()
    stat = float((b.T @ np.linalg.pinv(v) @ b).item())
    res._cre_means = mean_terms
    res._cre_mundlak_stat = stat
    res._cre_mundlak_df = len(mean_terms)
    res._cre_mundlak_p = float(stats.chi2.sf(stat, len(mean_terms)))

    tidy_df = _tidy_lm(res, 1)
    comparison = lmp.compare({"CRE (Mundlak)": res})
    if format == "df":
        etable: Any = tidy_df
    elif format == "md":
        etable = str(comparison.summary)
    elif format == "html":
        etable = comparison.summary.as_html()
    else:  # gt / default: the comparison object renders via _repr_html_
        etable = comparison
    return CRETableResult(models=[res], etable=etable, df=tidy_df)
