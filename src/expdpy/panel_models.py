"""Classic panel estimators (pooled, between, fixed, random effects) and the Hausman test.

These complement the pyfixest-based core with the ``linearmodels`` package, which provides
the random-effects estimator and the Hausman fixed-vs-random-effects test that fixest does
not. ``linearmodels`` is an optional dependency: install it with ``pip install
'expdpy[panel]'``. It is imported lazily, so ``import expdpy`` works without it — the helpful
error appears only when these functions are actually called.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import stats

from expdpy._estimation import as_list
from expdpy._types import HausmanTestResult, RegressionTableResult
from expdpy._validation import ensure_dataframe

__all__ = ["prepare_hausman_test", "prepare_panel_table"]

_MODEL_LABELS = {
    "pooled": "Pooled OLS",
    "between": "Between",
    "fe": "Fixed effects",
    "re": "Random effects",
}


def _require_linearmodels() -> Any:
    """Import ``linearmodels.panel`` or raise a helpful install message."""
    try:
        import linearmodels.panel as lmp
    except ImportError as exc:  # pragma: no cover - exercised via the test monkeypatch
        raise ImportError(
            "random effects, the between estimator and the Hausman test require the optional "
            "'linearmodels' package; install it with:  pip install 'expdpy[panel]'"
        ) from exc
    return lmp


def _panel_frame(
    df: pd.DataFrame, dv: str, idvs: list[str], entity: str, time: str
) -> tuple[pd.Series, pd.DataFrame]:
    """Return the (y, X) pair indexed by (entity, time) for linearmodels."""
    cols = [entity, time, dv, *idvs]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"columns not found in df: {missing}")
    panel = df[cols].dropna().set_index([entity, time])
    return panel[dv], panel[idvs]


def _fit_cov(model: Any, cov_type: str, cluster_entity: bool) -> Any:
    """Fit a linearmodels model with the requested covariance, falling back if unsupported."""
    if cov_type == "clustered":
        try:
            return model.fit(cov_type="clustered", cluster_entity=cluster_entity)
        except (TypeError, ValueError):  # pragma: no cover - estimator-specific
            return model.fit(cov_type="robust")
    return model.fit(cov_type=cov_type)


def _attach_attrs(res: Any, dv: str, entity: str, kind: str, cov_type: str) -> None:
    """Attach pyfixest-style attributes so RegressionTableResult.interpret() understands it."""
    res._depvar = dv
    res._N = int(res.nobs)
    res._r2 = float(res.rsquared)
    res._has_fixef = kind == "fe"
    if kind == "fe":
        res._fixef = entity
        res._r2_within = float(getattr(res, "rsquared_within", float("nan")))
    res._is_clustered = cov_type == "clustered" and kind != "between"
    res._clustervar = [entity] if res._is_clustered else []


def _tidy_lm(res: Any, model_id: int) -> pd.DataFrame:
    """Build a pyfixest-compatible tidy frame from a linearmodels result."""
    terms = ["Intercept" if t == "const" else t for t in res.params.index]
    return pd.DataFrame(
        {
            "model": model_id,
            "term": terms,
            "Estimate": res.params.to_numpy(),
            "Std. Error": res.std_errors.to_numpy(),
            "t value": res.tstats.to_numpy(),
            "Pr(>|t|)": res.pvalues.to_numpy(),
        }
    )


def prepare_panel_table(
    df: pd.DataFrame,
    dv: str,
    idvs: Sequence[str] | str,
    *,
    entity: str,
    time: str,
    models: Sequence[Literal["pooled", "between", "fe", "re"]] = (
        "pooled",
        "between",
        "fe",
        "re",
    ),
    cov_type: Literal["clustered", "robust", "unadjusted"] = "clustered",
    cluster_entity: bool = True,
    format: Literal["gt", "md", "df", "html"] = "gt",
) -> RegressionTableResult:
    """Estimate pooled / between / fixed / random-effects models side by side.

    A linearmodels-backed companion to :func:`expdpy.prepare_regression_table`, returned in
    the same :class:`~expdpy.RegressionTableResult` container so ``.df``, ``.interpret()``
    and the apps work the same way. The one-way fixed-effects estimate matches
    ``prepare_regression_table(feffects=[entity])``.

    Parameters
    ----------
    df
        Long panel data frame.
    dv
        Dependent variable name.
    idvs
        Independent variable name(s).
    entity, time
        The cross-section and time identifiers.
    models
        Which estimators to include, in order.
    cov_type
        Covariance estimator: ``"clustered"`` (default), ``"robust"`` or ``"unadjusted"``.
    cluster_entity
        Cluster by entity when ``cov_type="clustered"``.
    format
        Output format for the rendered comparison table.

    Returns
    -------
    RegressionTableResult
        ``models`` (fitted linearmodels results), ``etable`` (the comparison table) and
        ``df`` (tidy coefficients with the same columns as the pyfixest path).
    """
    lmp = _require_linearmodels()
    df = ensure_dataframe(df)
    idv_list = as_list(idvs)
    if not idv_list:
        raise ValueError("at least one independent variable is required")
    y, x = _panel_frame(df, dv, idv_list, entity, time)
    x_const = x.assign(const=1.0)

    fitted: dict[str, Any] = {}
    model_list: list[Any] = []
    tidies: list[pd.DataFrame] = []
    for i, kind in enumerate(models, start=1):
        if kind == "pooled":
            res = _fit_cov(lmp.PooledOLS(y, x_const), cov_type, cluster_entity)
        elif kind == "between":
            res = lmp.BetweenOLS(y, x_const).fit()
        elif kind == "fe":
            res = _fit_cov(
                lmp.PanelOLS(y, x, entity_effects=True), cov_type, cluster_entity
            )
        elif kind == "re":
            res = _fit_cov(lmp.RandomEffects(y, x_const), cov_type, cluster_entity)
        else:  # pragma: no cover - guarded by the Literal type
            raise ValueError(f"unknown panel model {kind!r}")
        _attach_attrs(res, dv, entity, kind, cov_type)
        fitted[_MODEL_LABELS[kind]] = res
        model_list.append(res)
        tidies.append(_tidy_lm(res, i))

    tidy_df = pd.concat(tidies, ignore_index=True)
    comparison = lmp.compare(fitted)
    if format == "df":
        etable: Any = tidy_df
    elif format == "md":
        etable = str(comparison.summary)
    elif format == "html":
        etable = comparison.summary.as_html()
    else:  # gt / default: the comparison object renders via _repr_html_
        etable = comparison
    return RegressionTableResult(models=model_list, etable=etable, df=tidy_df)


def prepare_hausman_test(
    df: pd.DataFrame,
    dv: str,
    idvs: Sequence[str] | str,
    *,
    entity: str,
    time: str,
) -> HausmanTestResult:
    """Run the Hausman test comparing fixed-effects and random-effects estimates.

    Parameters
    ----------
    df
        Long panel data frame.
    dv
        Dependent variable name.
    idvs
        Independent variable name(s).
    entity, time
        The cross-section and time identifiers.

    Returns
    -------
    HausmanTestResult
        The test statistic, degrees of freedom, p-value and the compared coefficients.
    """
    lmp = _require_linearmodels()
    df = ensure_dataframe(df)
    idv_list = as_list(idvs)
    if not idv_list:
        raise ValueError("at least one independent variable is required")
    y, x = _panel_frame(df, dv, idv_list, entity, time)

    fe = lmp.PanelOLS(y, x, entity_effects=True).fit()
    re = lmp.RandomEffects(y, x.assign(const=1.0)).fit()

    common = [c for c in fe.params.index if c in re.params.index and c != "const"]
    if not common:  # pragma: no cover - defensive
        raise ValueError("no shared coefficients to compare between FE and RE")

    b_diff = (fe.params[common] - re.params[common]).to_numpy().reshape(-1, 1)
    v_diff = (
        fe.cov.loc[common, common].to_numpy() - re.cov.loc[common, common].to_numpy()
    )
    statistic = float((b_diff.T @ np.linalg.pinv(v_diff) @ b_diff).item())
    df_test = len(common)
    p_value = float(stats.chi2.sf(statistic, df_test))

    return HausmanTestResult(
        statistic=statistic,
        df_test=df_test,
        p_value=p_value,
        fe_coefs=pd.DataFrame(
            {"term": common, "estimate": fe.params[common].to_numpy()}
        ),
        re_coefs=pd.DataFrame(
            {"term": common, "estimate": re.params[common].to_numpy()}
        ),
    )
