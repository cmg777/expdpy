"""The OLS estimator with model comparison and panel-friendly standard errors.

``analyze_estimation`` is the home for everything beyond the classic OLS table that
``analyze_regression_table`` provides: richer standard errors (heteroskedastic,
cluster-robust, Newey-West / Driscoll-Kraay), weights, and stepwise / multi-outcome
comparison (estimate a sequence of nested models in one call and read them side by side).
It builds on the shared :mod:`expdpy._estimation` engine, so its numbers stay consistent
with the rest of the library.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal, cast

import pandas as pd
import pyfixest as pf
from pandas.api import types as pdt

from expdpy._estimation import (
    SSC,
    ModelSpec,
    VCovSpec,
    as_list,
    capture_stdout,
    fit_model,
    tidy_model,
)
from expdpy._estimation._spec import Stepwise, VCovKind
from expdpy._labels import label_map
from expdpy._types import EstimationResult
from expdpy._validation import ensure_dataframe

__all__ = ["analyze_estimation"]

_FEW_CLUSTERS = 40


def _resolve_vcov(
    vcov: str | Mapping[str, str] | VCovSpec | None,
    cluster: Sequence[str] | str | None,
    time_id: str | None,
    panel_id: str | None,
    lag: int | None,
) -> VCovSpec:
    """Build a :class:`VCovSpec` from the friendly ``analyze_estimation`` arguments."""
    if isinstance(vcov, VCovSpec):
        return vcov
    cl = as_list(cluster)
    if isinstance(vcov, Mapping):
        (kind, value), *_ = vcov.items()
        parts = tuple(p.strip() for p in str(value).split("+") if p.strip())
        return VCovSpec(kind=cast(VCovKind, kind), cluster=parts)
    kind = ("CRV1" if cl else "iid") if vcov is None else vcov
    return VCovSpec(
        kind=cast(VCovKind, kind),
        cluster=tuple(cl),
        time_id=time_id,
        panel_id=panel_id,
        lag=lag,
    )


def _fit_stats(models: list[Any]) -> pd.DataFrame:
    """Build a one-row-per-model fit-statistics frame (N, R², within-R²)."""
    rows = []
    for i, m in enumerate(models):
        rows.append(
            {
                "model": i + 1,
                "depvar": getattr(m, "_depvar", None),
                "N": int(getattr(m, "_N", 0)),
                "has_fe": bool(getattr(m, "_has_fixef", False)),
                "r2": float(getattr(m, "_r2", float("nan"))),
                "r2_within": float(getattr(m, "_r2_within", float("nan"))),
            }
        )
    return pd.DataFrame(rows)


def analyze_estimation(
    df: pd.DataFrame,
    dv: Sequence[str] | str,
    idvs: Sequence[str] | str | None = None,
    *,
    feffects: Sequence[str] | str | None = None,
    stepwise: Stepwise | None = None,
    vcov: str | Mapping[str, str] | VCovSpec | None = None,
    cluster: Sequence[str] | str | None = None,
    time_id: str | None = None,
    panel_id: str | None = None,
    lag: int | None = None,
    weights: str | None = None,
    ssc: Any | None = None,
    format: Literal["gt", "tex", "md", "df", "html"] = "gt",
) -> EstimationResult:
    """Estimate an OLS model, optionally several nested or multi-outcome models at once.

    A richer companion to :func:`expdpy.analyze_regression_table`: same OLS/fixed-effects
    core, but with stepwise nested-model comparison, serial-correlation-robust standard
    errors (Newey-West / Driscoll-Kraay), multiple outcomes and weights.

    Parameters
    ----------
    df
        Data frame containing the variables.
    dv
        Dependent-variable name, or several names to estimate the same right-hand side for
        each outcome side by side.
    idvs
        Independent regressor name(s).
    feffects
        Fixed-effect variable name(s) absorbed during estimation.
    stepwise
        Wrap ``idvs`` in a stepwise sequence — ``"sw"``/``"sw0"`` (separate) or
        ``"csw"``/``"csw0"`` (cumulative) — to estimate nested models in one call and read
        them side by side (great for "watch the estimate move as I add controls").
    vcov
        Standard-error type: ``"iid"``, ``"hetero"``/``"HC1"`` (``"HC2"``/``"HC3"`` without
        fixed effects), ``"CRV1"``/``"CRV3"`` (cluster-robust, with ``cluster=``),
        ``"NW"``/``"DK"`` (serial-correlation robust, with ``time_id=``/``panel_id=``), a
        pyfixest-style ``{"CRV1": "firm"}`` mapping, or a :class:`VCovSpec`. Defaults to
        clustered SEs when ``cluster`` is given, else ``"iid"``.
    cluster
        Cluster variable name(s) — a shortcut that selects ``"CRV1"`` standard errors.
    time_id, panel_id, lag
        Required for ``vcov="NW"``/``"DK"`` (the time and panel identifiers, optional lag).
    weights
        Optional weights column name.
    ssc
        Small-sample-correction object (defaults to the Stata-``reghdfe``-consistent one).
    format
        Output format for the rendered ``etable``.

    Returns
    -------
    EstimationResult
        ``models``, ``etable``, ``df`` (tidy coefficients), ``model_kind``, ``fit_stats``
        and ``notes``. Use ``.interpret()`` / ``.explain()`` for plain-language output.

    Examples
    --------
    Basic — a single OLS regression with heteroskedastic standard errors:

    ```python
    import expdpy as ex
    from expdpy.data import load_kuznets, load_kuznets_data_def

    df = ex.set_labels(load_kuznets(), load_kuznets_data_def(), set_panel=True)
    ex.analyze_estimation(
        df,
        dv="gini_regional",
        idvs=["log_gdp_pc", "log_gdp_pc_sq"],
        vcov="hetero",
    ).etable
    ```

    Advanced — a cumulative-stepwise comparison with heteroskedastic SEs:

    ```python
    import expdpy as ex
    from expdpy.data import load_kuznets, load_kuznets_data_def

    df = ex.set_labels(load_kuznets(), load_kuznets_data_def(), set_panel=True)
    res = ex.analyze_estimation(
        df,
        dv="gini_regional",
        idvs=["log_gdp_pc", "log_gdp_pc_sq", "log_gdp_pc_cu"],
        stepwise="csw",
        vcov="hetero",
    )
    res.etable          # three nested models side by side
    res.fit_stats       # N and R² per model
    print(res.interpret())
    ```
    """
    df = ensure_dataframe(df)
    dv_list = as_list(dv)
    idv_list = as_list(idvs)
    fe_list = as_list(feffects)

    if not dv_list:
        raise ValueError("at least one dependent variable is required")

    vspec = _resolve_vcov(vcov, cluster, time_id, panel_id, lag)
    extra = [c for c in (vspec.time_id, vspec.panel_id, weights) if c]
    used = list(
        dict.fromkeys(
            [
                *dv_list,
                *idv_list,
                *fe_list,
                *vspec.cluster,
                *extra,
            ]
        )
    )
    missing = [c for c in used if c not in df.columns]
    if missing:
        raise KeyError(f"columns not found in df: {missing}")

    data = df[used].dropna().copy()
    for fe in fe_list:
        data[fe] = data[fe].astype("category")

    # Newey-West / Driscoll-Kraay need numeric panel/time identifiers; encode any that are
    # non-numeric (factorize is order-preserving for sortable labels such as years).
    if vspec.kind in ("NW", "DK"):
        for col in (vspec.time_id, vspec.panel_id):
            if col and not pdt.is_numeric_dtype(data[col]):
                data[col] = pd.factorize(data[col], sort=True)[0]

    spec = ModelSpec(
        dv=tuple(dv_list),
        idvs=tuple(idv_list),
        feffects=tuple(fe_list),
        stepwise=stepwise,
        vcov=vspec,
        weights=weights,
    )
    fitted = fit_model(data, spec, ssc=ssc if ssc is not None else SSC)
    models = fitted.to_list() if hasattr(fitted, "to_list") else [fitted]

    tidy_df = pd.concat(
        [tidy_model(m, i + 1) for i, m in enumerate(models)], ignore_index=True
    )
    fit_stats = _fit_stats(models)

    notes: list[str] = []
    if vspec.kind in ("CRV1", "CRV3") and vspec.cluster:
        n_groups = min(int(data[c].nunique()) for c in vspec.cluster)
        if n_groups < _FEW_CLUSTERS:
            notes.append(
                f"Only {n_groups} clusters: cluster-robust inference is unreliable with few "
                "clusters — consider a wild cluster bootstrap via analyze_robust_inference()."
            )

    # Relabel the rendered table's rows from the data dictionary; ``.df`` stays raw.
    labels = label_map(df) or None
    etable_type = "gt" if format == "html" else format
    if etable_type == "md":
        with capture_stdout() as buf:
            pf.etable(models, type="md", labels=labels)
        etable: Any = buf.getvalue()
    else:
        etable = pf.etable(models, type=etable_type, labels=labels)
        if format == "html" and hasattr(etable, "as_raw_html"):
            etable = etable.as_raw_html()

    return EstimationResult(
        models=models,
        etable=etable,
        df=tidy_df,
        model_kind="ols",
        fit_stats=fit_stats,
        notes=tuple(notes),
    )
