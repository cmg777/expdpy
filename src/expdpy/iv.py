"""Instrumental-variables / two-stage least squares (2SLS) via pyfixest.

Two public entry points share one fitting core:

* :func:`analyze_iv_regression` — general IV/2SLS (cross-section by default), and
* :func:`analyze_panel_iv_regression` — the panel-aware wrapper that absorbs entity (and,
  by default, time) fixed effects and clusters by entity, i.e. pyfixest's ``xtivreg2 ... fe``.

Both use pyfixest's native IV formula (``dv ~ exog | fe | endog ~ instruments``), so the
estimation stays in the tested backend rather than a hand-rolled 2SLS. Alongside the
coefficient table each surfaces the **first-stage F statistic**, the standard weak-instrument
diagnostic.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import pandas as pd
import pyfixest as pf
from pandas.api import types as pdt

from expdpy._estimation import (
    SSC,
    VCovSpec,
    as_list,
    build_vcov,
    capture_stdout,
    tidy_model,
)
from expdpy._labels import label_map
from expdpy._panel import resolve_panel
from expdpy._types import IVRegressionResult
from expdpy._validation import ensure_dataframe

__all__ = ["analyze_iv_regression", "analyze_panel_iv_regression"]

#: Conventional Staiger-Stock rule-of-thumb threshold for a weak first stage.
WEAK_IV_F = 10.0


def _scalar(value: Any) -> float:
    """Coerce a pyfixest diagnostic (scalar or 1-element array) to ``float`` (NaN on fail)."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _iv_formula(
    dv: str,
    exog: list[str],
    endog: list[str],
    instruments: list[str],
    feffects: list[str],
) -> str:
    """Build pyfixest's IV formula ``dv ~ exog | fe | endog ~ instruments``.

    The fixed-effects block is omitted when there are no fixed effects, matching pyfixest's
    two-part IV syntax (``dv ~ exog | endog ~ instruments``).
    """
    exog_str = " + ".join(exog) if exog else "1"
    iv_str = f"{' + '.join(endog)} ~ {' + '.join(instruments)}"
    if feffects:
        return f"{dv} ~ {exog_str} | {' + '.join(feffects)} | {iv_str}"
    return f"{dv} ~ {exog_str} | {iv_str}"


def _render_etable(
    models: list[Any],
    format: str,
    labels: dict[str, str] | None,
) -> Any:
    """Render the pyfixest ``etable`` in the requested ``format`` (mirrors the OLS table)."""
    etable_type = "gt" if format == "html" else format
    if etable_type == "md":
        with capture_stdout() as buf:
            pf.etable(models, type="md", labels=labels)
        return buf.getvalue()
    etable = pf.etable(models, type=etable_type, labels=labels)
    if format == "html" and hasattr(etable, "as_raw_html"):
        return etable.as_raw_html()
    return etable


def _iv_result(
    df: pd.DataFrame,
    dv: str,
    endog: list[str],
    instruments: list[str],
    exog: list[str],
    feffects: list[str],
    clusters: list[str],
    format: str,
) -> IVRegressionResult:
    """Validate, fit a single 2SLS model and assemble the :class:`IVRegressionResult`."""
    if not endog:
        raise ValueError("at least one endogenous regressor is required")
    if not instruments:
        raise ValueError("at least one instrument is required")
    if len(instruments) < len(endog):
        raise ValueError(
            "the model is under-identified: need at least as many instruments "
            f"({len(instruments)}) as endogenous regressors ({len(endog)})"
        )
    if not (pdt.is_numeric_dtype(df[dv]) or pdt.is_bool_dtype(df[dv])):
        raise NotImplementedError(
            f"dependent variable '{dv}' is non-numeric; IV is supported only for the "
            "OLS family (2SLS), not logit/multinomial models."
        )

    used = list(dict.fromkeys([dv, *exog, *endog, *instruments, *feffects, *clusters]))
    data = df[used].dropna().copy()
    for fe in feffects:
        data[fe] = data[fe].astype("category")

    fml = _iv_formula(dv, exog, endog, instruments, feffects)
    vcov_spec = (
        VCovSpec(kind="CRV1", cluster=tuple(clusters)) if clusters else VCovSpec()
    )
    vcov, vcov_kwargs = build_vcov(vcov_spec)
    kwargs: dict[str, Any] = {"vcov": vcov, "ssc": SSC}
    if vcov_kwargs is not None:
        kwargs["vcov_kwargs"] = vcov_kwargs
    model = pf.feols(fml, data=data, **kwargs)

    labels = label_map(df) or None
    return IVRegressionResult(
        model=model,
        etable=_render_etable([model], format, labels),
        df=tidy_model(model, 1),
        endog=tuple(endog),
        instruments=tuple(instruments),
        exog=tuple(exog),
        first_stage_f=_scalar(getattr(model, "_f_stat_1st_stage", float("nan"))),
        first_stage_p=_scalar(getattr(model, "_p_value_1st_stage", float("nan"))),
    )


def analyze_iv_regression(
    df: pd.DataFrame,
    dv: str,
    endog: Sequence[str] | str,
    instruments: Sequence[str] | str,
    exog: Sequence[str] | None = None,
    feffects: Sequence[str] | None = None,
    clusters: Sequence[str] | None = None,
    *,
    format: Literal["gt", "tex", "md", "df", "html"] = "gt",
) -> IVRegressionResult:
    """Fit an instrumental-variables (2SLS) regression with a weak-instrument diagnostic.

    Endogenous regressors are instrumented with excluded ``instruments`` while ``exog``
    (included exogenous controls) and ``feffects`` (absorbed fixed effects) enter directly.
    The fit delegates to pyfixest's native IV estimator, and the result carries the
    **first-stage F statistic** — the conventional check for weak instruments (a value below
    about 10 is the Staiger-Stock warning sign). For panel data, prefer
    :func:`analyze_panel_iv_regression`, which manages the entity/time fixed effects for you.

    Parameters
    ----------
    df
        Data frame containing the data.
    dv
        Dependent (outcome) variable name.
    endog
        Endogenous regressor name(s) to be instrumented.
    instruments
        Excluded instrument name(s). At least as many instruments as endogenous regressors
        are required (the order condition); more instruments over-identify the model.
    exog
        Included exogenous regressor (control) names, if any.
    feffects
        Fixed-effects variable names absorbed by pyfixest.
    clusters
        Cluster variable name(s) for cluster-robust standard errors.
    format
        Output format for the rendered ``etable``: ``"gt"`` (Great Tables), ``"tex"``,
        ``"md"``, ``"df"`` (DataFrame) or ``"html"``.

    Returns
    -------
    IVRegressionResult
        ``model`` (the fitted pyfixest IV model), ``etable`` (rendered table), ``df`` (tidy
        coefficient frame), the ``endog`` / ``instruments`` / ``exog`` names, and
        ``first_stage_f`` / ``first_stage_p`` (the first-stage weak-instrument F and p-value).

    Raises
    ------
    ValueError
        If no endogenous regressor or instrument is given, or the model is under-identified
        (fewer instruments than endogenous regressors).
    NotImplementedError
        If the dependent variable is non-numeric (only OLS-family IV is supported).

    Examples
    --------
    The canonical Acemoglu-Johnson-Robinson (2001) example: instrument average protection
    against expropriation risk with log settler mortality, on the 64-country base sample. The
    instrumented slope is the famous ≈ 0.94:

    ```python
    import expdpy as ex
    from expdpy.data import load_colonial_origins, load_colonial_origins_data_def

    df = ex.set_labels(load_colonial_origins(), load_colonial_origins_data_def())
    base = df[df["base_sample"] == 1]

    result = ex.analyze_iv_regression(
        base,
        dv="log_gdp_pc_1995",
        endog="expropriation_risk",
        instruments="log_settler_mortality",
    )
    result.etable
    result.first_stage_f
    print(result.interpret())
    ```
    """
    df = ensure_dataframe(df)
    return _iv_result(
        df,
        dv,
        as_list(endog),
        as_list(instruments),
        as_list(exog),
        as_list(feffects),
        as_list(clusters),
        format,
    )


def analyze_panel_iv_regression(
    df: pd.DataFrame,
    dv: str,
    endog: Sequence[str] | str,
    instruments: Sequence[str] | str,
    exog: Sequence[str] | None = None,
    *,
    entity: str | None = None,
    time: str | None = None,
    twoway: bool = True,
    cluster_entity: bool = True,
    format: Literal["gt", "tex", "md", "df", "html"] = "gt",
) -> IVRegressionResult:
    """Fit a panel IV (2SLS) regression absorbing entity (and time) fixed effects.

    The panel analogue of :func:`analyze_iv_regression` and of Stata's ``xtivreg2 ... fe``:
    it instruments the endogenous regressor(s) while absorbing **entity** fixed effects (and,
    when ``twoway``, **time** fixed effects), clustering by entity by default. The panel ids
    follow the usual expdpy resolution — explicit ``entity`` / ``time`` win, otherwise the pair
    declared by :func:`expdpy.set_panel` is used.

    Parameters
    ----------
    df
        Data frame containing the data.
    dv
        Dependent (outcome) variable name.
    endog
        Endogenous regressor name(s) to be instrumented.
    instruments
        Excluded instrument name(s) (order condition: at least as many as ``endog``).
    exog
        Included exogenous regressor (control) names, if any.
    entity, time
        Panel identifiers (unit and period). Resolved from :func:`expdpy.set_panel` when not
        passed; ``entity`` is required, ``time`` is needed only for the two-way specification.
    twoway
        If ``True`` (default) absorb both entity and time fixed effects; if ``False`` absorb
        only entity fixed effects (one-way).
    cluster_entity
        If ``True`` (default) cluster the standard errors by ``entity``.
    format
        Output format for the rendered ``etable`` (see :func:`analyze_iv_regression`).

    Returns
    -------
    IVRegressionResult
        As :func:`analyze_iv_regression`, with the entity/time fixed effects absorbed.

    Examples
    --------
    Instrument night-time lights (a proxy for local economic activity) with lagged rainfall and
    drought across African regions, absorbing region and year fixed effects and clustering by
    region — a panel ``xtivreg2 fe``. The first-stage F is well above 10:

    ```python
    import expdpy as ex
    from expdpy.data import load_regional_conflict, load_regional_conflict_data_def

    df = ex.set_labels(
        load_regional_conflict(), load_regional_conflict_data_def(), set_panel=True
    )

    result = ex.analyze_panel_iv_regression(
        df,
        dv="conflict",
        endog="log_lights_lag1",
        instruments=["rain_lag2", "drought_lag2"],
    )
    result.etable
    result.first_stage_f
    print(result.interpret())
    ```
    """
    df = ensure_dataframe(df)
    entity, time = resolve_panel(df, entity=entity, time=time, require_entity=True)
    feffects = [entity] if entity else []
    if twoway and time:
        feffects.append(time)
    clusters = [entity] if (cluster_entity and entity) else []
    return _iv_result(
        df,
        dv,
        as_list(endog),
        as_list(instruments),
        as_list(exog),
        feffects,
        clusters,
        format,
    )
