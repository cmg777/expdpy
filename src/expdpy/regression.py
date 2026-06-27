"""Regression tables: OLS with fixed effects and clustered SEs via pyfixest."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

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
from expdpy._labels import label_map
from expdpy._types import RegressionTableResult
from expdpy._validation import drop_missing, ensure_dataframe

__all__ = ["analyze_regression_table"]

# Backwards-compatible aliases re-exported for expdpy.fwl (the FWL plot reuses the same
# small-sample correction and list-normalization to stay numerically consistent).
_SSC = SSC
_as_list = as_list
_tidy = tidy_model


def _fit_one(
    df: pd.DataFrame,
    dv: str,
    idvs: list[str],
    feffects: list[str],
    clusters: list[str],
) -> Any:
    """Fit a single OLS model with optional fixed effects and clustered SEs."""
    if not (pdt.is_numeric_dtype(df[dv]) or pdt.is_bool_dtype(df[dv])):
        raise NotImplementedError(
            f"dependent variable '{dv}' is non-numeric; logit/multinomial models are not "
            "supported by expdpy (OLS only)."
        )
    used = list(dict.fromkeys([dv, *idvs, *feffects, *clusters]))
    data = df[used].dropna().copy()
    for fe in feffects:
        data[fe] = data[fe].astype("category")

    vcov = VCovSpec(kind="CRV1", cluster=tuple(clusters)) if clusters else VCovSpec()
    spec = ModelSpec(dv=(dv,), idvs=tuple(idvs), feffects=tuple(feffects), vcov=vcov)
    return fit_model(data, spec)


def analyze_regression_table(
    df: pd.DataFrame,
    dvs: Sequence[str] | str,
    idvs: Sequence[str] | Sequence[Sequence[str]],
    feffects: Sequence[str] | Sequence[Sequence[str]] | None = None,
    clusters: Sequence[str] | Sequence[Sequence[str]] | None = None,
    *,
    byvar: str | None = None,
    format: Literal["gt", "tex", "md", "df", "html"] = "gt",
) -> RegressionTableResult:
    """Build a regression table of one or more OLS models.

    Supports fixed effects and clustered standard errors (via pyfixest), multiple models
    side by side, or a single model estimated separately across the levels of ``byvar``.

    Parameters
    ----------
    df
        Data frame containing the data.
    dvs
        Dependent variable name, or a list of names (one per model).
    idvs
        Independent variable names. For multiple models, a list of lists.
    feffects
        Fixed-effects variable names (per model when multiple models are given).
    clusters
        Cluster variable names for clustered standard errors (per model when multiple).
    byvar
        A categorical variable to estimate the single model separately by. Only valid with
        a single dependent variable. Levels with too few observations to estimate the model
        (``n <= len(idvs) + 1``) are skipped.
    format
        Output format for the rendered ``etable``: ``"gt"`` (Great Tables), ``"tex"``,
        ``"md"``, ``"df"`` (DataFrame) or ``"html"``.

    Returns
    -------
    RegressionTableResult
        ``models`` (fitted pyfixest models), ``etable`` (rendered table) and ``df`` (tidy
        coefficient frame).

    Examples
    --------
    Basic — a pooled OLS regression of the cubic Kuznets curve (the data
    dictionary supplies the readable labels shown in the rendered table):

    ```python
    import expdpy as ex
    from expdpy.data import load_kuznets, load_kuznets_data_def

    df = ex.set_labels(load_kuznets(), load_kuznets_data_def(), set_panel=True)
    ex.analyze_regression_table(
        df,
        dvs="gini_regional",
        idvs=["log_gdp_pc", "log_gdp_pc_sq", "log_gdp_pc_cu"],
    ).etable
    ```

    Advanced — absorb two-way (country + year) fixed effects with standard errors
    clustered by country, then read the tidy coefficient frame and fitted models:

    ```python
    import expdpy as ex
    from expdpy.data import load_kuznets, load_kuznets_data_def

    df = ex.set_labels(load_kuznets(), load_kuznets_data_def(), set_panel=True)
    result = ex.analyze_regression_table(
        df,
        dvs="gini_regional",
        idvs=["log_gdp_pc", "log_gdp_pc_sq", "log_gdp_pc_cu"],
        feffects=["country", "year"],
        clusters=["country"],
    )
    result.etable
    result.df
    result.models
    ```
    """
    df = ensure_dataframe(df)
    dvs_list = _as_list(dvs)
    multi = len(dvs_list) > 1

    if byvar is not None and multi:
        raise ValueError("you cannot subset multiple models in one table")

    models: list[Any] = []
    tidies: list[pd.DataFrame] = []

    if byvar is not None:
        dv = dvs_list[0]
        idv = _as_list(idvs)  # single model -> flat list
        fe = _as_list(feffects)
        cl = _as_list(clusters)
        used = list(dict.fromkeys([byvar, dv, *idv, *fe, *cl]))
        sub = drop_missing(df[used], used, func="analyze_regression_table").copy()
        if not (
            isinstance(sub[byvar].dtype, pd.CategoricalDtype)
            or pdt.is_object_dtype(sub[byvar])
        ):
            sub[byvar] = sub[byvar].astype("category")
        levels = sorted(sub[byvar].dropna().unique(), key=str)
        levels = [lvl for lvl in levels if (sub[byvar] == lvl).sum() > len(idv) + 1]
        if not levels:
            raise ValueError("no by levels with sufficient degrees of freedom")

        full = _fit_one(sub, dv, idv, fe, cl)
        models.append(full)
        tidies.append(_tidy(full, 1, "Full Sample"))
        for k, lvl in enumerate(levels, start=2):
            m = _fit_one(sub[sub[byvar] == lvl], dv, idv, fe, cl)
            models.append(m)
            tidies.append(_tidy(m, k, str(lvl)))
    else:
        if multi:
            idvs_seq = list(idvs)
            fe_seq = list(feffects) if feffects is not None else [[]] * len(dvs_list)
            cl_seq = list(clusters) if clusters is not None else [[]] * len(dvs_list)
            for k, dv in enumerate(dvs_list):
                m = _fit_one(
                    df,
                    dv,
                    _as_list(idvs_seq[k]),
                    _as_list(fe_seq[k]),
                    _as_list(cl_seq[k]),
                )
                models.append(m)
                tidies.append(_tidy(m, k + 1, None))
        else:
            m = _fit_one(
                df,
                dvs_list[0],
                _as_list(idvs),
                _as_list(feffects),
                _as_list(clusters),
            )
            models.append(m)
            tidies.append(_tidy(m, 1, None))

    # Relabel the rendered table's coefficient/dependent-variable rows from the data
    # dictionary when available; the tidy ``.df`` keeps the raw term names.
    labels = label_map(df) or None
    etable_type = "gt" if format == "html" else format
    if etable_type == "md":
        # pyfixest prints markdown to stdout and returns None; capture it.
        with capture_stdout() as buf:
            pf.etable(models, type="md", labels=labels)
        etable: Any = buf.getvalue()
    else:
        etable = pf.etable(models, type=etable_type, labels=labels)
        if format == "html" and hasattr(etable, "as_raw_html"):
            etable = etable.as_raw_html()

    tidy_df = pd.concat(tidies, ignore_index=True)
    return RegressionTableResult(models=models, etable=etable, df=tidy_df)
