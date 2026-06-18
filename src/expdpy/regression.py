"""Regression tables: OLS with fixed effects and clustered SEs via pyfixest."""

from __future__ import annotations

import contextlib
import io
from collections.abc import Sequence
from typing import Any, Literal

import pandas as pd
import pyfixest as pf
from pandas.api import types as pdt

from expdpy._types import RegressionTableResult
from expdpy._validation import ensure_dataframe

__all__ = ["prepare_regression_table"]

# Stata 'reghdfe'-consistent small-sample correction (matches lfe::felm cmethod='reghdfe').
_SSC = pf.ssc(k_adj=True, G_adj=True)


def _as_list(value: Any) -> list[str]:
    """Normalize ``None``/``""``/str/sequence into a flat list of non-empty strings."""
    if value is None or value == "":
        return []
    if isinstance(value, str):
        return [value]
    return [v for v in value if v]


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

    fml = f"{dv} ~ {' + '.join(idvs)}"
    if feffects:
        fml += " | " + " + ".join(feffects)
    vcov: Any = {"CRV1": " + ".join(clusters)} if clusters else "iid"
    return pf.feols(fml, data=data, vcov=vcov, ssc=_SSC)


def _tidy(model: Any, model_id: int, byvalue: str | None) -> pd.DataFrame:
    out = model.tidy().reset_index()
    out = out.rename(columns={out.columns[0]: "term"})
    out.insert(0, "model", model_id)
    if byvalue is not None:
        out["byvalue"] = byvalue
    return out


def prepare_regression_table(
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
        sub = df[used].dropna().copy()
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

    etable_type = "gt" if format == "html" else format
    if etable_type == "md":
        # pyfixest prints markdown to stdout and returns None; capture it.
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            pf.etable(models, type="md")
        etable: Any = buf.getvalue()
    else:
        etable = pf.etable(models, type=etable_type)
        if format == "html" and hasattr(etable, "as_raw_html"):
            etable = etable.as_raw_html()

    tidy_df = pd.concat(tidies, ignore_index=True)
    return RegressionTableResult(models=models, etable=etable, df=tidy_df)
