"""The ExPdPy analysis-sample pipeline (port of create_analysis_sample)."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from expdpy.outliers import treat_outliers
from expdpy.streamlit_app._udv import evaluate_var_def

__all__ = ["apply_user_vars", "build_analysis_sample"]

# outlier_treatment radio value -> (percentile, truncate)
_OUTLIER = {
    "1": None,
    "2": (0.01, False),
    "3": (0.05, False),
    "4": (0.01, True),
    "5": (0.05, True),
}


def apply_user_vars(
    df: pd.DataFrame,
    var_def: pd.DataFrame,
    cs_id: Sequence[str] | None = None,
    ts_id: str | None = None,
) -> pd.DataFrame:
    """Build an analysis sample by evaluating ``var_def`` expressions (advanced mode).

    Parameters
    ----------
    df
        Base sample.
    var_def
        A frame with at least ``var_name`` and ``var_def`` columns. Each ``var_def`` is a
        safe expression (see :func:`expdpy.streamlit_app._udv.evaluate_var_def`).
    cs_id, ts_id
        Panel identifiers for ``lag``/``lead``.

    Returns
    -------
    pandas.DataFrame
        The new analysis sample (one column per ``var_def`` row).
    """
    out = {}
    for _, row in var_def.iterrows():
        name, expr = row["var_name"], row["var_def"]
        if expr in df.columns:  # plain rename / passthrough
            out[name] = df[expr].to_numpy()
        else:
            out[name] = evaluate_var_def(str(expr), df, cs_id, ts_id).to_numpy()
    return pd.DataFrame(out, index=df.index)


def build_analysis_sample(
    df: pd.DataFrame,
    cs_id: Sequence[str] | None,
    ts_id: str | None,
    config: dict,
) -> pd.DataFrame:
    """Apply subset, balanced-panel and outlier-treatment steps to ``df``.

    The order mirrors ExPanDaR: subset selection -> balanced-panel filter -> outlier
    treatment.

    Parameters
    ----------
    df
        The (already constructed) analysis sample.
    cs_id, ts_id
        Panel identifiers.
    config
        The app configuration (see :mod:`expdpy.streamlit_app._state`).

    Returns
    -------
    pandas.DataFrame
        The prepared sample.
    """
    cs_id = list(cs_id) if cs_id else []
    out = df.copy()

    # 1. Subset by a factor level.
    sf = config.get("subset_factor", "Full Sample")
    sv = config.get("subset_value", "All")
    if (
        sf not in (None, "Full Sample")
        and sf in out.columns
        and sv not in (None, "All")
    ):
        out = out[out[sf].astype(str) == str(sv)]

    # 2. Balanced panel: keep cross-sections present in every period.
    if config.get("balanced_panel") and cs_id and ts_id and ts_id in out.columns:
        n_periods = out[ts_id].nunique()
        counts = out.groupby(cs_id, observed=True)[ts_id].nunique()
        keep = counts[counts == n_periods].index
        out = out.set_index(cs_id)
        out = out.loc[out.index.isin(keep)].reset_index()

    # 3. Outlier treatment.
    spec = _OUTLIER.get(str(config.get("outlier_treatment", "1")))
    if spec is not None:
        percentile, truncate = spec
        num_cols = [
            c
            for c in out.columns
            if c not in [*cs_id, ts_id]
            and pd.api.types.is_numeric_dtype(out[c])
            and not pd.api.types.is_bool_dtype(out[c])
        ]
        of = config.get("outlier_factor", "None")
        by = (
            out[of].to_numpy()
            if of not in (None, "None") and of in out.columns
            else None
        )
        if num_cols:
            out[num_cols] = treat_outliers(
                out[num_cols], percentile, truncate=truncate, by=by
            )
    return out
