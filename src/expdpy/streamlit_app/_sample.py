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
    entities: Sequence[str] | None = None,
    time: str | None = None,
) -> pd.DataFrame:
    """Build an analysis sample by evaluating ``var_def`` expressions (advanced mode).

    Parameters
    ----------
    df
        Base sample.
    var_def
        A frame with at least ``var_name`` and ``var_def`` columns. Each ``var_def`` is a
        safe expression (see :func:`expdpy.streamlit_app._udv.evaluate_var_def`).
    entities, time
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
            out[name] = evaluate_var_def(str(expr), df, entities, time).to_numpy()
    return pd.DataFrame(out, index=df.index)


def build_analysis_sample(
    df: pd.DataFrame,
    entities: Sequence[str] | None,
    time: str | None,
    config: dict,
) -> pd.DataFrame:
    """Apply subset, balanced-panel and outlier-treatment steps to ``df``.

    The order mirrors ExPanDaR: subset selection -> balanced-panel filter -> outlier
    treatment.

    Parameters
    ----------
    df
        The (already constructed) analysis sample.
    entities, time
        Panel identifiers.
    config
        The app configuration (see :mod:`expdpy.streamlit_app._state`).

    Returns
    -------
    pandas.DataFrame
        The prepared sample.
    """
    entities = list(entities) if entities else []
    out = df.copy()

    # 1. Subset: period range, then category-value filters, then continuous-range filters
    #    (all combined with AND). Each is a no-op when its selection spans the full data.
    pr = config.get("period_range")
    if pr and time and time in out.columns:
        out = out[out[time].between(pr[0], pr[1])]
    for var, values in config.get("cat_filters", ()):  # ((var, (v1, v2, …)), …)
        if var in out.columns and values:
            wanted = {str(v) for v in values}
            out = out[out[var].astype(str).isin(wanted)]
    for var, (lo, hi) in config.get("range_filters", ()):  # ((var, (lo, hi)), …)
        if var in out.columns:
            out = out[out[var].between(lo, hi)]

    # 2. Balanced panel: keep cross-sections present in every period.
    if config.get("balanced_panel") and entities and time and time in out.columns:
        n_periods = out[time].nunique()
        counts = out.groupby(entities, observed=True)[time].nunique()
        keep = counts[counts == n_periods].index
        out = out.set_index(entities)
        out = out.loc[out.index.isin(keep)].reset_index()

    # 3. Outlier treatment.
    spec = _OUTLIER.get(str(config.get("outlier_treatment", "1")))
    if spec is not None:
        percentile, truncate = spec
        num_cols = [
            c
            for c in out.columns
            if c not in [*entities, time]
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
