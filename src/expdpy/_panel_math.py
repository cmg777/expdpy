"""Within/between variance-decomposition primitives (Stata ``xtsum`` style).

These helpers split a variable's variation into a **between** component (differences across
units) and a **within** component (variation over time inside a unit) — the decomposition at
the heart of panel-data exploration and of the fixed-effects estimator. They are shared by the
``xtsum`` table, the within-vs-between scatter and the within-persistence view.

For a variable ``x_it`` with unit ``i``: write the grand mean as ``xbar`` and each unit mean
as ``xbar_i``. The **within** transform recenters on the grand mean, ``xtilde_it = x_it -
xbar_i + xbar`` (so its mean equals ``xbar``, matching Stata's printout), while the **between**
data are the ``n`` unit means ``xbar_i``.

All standard deviations use ``ddof = 1``. The identity ``Var_overall = Var_between +
Var_within`` holds exactly only for balanced panels; treat it as approximate otherwise.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

__all__ = ["entity_means", "panel_decompose", "within_demean"]


def panel_decompose(series: pd.Series, entity: pd.Series) -> dict[str, float]:
    """Decompose one numeric ``series`` into overall / between / within statistics.

    Parameters
    ----------
    series
        The numeric variable. Coerced to float; non-finite/NA values are dropped.
    entity
        The unit label for each observation, aligned positionally with ``series``.

    Returns
    -------
    dict
        Keys: ``n_obs``, ``n_entities``, ``t_bar``, ``overall_mean``, ``overall_sd``,
        ``overall_min``, ``overall_max``, ``between_sd``, ``between_min``, ``between_max``,
        ``within_sd``, ``within_min``, ``within_max``. Empty input yields all-``nan`` (with
        zero counts); ``between_sd`` is ``nan`` when fewer than two units contribute.
    """
    s = pd.to_numeric(pd.Series(np.asarray(series)), errors="coerce")
    g = pd.Series(np.asarray(entity)).reset_index(drop=True)
    s = s.reset_index(drop=True)
    mask = s.notna()
    s = s[mask]
    g = g[mask]

    nan = float("nan")
    n_obs = int(s.size)
    if n_obs == 0:
        return {
            "n_obs": 0,
            "n_entities": 0,
            "t_bar": nan,
            "overall_mean": nan,
            "overall_sd": nan,
            "overall_min": nan,
            "overall_max": nan,
            "between_sd": nan,
            "between_min": nan,
            "between_max": nan,
            "within_sd": nan,
            "within_min": nan,
            "within_max": nan,
        }

    xbar = float(s.mean())
    overall_sd = float(s.std(ddof=1)) if n_obs > 1 else nan

    means = s.groupby(g, observed=True).mean()
    n_entities = int(means.size)
    t_bar = n_obs / n_entities if n_entities else nan
    between_sd = float(means.std(ddof=1)) if n_entities > 1 else nan

    within = s - g.map(means) + xbar
    within_sd = float(within.std(ddof=1)) if n_obs > 1 else nan

    return {
        "n_obs": n_obs,
        "n_entities": n_entities,
        "t_bar": float(t_bar),
        "overall_mean": xbar,
        "overall_sd": overall_sd,
        "overall_min": float(s.min()),
        "overall_max": float(s.max()),
        "between_sd": between_sd,
        "between_min": float(means.min()),
        "between_max": float(means.max()),
        "within_sd": within_sd,
        "within_min": float(within.min()),
        "within_max": float(within.max()),
    }


def entity_means(df: pd.DataFrame, cols: Sequence[str], entity: str) -> pd.DataFrame:
    """Return per-unit time-averages of ``cols`` (the *between* data, one row per unit)."""
    return df.groupby(entity, observed=True)[list(cols)].mean()


def within_demean(
    df: pd.DataFrame,
    cols: Sequence[str],
    entity: str,
    *,
    add_grand_mean: bool = True,
) -> pd.DataFrame:
    """Return the within (unit-demeaned) transform of ``cols``.

    Subtracts each unit's mean from every observation. With ``add_grand_mean=True`` the grand
    mean is added back, so the result stays in the variable's natural units and shares its
    overall mean (the Stata ``xtsum`` within convention); with ``False`` the columns are
    centered at zero per unit.
    """
    cols = list(cols)
    grouped = df.groupby(entity, observed=True)[cols]
    demeaned = df[cols] - grouped.transform("mean")
    if add_grand_mean:
        demeaned = demeaned + df[cols].mean()
    return demeaned
