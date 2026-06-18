"""Pairwise correlation engine shared by the correlation table and graph.

Faithful port of ExPanDaR's internal ``cor_mat()`` helper: for every pair of columns it
computes the correlation, two-sided p-value and the number of *pairwise* complete
observations (rows finite in both columns), placing Pearson or Spearman results depending
on ``method``. p-values use the asymptotic approximation (R's ``cor.test(..., exact = FALSE)``).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

__all__ = ["CorMat", "cor_mat"]


@dataclass(frozen=True)
class CorMat:
    """Square correlation/p-value/observation-count matrices (aligned by column name)."""

    r: pd.DataFrame
    p: pd.DataFrame
    n: pd.DataFrame


def cor_mat(df: pd.DataFrame, method: str) -> CorMat:
    """Compute a pairwise correlation matrix.

    Parameters
    ----------
    df
        Data frame of numeric/logical columns.
    method
        ``"pearson"`` or ``"spearman"``.

    Returns
    -------
    CorMat
        ``r`` (coefficients, diagonal 1.0), ``p`` (p-values, diagonal 0.0) and ``n``
        (pairwise observation counts, diagonal = non-missing count per column).
    """
    if method not in ("pearson", "spearman"):
        raise ValueError("method must be 'pearson' or 'spearman'")

    cols = list(df.columns)
    mat = df.to_numpy(dtype=float)
    n_cols = mat.shape[1]

    r = np.full((n_cols, n_cols), np.nan)
    p = np.full((n_cols, n_cols), np.nan)
    n = np.full((n_cols, n_cols), np.nan)

    finite = np.isfinite(mat)
    np.fill_diagonal(r, 1.0)
    np.fill_diagonal(p, 0.0)
    for k in range(n_cols):
        n[k, k] = int(finite[:, k].sum())

    corr_fn = stats.pearsonr if method == "pearson" else stats.spearmanr
    for i in range(n_cols - 1):
        for j in range(i + 1, n_cols):
            mask = finite[:, i] & finite[:, j]
            count = int(mask.sum())
            n[i, j] = n[j, i] = count
            if count > 2:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    coef, pval = corr_fn(mat[mask, i], mat[mask, j])
                r[i, j] = r[j, i] = float(coef)
                p[i, j] = p[j, i] = float(pval)

    return CorMat(
        r=pd.DataFrame(r, index=cols, columns=cols),
        p=pd.DataFrame(p, index=cols, columns=cols),
        n=pd.DataFrame(n, index=cols, columns=cols).astype("Int64"),
    )
