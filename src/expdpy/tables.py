"""Tabular summaries: descriptive statistics, correlations, extreme observations."""

from __future__ import annotations

from collections.abc import Sequence
from string import ascii_uppercase

import numpy as np
import pandas as pd
from great_tables import GT

from expdpy._corr import cor_mat
from expdpy._types import (
    CorrelationTableResult,
    DescriptiveTableResult,
    ExtObsTableResult,
)
from expdpy._validation import ensure_dataframe, numeric_logical_columns

__all__ = [
    "prepare_correlation_table",
    "prepare_descriptive_table",
    "prepare_ext_obs_table",
]

_DESC_COLUMNS = ["N", "Mean", "Std. dev.", "Min.", "25 %", "Median", "75 %", "Max."]


def _excel_letters(n: int) -> list[str]:
    """Return spreadsheet-style labels A, B, ..., Z, AA, AB, ... of length ``n``."""
    letters = list(ascii_uppercase)
    labels = list(letters)
    for first in letters:
        labels.extend(first + second for second in letters)
    return labels[:n]


def prepare_descriptive_table(
    df: pd.DataFrame,
    digits: Sequence[int | None] = (0, 3, 3, 3, 3, 3, 3, 3),
    *,
    caption: str = "Descriptive Statistics",
) -> DescriptiveTableResult:
    """Report descriptive statistics for the numeric/logical variables of ``df``.

    For every numeric or logical column the function reports the number of non-missing
    observations, mean, standard deviation (``ddof = 1``), minimum, first quartile,
    median, third quartile and maximum.

    Parameters
    ----------
    df
        Data frame containing at least one numeric/logical variable and two observations.
    digits
        Sequence of length 8 giving the number of decimals for each statistic column
        (``N, Mean, Std. dev., Min., 25 %, Median, 75 %, Max.``). A value of ``None``
        drops that column from the output.
    caption
        Table title used for the Great Tables header.

    Returns
    -------
    DescriptiveTableResult
        ``df`` (the statistics table) and ``gt`` (a Great Tables object).

    Examples
    --------
    Basic — descriptive statistics for every numeric column of the panel:

    ```python
    import expdpy as ex
    from expdpy.data import load_kuznets

    df = load_kuznets()
    ex.prepare_descriptive_table(df).gt
    ```

    Advanced — set the decimals per statistic (``None`` drops that column), add a
    caption, and read the tidy statistics frame back from ``.df``:

    ```python
    result = ex.prepare_descriptive_table(
        df,
        digits=(0, 2, 2, None, None, 2, None, None),
        caption="Kuznets panel",
    )
    result.gt
    result.df.head()
    ```
    """
    df = ensure_dataframe(df)
    cols = numeric_logical_columns(df)
    if len(cols) < 1 or len(df) < 2:
        raise ValueError("unsuitable data frame (does it contain numerical data?)")
    digits = list(digits)
    if len(digits) != 8:
        raise ValueError("digits vector has wrong length (!= 8)")

    num = df[cols].astype(float)
    quantiles = num.quantile([0.0, 0.25, 0.5, 0.75, 1.0], interpolation="linear")
    stats = pd.DataFrame(
        {
            "N": df[cols].notna().sum().astype(int),
            "Mean": num.mean(),
            "Std. dev.": num.std(ddof=1),
            "Min.": quantiles.loc[0.0],
            "25 %": quantiles.loc[0.25],
            "Median": quantiles.loc[0.5],
            "75 %": quantiles.loc[0.75],
            "Max.": quantiles.loc[1.0],
        }
    )[_DESC_COLUMNS]

    keep = [c for c, d in zip(_DESC_COLUMNS, digits, strict=True) if d is not None]
    keep_digits = [d for d in digits if d is not None]
    stats = stats[keep]

    gt = GT(stats.reset_index(names="Variable"), rowname_col="Variable").tab_header(
        title=caption
    )
    for col, dec in zip(keep, keep_digits, strict=True):
        gt = gt.fmt_number(columns=col, decimals=int(dec), use_seps=True)
    return DescriptiveTableResult(df=stats, gt=gt)


def prepare_correlation_table(
    df: pd.DataFrame,
    digits: int = 2,
    bold: float = 0.05,
    *,
    caption: str | None = None,
) -> CorrelationTableResult:
    """Correlation table with Pearson above and Spearman below the diagonal.

    Parameters
    ----------
    df
        Data frame with at least two numeric/logical variables and five observations.
    digits
        Number of decimals to display (``0 < digits < 5``).
    bold
        Correlations with a p-value below ``bold`` are shown in bold. Set to ``0`` to
        disable. Must satisfy ``0 <= bold < 1``.
    caption
        Optional table title.

    Returns
    -------
    CorrelationTableResult
        ``df_corr``/``df_prob``/``df_n`` (numeric matrices keyed by the original variable
        names) and ``gt`` (a Great Tables object using letter labels).

    Examples
    --------
    Basic — Pearson (above) and Spearman (below) correlations for a few variables:

    ```python
    import expdpy as ex
    from expdpy.data import load_kuznets

    df = load_kuznets()
    ex.prepare_correlation_table(df[["gini_regional", "gdp_pc", "log_gdp_pc"]]).gt
    ```

    Advanced — more decimals, a stricter bold threshold, a caption, and the raw
    coefficient/p-value matrices from ``.df_corr`` / ``.df_prob``:

    ```python
    result = ex.prepare_correlation_table(
        df[["gini_regional", "gdp_pc", "log_gdp_pc", "trade_share"]],
        digits=3,
        bold=0.01,
        caption="Correlations (kuznets)",
    )
    result.gt
    result.df_corr
    result.df_prob
    ```
    """
    df = ensure_dataframe(df)
    df = df[numeric_logical_columns(df)]
    if len(df) < 5 or df.shape[1] < 2:
        raise ValueError(
            "'df' needs at least two variables and five observations of numerical data"
        )
    if not (0 < digits < 5):
        raise ValueError("'digits' needs to be a scalar with 0 < digits < 5")
    if not (0 <= bold < 1):
        raise ValueError("'bold' needs to be a scalar with 0 <= bold < 1")

    pcorr = cor_mat(df, "pearson")
    scorr = cor_mat(df, "spearman")
    lower = np.tril(np.ones(pcorr.r.shape, dtype=bool), k=-1)

    def _combine(p: pd.DataFrame, s: pd.DataFrame) -> pd.DataFrame:
        out = p.to_numpy(dtype=float).copy()
        out[lower] = s.to_numpy(dtype=float)[lower]
        return pd.DataFrame(out, index=p.index, columns=p.columns)

    corr_r = _combine(pcorr.r, scorr.r)
    corr_p = _combine(pcorr.p, scorr.p)
    corr_n = pd.DataFrame(
        np.where(lower, scorr.n.to_numpy(), pcorr.n.to_numpy()),
        index=pcorr.n.index,
        columns=pcorr.n.columns,
    ).astype("Int64")
    np.fill_diagonal(corr_p.values, 1.0)

    names = list(df.columns)
    labels = _excel_letters(len(names))
    fmt = f"{{:.{digits}f}}"
    r_arr = corr_r.to_numpy(dtype=float)
    p_arr = corr_p.to_numpy(dtype=float)
    display = np.empty(corr_r.shape, dtype=object)
    for i in range(corr_r.shape[0]):
        for j in range(corr_r.shape[1]):
            if i == j:
                display[i, j] = ""
            else:
                cell = fmt.format(r_arr[i, j])
                if bold > 0 and p_arr[i, j] < bold:
                    cell = f"**{cell}**"
                display[i, j] = cell

    disp_df = pd.DataFrame(display, columns=labels)
    disp_df.insert(
        0, "term", [f"{lab}: {name}" for lab, name in zip(labels, names, strict=True)]
    )

    n_min, n_max = int(corr_n.to_numpy().min()), int(corr_n.to_numpy().max())
    if n_min == n_max:
        n_note = f"Number of observations: {n_min:,}."
    else:
        n_note = f"The number of observations ranges from {n_min:,} to {n_max:,}."
    note = (
        "This table reports Pearson correlations above and Spearman correlations below "
        f"the diagonal. {n_note}"
    )
    if bold > 0:
        note += f" Correlations with significance levels below {bold * 100:.0f}% appear in bold."

    gt = (
        GT(disp_df, rowname_col="term")
        .fmt_markdown(columns=labels)
        .tab_source_note(note)
    )
    if caption is not None:
        gt = gt.tab_header(title=caption)

    return CorrelationTableResult(df_corr=corr_r, df_prob=corr_p, df_n=corr_n, gt=gt)


def prepare_ext_obs_table(
    df: pd.DataFrame,
    n: int = 5,
    cs_id: Sequence[str] | str | None = None,
    ts_id: str | None = None,
    var: str | None = None,
    *,
    digits: int = 3,
) -> ExtObsTableResult:
    """Display the top and bottom ``n`` observations sorted by ``var``.

    Parameters
    ----------
    df
        Data frame.
    n
        Number of extreme observations on each side (``2 * n <= len(df)``).
    cs_id
        Cross-sectional identifier column(s). If both ``cs_id`` and ``ts_id`` are omitted,
        all variables are tabulated; otherwise only the identifiers and ``var``.
    ts_id
        Time-series identifier column.
    var
        Variable to sort by. Defaults to the last numeric column that is not an identifier.
    digits
        Number of decimals for numeric cells.

    Returns
    -------
    ExtObsTableResult
        ``df`` (the ``2 * n`` extreme rows) and ``gt`` (a Great Tables object with a
        separator row between the top and bottom blocks).

    Examples
    --------
    Basic — the five highest and lowest observations (sorted by the last numeric
    column), tabulating all variables:

    ```python
    import expdpy as ex
    from expdpy.data import load_kuznets

    df = load_kuznets()
    ex.prepare_ext_obs_table(df, n=5).gt
    ```

    Advanced — the ten most extreme observations of a chosen variable, showing only
    the panel identifiers and that variable:

    ```python
    ex.prepare_ext_obs_table(
        df, n=10, cs_id=["country"], ts_id="year", var="gini_regional"
    ).gt
    ```
    """
    df = ensure_dataframe(df)
    if 2 * n > len(df):
        raise ValueError("'n' needs to be <= nrow(df) / 2")

    cs_list = [cs_id] if isinstance(cs_id, str) else (list(cs_id) if cs_id else [])
    id_cols = cs_list + ([ts_id] if ts_id else [])
    if var is None:
        candidates = [
            c
            for c in df.columns
            if c not in id_cols and pd.api.types.is_numeric_dtype(df[c])
        ]
        if not candidates:
            raise ValueError("no numeric variable available to sort by")
        var = candidates[-1]
    if var not in df.columns:
        raise ValueError("var needs to be in df")

    cols = [*id_cols, var] if id_cols else [c for c in df.columns if c != var] + [var]

    sub = df.loc[np.isfinite(df[var].to_numpy(dtype=float)), cols]
    ordered = sub.sort_values(var, ascending=False)
    top = ordered.head(n)
    bottom = ordered.tail(n)
    out = pd.concat([top, bottom])

    # A blank separator row between the top and bottom blocks. Build it as its own all-NaN
    # frame so concat unifies dtypes (int -> float) cleanly, instead of assigning NaN into
    # the existing int columns in place (which pandas now warns will become an error).
    separator = pd.DataFrame([[np.nan] * len(top.columns)], columns=top.columns)
    display = pd.concat([top, separator, bottom], ignore_index=True)

    gt = (
        GT(display)
        .fmt_number(
            columns=[c for c in cols if pd.api.types.is_numeric_dtype(df[c])],
            decimals=digits,
            use_seps=True,
        )
        .sub_missing(missing_text="...")
    )
    return ExtObsTableResult(df=out, gt=gt)
