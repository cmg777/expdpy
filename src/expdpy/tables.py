"""Tabular summaries: descriptive statistics, correlations, extreme observations."""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from string import ascii_uppercase
from typing import Any

import numpy as np
import pandas as pd
from great_tables import GT
from pandas.api import types as pdt

from expdpy._corr import cor_mat
from expdpy._labels import resolve_label, resolve_labels
from expdpy._panel import resolve_panel, stored_panel
from expdpy._types import (
    CorrelationTableResult,
    DescriptiveTableResult,
    ExtObsTableResult,
)
from expdpy._validation import ensure_dataframe, numeric_logical_columns

__all__ = [
    "explore_correlation_table",
    "explore_descriptive_table",
    "explore_ext_obs_table",
]

_ALL_STATS = ("N", "Mean", "Std. dev.", "Min.", "25 %", "Median", "75 %", "Max.")
_DEFAULT_STATS = ("Mean", "Std. dev.", "Median", "Min.", "Max.")
_MAX_MISSING_LISTED = 8


def _excel_letters(n: int) -> list[str]:
    """Return spreadsheet-style labels A, B, ..., Z, AA, AB, ... of length ``n``."""
    letters = list(ascii_uppercase)
    labels = list(letters)
    for first in letters:
        labels.extend(first + second for second in letters)
    return labels[:n]


def _compute_stats(num: pd.DataFrame) -> pd.DataFrame:
    """Return the eight descriptive statistics (as columns) for each column of ``num``."""
    quantiles = num.quantile([0.0, 0.25, 0.5, 0.75, 1.0], interpolation="linear")
    return pd.DataFrame(
        {
            "N": num.notna().sum().astype(int),
            "Mean": num.mean(),
            "Std. dev.": num.std(ddof=1),
            "Min.": quantiles.loc[0.0],
            "25 %": quantiles.loc[0.25],
            "Median": quantiles.loc[0.5],
            "75 %": quantiles.loc[0.75],
            "Max.": quantiles.loc[1.0],
        }
    )[list(_ALL_STATS)]


def _resolve_stats(stats: Sequence[str]) -> list[str]:
    """Validate and de-duplicate the requested statistics, preserving order."""
    out: list[str] = []
    for s in stats:
        if s not in _ALL_STATS:
            raise ValueError(
                f"unknown statistic {s!r}; choose from {', '.join(_ALL_STATS)}"
            )
        if s not in out:
            out.append(s)
    if not out:
        raise ValueError("'stats' must name at least one statistic")
    return out


def _digits_for(stat: str, digits: int | Mapping[str, int]) -> int:
    """Return the number of decimals for ``stat`` (always 0 for the count ``N``)."""
    if stat == "N":
        return 0
    if isinstance(digits, Mapping):
        return int(digits.get(stat, 3))
    return int(digits)


def _sorted_unique(values: pd.Series) -> list:
    """Distinct non-missing values of ``values``, sorted when orderable."""
    uniq = list(pd.unique(pd.Series(values).dropna()))
    try:
        return sorted(uniq)
    except TypeError:  # pragma: no cover - unorderable mixed period types
        return uniq


def explore_descriptive_table(
    df: pd.DataFrame,
    stats: Sequence[str] = _DEFAULT_STATS,
    *,
    digits: int | Mapping[str, int] = 3,
    periods: Sequence[object] | None = None,
    entity: str | None = None,
    time: str | None = None,
    caption: str = "Descriptive Statistics",
) -> DescriptiveTableResult:
    """Report descriptive statistics for the numeric/logical variables of ``df``.

    The table reflects the panel structure of the data. When a ``time`` column is known
    (declared via :func:`expdpy.set_panel` / :func:`expdpy.set_labels`, or passed
    explicitly), each statistic is shown **by period** — by default at the first and last
    period — under a spanning column header (e.g. ``Mean`` over ``2015`` and ``2025``).
    Without a time dimension the table falls back to a single column per statistic. Rows
    are labelled from the data dictionary when available, and the number of observations
    and any variable with missing data are reported in the notes beneath the table.

    Parameters
    ----------
    df
        Data frame containing at least one numeric/logical variable and two observations.
    stats
        Statistics to display, in order, chosen from ``N``, ``Mean``, ``Std. dev.``,
        ``Min.``, ``25 %``, ``Median``, ``75 %``, ``Max.``. Defaults to ``Mean``,
        ``Std. dev.``, ``Median``, ``Min.``, ``Max.``. (The returned ``.df`` always carries
        all eight statistics regardless of this selection.)
    digits
        Number of decimals for the displayed statistics: a single ``int`` applied to all,
        or a ``{statistic: decimals}`` mapping for per-statistic overrides (``N`` is always
        shown as an integer).
    periods
        Periods to show as sub-columns in the by-period layout. ``None`` (default) shows
        the first and last period; otherwise the listed period values (those not present
        are dropped with a warning). Ignored when no time dimension is summarized.
    entity, time
        Optional panel identifiers (defaulting to those declared via
        :func:`expdpy.set_panel`). A resolved ``time`` drives the by-period layout; when
        both resolve, a note also reports the panel dimensions. For the within/between
        split of each variable see :func:`expdpy.explore_xtsum_table`.
    caption
        Table title used for the Great Tables header.

    Returns
    -------
    DescriptiveTableResult
        ``df`` (the pooled eight-statistic summary), ``gt`` (a Great Tables object) and
        ``by_period`` (a tidy ``variable``-by-``period`` frame, or ``None``).

    Examples
    --------
    Basic — declare the panel via the data dictionary, then summarize by period (first and
    last year) with labelled rows:

    ```python
    import expdpy as ex
    from expdpy.data import load_kuznets, load_kuznets_data_def

    df = ex.set_labels(load_kuznets(), load_kuznets_data_def(), set_panel=True)
    ex.explore_descriptive_table(df).gt
    ```

    Advanced — choose the statistics and decimals, pick specific periods, and read the tidy
    by-period frame back from ``.by_period``:

    ```python
    import expdpy as ex
    from expdpy.data import load_kuznets, load_kuznets_data_def

    df = ex.set_labels(load_kuznets(), load_kuznets_data_def(), set_panel=True)
    result = ex.explore_descriptive_table(
        df,
        stats=("Mean", "Median", "Std. dev."),
        digits=2,
        periods=[2015, 2020, 2025],
        caption="Kuznets panel",
    )
    result.gt
    result.by_period.head()
    ```
    """
    df = ensure_dataframe(df)
    # Validate an explicitly-named id strictly (the named column must exist), then fill any
    # unset id from the declared panel — but tolerate a *declared* id whose column was
    # dropped by a column subset (a descriptive table run on a subset should still
    # summarize the columns it has, falling back to the flat layout).
    for which, col in (("entity", entity), ("time", time)):
        if col is not None and col not in df.columns:
            raise ValueError(f"{which} column {col!r} is not in df")
    stored_entity, stored_time = stored_panel(df)
    if entity is None and stored_entity in df.columns:
        entity = stored_entity
    if time is None and stored_time in df.columns:
        time = stored_time
    stats = _resolve_stats(stats)
    cols = numeric_logical_columns(df)
    # In a panel the entity/time ids are coordinates, not variables to summarize.
    id_cols = {c for c in (entity, time) if c is not None}
    var_cols = [c for c in cols if c not in id_cols]
    if len(var_cols) < 1 or len(df) < 2:
        raise ValueError("unsuitable data frame (does it contain numerical data?)")

    # `.df` is always the pooled, full eight-statistic summary (raw-name index).
    pooled = _compute_stats(df[var_cols].astype(float))

    # Notes shared by both layouts: observation count, then any missing-data variables.
    notes = [f"Observations: {len(df):,}."]
    missing = [
        (resolve_label(df, c), int(df[c].isna().sum()))
        for c in var_cols
        if df[c].isna().any()
    ]
    if missing:
        shown = ", ".join(f"{lab} ({k:,})" for lab, k in missing[:_MAX_MISSING_LISTED])
        if len(missing) > _MAX_MISSING_LISTED:
            shown += f", and {len(missing) - _MAX_MISSING_LISTED} more"
        notes.append(f"Variables with missing data: {shown}.")
    if entity is not None and time is not None:
        n_entities = int(df[entity].nunique())
        n_periods = int(df[time].nunique())
        t_bar = (
            len(df.dropna(subset=[entity])) / n_entities if n_entities else float("nan")
        )
        notes.append(
            f"Panel: {n_entities:,} units ({resolve_label(df, entity)}) over "
            f"{n_periods:,} periods ({resolve_label(df, time)}); {t_bar:.1f} "
            "observations per unit on average."
        )

    if time is None:
        # Overall (flat) layout: one column per statistic, labelled rows.
        disp = pd.DataFrame({"Variable": resolve_labels(df, var_cols)})
        for s in stats:
            disp[s] = pooled[s].to_numpy()
        gt = GT(disp, rowname_col="Variable").tab_header(title=caption)
        for s in stats:
            gt = gt.fmt_number(
                columns=s, decimals=_digits_for(s, digits), use_seps=True
            )
        for note in notes:
            gt = gt.tab_source_note(note)
        return DescriptiveTableResult(df=pooled, gt=gt, by_period=None)

    # By-period layout: a spanner per statistic over its selected periods.
    all_periods = _sorted_unique(df[time])
    if periods is None:
        sel = list(dict.fromkeys([all_periods[0], all_periods[-1]]))
    else:
        present = set(all_periods)
        sel = [p for p in periods if p in present]
        for p in periods:
            if p not in present:
                warnings.warn(
                    f"period {p!r} not present in {time!r}; dropped", stacklevel=2
                )
        if not sel:
            raise ValueError(f"none of the requested periods are present in {time!r}")

    per_period = {
        p: _compute_stats(df.loc[df[time] == p, var_cols].astype(float)) for p in sel
    }

    # Wide display frame: one positionally-encoded column per (statistic, period).
    disp = pd.DataFrame({"Variable": resolve_labels(df, var_cols)})
    col_periods: dict[str, Any] = {}
    spanner_cols: dict[str, list[str]] = {}
    for si, s in enumerate(stats):
        keys = []
        for pi, p in enumerate(sel):
            key = f"s{si}_p{pi}"
            disp[key] = per_period[p][s].reindex(var_cols).to_numpy()
            col_periods[key] = str(p)
            keys.append(key)
        spanner_cols[s] = keys

    gt = GT(disp, rowname_col="Variable").tab_header(title=caption)
    for s in stats:
        gt = gt.tab_spanner(label=s, columns=spanner_cols[s])
        gt = gt.fmt_number(
            columns=spanner_cols[s], decimals=_digits_for(s, digits), use_seps=True
        )
    gt = gt.cols_label(col_periods).sub_missing(missing_text="")
    if len(sel) < len(all_periods):
        # Describe the span by chronological extent, not the order periods= was typed in.
        try:
            span = sorted(sel)
            lo, hi = span[0], span[-1]
        except TypeError:  # pragma: no cover - unorderable mixed period types
            lo, hi = sel[0], sel[-1]
        notes.append(
            f"Columns show {len(sel)} of {len(all_periods)} periods ({lo} to {hi})."
        )
    for note in notes:
        gt = gt.tab_source_note(note)

    # Tidy long by-period frame keyed by variable and period (raw names, shown statistics).
    records = [
        {"variable": var, "period": p, **{s: per_period[p].loc[var, s] for s in stats}}
        for p in sel
        for var in var_cols
    ]
    return DescriptiveTableResult(
        df=pooled, gt=gt, by_period=pd.DataFrame.from_records(records)
    )


def explore_correlation_table(
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
    Basic — Pearson (above) and Spearman (below) correlations for a few variables
    (slice the columns first, then label so the data dictionary supplies row names):

    ```python
    import expdpy as ex
    from expdpy.data import load_kuznets, load_kuznets_data_def

    df = ex.set_labels(
        load_kuznets()[["gini_regional", "gdp_pc", "log_gdp_pc"]],
        load_kuznets_data_def(),
    )
    ex.explore_correlation_table(df).gt
    ```

    Advanced — more decimals, a stricter bold threshold, a caption, and the raw
    coefficient/p-value matrices from ``.df_corr`` / ``.df_prob``:

    ```python
    import expdpy as ex
    from expdpy.data import load_kuznets, load_kuznets_data_def

    df = ex.set_labels(
        load_kuznets()[["gini_regional", "gdp_pc", "log_gdp_pc", "trade_share"]],
        load_kuznets_data_def(),
    )
    result = ex.explore_correlation_table(
        df,
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
    labels_src = df  # resolve labels before the column reslice drops df.attrs
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
    name_labels = resolve_labels(labels_src, names)
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
        0,
        "term",
        [f"{lab}: {lbl}" for lab, lbl in zip(labels, name_labels, strict=True)],
    )

    n_min, n_max = int(corr_n.to_numpy().min()), int(corr_n.to_numpy().max())
    if n_min == n_max:
        n_note = f"Number of observations: {n_min:,}."
    elif n_min == 0:
        n_note = (
            f"The number of overlapping observations ranges from 0 (no pair of complete "
            f"cases) to {n_max:,}."
        )
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


def explore_ext_obs_table(
    df: pd.DataFrame,
    n: int = 5,
    var: str | None = None,
    *,
    entity: Sequence[str] | str | None = None,
    time: str | None = None,
    digits: int = 3,
) -> ExtObsTableResult:
    """Display the top and bottom ``n`` observations sorted by ``var``.

    Parameters
    ----------
    df
        Data frame.
    n
        Number of extreme observations on each side (``2 * n <= len(df)``).
    var
        Variable to sort by (must be numeric). Defaults to the last numeric column that is
        not an identifier.
    entity
        Cross-sectional identifier column(s). Defaults to the panel ``entity`` declared via
        :func:`expdpy.set_panel`. If both ``entity`` and ``time`` are unset (and none is
        declared), all variables are tabulated; otherwise only the identifiers and ``var``.
    time
        Time identifier column. Defaults to the panel ``time``.
    digits
        Number of decimals for numeric cells.

    Returns
    -------
    ExtObsTableResult
        ``df`` (the ``2 * n`` extreme rows) and ``gt`` (a Great Tables object grouping the
        highest and lowest blocks).

    Examples
    --------
    Basic — the five highest and lowest observations (sorted by the last numeric
    column), tabulating all variables:

    ```python
    import expdpy as ex
    from expdpy.data import load_kuznets, load_kuznets_data_def

    df = ex.set_labels(load_kuznets(), load_kuznets_data_def(), set_panel=True)
    ex.explore_ext_obs_table(df, n=5).gt
    ```

    Advanced — the ten most extreme observations of a chosen variable, showing only
    the panel identifiers and that variable (the panel is taken from the declared
    ``entity``/``time``):

    ```python
    import expdpy as ex
    from expdpy.data import load_kuznets, load_kuznets_data_def

    df = ex.set_labels(load_kuznets(), load_kuznets_data_def(), set_panel=True)
    ex.explore_ext_obs_table(df, n=10, var="gini_regional").gt
    ```
    """
    df = ensure_dataframe(df)
    if 2 * n > len(df):
        raise ValueError("'n' needs to be <= nrow(df) / 2")

    if entity is None and time is None:
        entity, time = resolve_panel(df, None, None)
    entity_list = (
        [entity] if isinstance(entity, str) else (list(entity) if entity else [])
    )
    id_cols = entity_list + ([time] if time else [])
    if var is None:
        candidates = [
            c for c in df.columns if c not in id_cols and pdt.is_numeric_dtype(df[c])
        ]
        if not candidates:
            raise ValueError("no numeric variable available to sort by")
        var = candidates[-1]
    if var not in df.columns:
        raise ValueError("var needs to be in df")
    if not pdt.is_numeric_dtype(df[var]):
        raise ValueError(f"var ({var!r}) needs to be numeric to sort by")

    cols = [*id_cols, var] if id_cols else [c for c in df.columns if c != var] + [var]

    sub = df.loc[np.isfinite(df[var].to_numpy(dtype=float)), cols]
    ordered = sub.sort_values(var, ascending=False)
    top = ordered.head(n)
    bottom = ordered.tail(n)
    out = pd.concat([top, bottom])

    # Group the highest and lowest blocks rather than separating them with a blank row.
    group_col = "  "  # a near-invisible header for the grouping stub
    display = pd.concat(
        [
            top.assign(**{group_col: f"Highest {n}"}),
            bottom.assign(**{group_col: f"Lowest {n}"}),
        ],
        ignore_index=True,
    )

    gt = (
        GT(display, groupname_col=group_col)
        .cols_label({c: resolve_label(df, c) for c in cols})
        .fmt_number(
            columns=[c for c in cols if pdt.is_numeric_dtype(df[c])],
            decimals=digits,
            use_seps=True,
        )
        .sub_missing(missing_text="...")
    )
    return ExtObsTableResult(df=out, gt=gt)
