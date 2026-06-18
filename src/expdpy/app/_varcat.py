"""Variable categorization for the ExPdPy app (port of create_var_categories)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import pandas as pd
from pandas.api import types as pdt

__all__ = ["VarCats", "create_var_categories"]


@dataclass
class VarCats:
    """Classification of a sample's columns, used to populate the app's selectors."""

    cs_id: list[str] = field(default_factory=list)
    ts_id: list[str] = field(default_factory=list)
    numeric: list[str] = field(default_factory=list)
    logical: list[str] = field(default_factory=list)
    factor: list[str] = field(default_factory=list)
    two_level: list[str] = field(default_factory=list)

    @property
    def numeric_logical(self) -> list[str]:
        """Numeric and logical columns combined (in column order is caller's concern)."""
        return [*self.numeric, *self.logical]

    @property
    def grouping(self) -> list[str]:
        """Columns usable for grouping/subsetting (factors and logicals)."""
        return [*self.factor, *self.logical]

    @property
    def fe_choices(self) -> list[str]:
        """Columns usable as fixed effects: panel identifiers + grouping factors.

        Unlike :attr:`grouping`, this includes the cross-sectional and time-series
        identifiers (``cs_id`` / ``ts_id``) so a panel can absorb the natural two-way
        (e.g. country + year) fixed effects.
        """
        return [*self.cs_id, *self.ts_id, *self.factor, *self.logical]


def create_var_categories(
    df: pd.DataFrame,
    cs_id: Sequence[str] | None = None,
    ts_id: str | None = None,
    *,
    factor_cutoff: int = 10,
) -> VarCats:
    """Classify the columns of ``df`` into id / numeric / logical / factor / two-level.

    A column is a *factor* if it is categorical/object, or numeric with at most
    ``factor_cutoff`` distinct non-missing values (and more than one). A *two_level*
    column has exactly two distinct values.

    Parameters
    ----------
    df
        The analysis sample.
    cs_id, ts_id
        Cross-sectional / time-series identifier column names.
    factor_cutoff
        Numeric columns with at most this many unique values are treated as factors.

    Returns
    -------
    VarCats
        The classification.
    """
    cs_id = list(cs_id) if cs_id else []
    ids = set(cs_id) | ({ts_id} if ts_id else set())
    vc = VarCats(cs_id=list(cs_id), ts_id=[ts_id] if ts_id else [])

    for col in df.columns:
        if col in ids:
            continue
        s = df[col]
        n_unique = int(s.dropna().nunique())
        if pdt.is_bool_dtype(s):
            vc.logical.append(col)
            if n_unique == 2:
                vc.two_level.append(col)
            continue
        is_cat = isinstance(s.dtype, pd.CategoricalDtype) or pdt.is_object_dtype(s)
        if is_cat:
            if n_unique > 1:
                vc.factor.append(col)
            if n_unique == 2:
                vc.two_level.append(col)
            continue
        # numeric
        vc.numeric.append(col)
        if 1 < n_unique <= factor_cutoff:
            vc.factor.append(col)
        if n_unique == 2:
            vc.two_level.append(col)
    return vc
