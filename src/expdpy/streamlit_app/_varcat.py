"""Variable categorization for the ExPdPy app (port of create_var_categories)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import pandas as pd
from pandas.api import types as pdt

__all__ = ["VarCats", "create_var_categories"]


@dataclass
class VarCats:
    """Classification of a sample's columns, used to populate the app's selectors."""

    entities: list[str] = field(default_factory=list)
    times: list[str] = field(default_factory=list)
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

        Unlike :attr:`grouping`, this includes the entity (unit) and time identifiers
        (``entities`` / ``times``) so a panel can absorb the natural two-way (e.g. country +
        year) fixed effects.
        """
        return [*self.entities, *self.times, *self.factor, *self.logical]


def create_var_categories(
    df: pd.DataFrame,
    entities: Sequence[str] | None = None,
    time: str | None = None,
    *,
    factor_cutoff: int = 10,
    types: Mapping[str, str] | None = None,
) -> VarCats:
    """Classify the columns of ``df`` into id / numeric / logical / factor / two-level.

    By default a column is a *factor* if it is categorical/object, or numeric with at most
    ``factor_cutoff`` distinct non-missing values (and more than one); a *two_level* column has
    exactly two distinct values. When ``types`` is given (the data dictionary's declared
    ``type``), a declared column is routed **authoritatively** — ``numeric`` to numeric,
    ``factor`` to factor (grouping), ``logical`` to logical — overriding the dtype guess;
    columns with no declared type fall back to the inference above. Columns declared
    ``entity`` / ``time`` are skipped (they are panel ids, handled via ``entities`` / ``time``).

    Parameters
    ----------
    df
        The analysis sample.
    entities, time
        Entity (unit) / time identifier column names.
    factor_cutoff
        Numeric columns with at most this many unique values are treated as factors.
    types
        Optional ``{column: declared_type}`` map from the data dictionary, made authoritative.

    Returns
    -------
    VarCats
        The classification.
    """
    entities = list(entities) if entities else []
    ids = set(entities) | ({time} if time else set())
    types = types or {}
    vc = VarCats(entities=list(entities), times=[time] if time else [])

    for col in df.columns:
        if col in ids:
            continue
        declared = types.get(col)
        if declared in ("entity", "time"):
            continue  # a panel id, not a variable to offer in the selectors
        s = df[col]
        n_unique = int(s.dropna().nunique())
        if declared in ("numeric", "factor", "logical"):
            # The dictionary's declared type wins over the dtype-based guess.
            if declared == "numeric":
                vc.numeric.append(col)
            elif declared == "logical":
                vc.logical.append(col)
            elif n_unique > 1:  # factor
                vc.factor.append(col)
            if n_unique == 2:
                vc.two_level.append(col)
            continue
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
