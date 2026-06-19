"""Tidy-coefficient-frame helper shared by the regression-style functions."""

from __future__ import annotations

from typing import Any

import pandas as pd

__all__ = ["tidy_model"]


def tidy_model(model: Any, model_id: int, byvalue: str | None = None) -> pd.DataFrame:
    """Return a tidy coefficient frame for one fitted model.

    Parameters
    ----------
    model
        A fitted pyfixest model exposing ``.tidy()``.
    model_id
        1-based identifier inserted as the ``model`` column (orders models in a table).
    byvalue
        Optional subgroup label inserted as a ``byvalue`` column (the ``byvar`` path).

    Returns
    -------
    pandas.DataFrame
        The model's ``tidy()`` frame with the coefficient index turned into a ``term``
        column and a leading ``model`` column (plus ``byvalue`` when given).
    """
    out = model.tidy().reset_index()
    out = out.rename(columns={out.columns[0]: "term"})
    out.insert(0, "model", model_id)
    if byvalue is not None:
        out["byvalue"] = byvalue
    return out
