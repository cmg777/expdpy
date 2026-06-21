"""Declare a panel's entity (unit) and time identifiers once, then reuse them.

Most Explore functions need both the cross-sectional **entity** id (the unit) and the
**time** id. Rather than repeat them on every call, :func:`set_panel` stashes the pair on the
frame's :attr:`pandas.DataFrame.attrs` and :func:`resolve_panel` reads them back. An explicit
argument passed to a function always wins over the stored default, so the helper is a
convenience, never a constraint.

Note that pandas does not always propagate ``attrs`` through operations (e.g. some merges or
column selections drop it). Call :func:`set_panel` again after such steps, or simply pass the
ids explicitly.
"""

from __future__ import annotations

import pandas as pd

from expdpy._validation import ensure_dataframe

__all__ = ["resolve_panel", "set_panel"]

_PANEL_KEY = "expdpy_panel"


def set_panel(
    df: pd.DataFrame, *, entity: str | None = None, time: str | None = None
) -> pd.DataFrame:
    """Declare the panel's ``entity`` and ``time`` columns on ``df`` and return it.

    The ids are stored under ``df.attrs["expdpy_panel"]`` so that subsequent ``prepare_*``
    calls can omit them. Explicit arguments to those functions still take precedence.

    Parameters
    ----------
    df
        The panel data frame (modified in place — its ``attrs`` are updated and the same
        object is returned).
    entity
        Name of the cross-sectional (unit) identifier column, or ``None`` to leave it unset.
    time
        Name of the time identifier column, or ``None`` to leave it unset.

    Returns
    -------
    pandas.DataFrame
        The same ``df``, with ``df.attrs["expdpy_panel"]`` updated.

    Examples
    --------
    Declare the panel once, then explore without repeating the ids:

    ```python
    import expdpy as ex
    from expdpy.data import load_kuznets

    df = ex.set_panel(load_kuznets(), entity="country", time="year")
    ex.prepare_xtsum_table(df, var=["gini_regional", "log_gdp_pc"]).gt
    ex.prepare_spaghetti_graph(df, var="gini_regional").fig
    ```
    """
    df = ensure_dataframe(df)
    for label, col in (("entity", entity), ("time", time)):
        if col is not None and col not in df.columns:
            raise ValueError(f"{label} column {col!r} is not in df")
    current = dict(df.attrs.get(_PANEL_KEY, {}))
    if entity is not None:
        current["entity"] = entity
    if time is not None:
        current["time"] = time
    df.attrs[_PANEL_KEY] = current
    return df


def resolve_panel(
    df: pd.DataFrame,
    entity: str | None = None,
    time: str | None = None,
    *,
    require_entity: bool = False,
    require_time: bool = False,
) -> tuple[str | None, str | None]:
    """Resolve the ``(entity, time)`` ids for ``df``: explicit args win, else ``df.attrs``.

    Parameters
    ----------
    df
        The panel data frame.
    entity, time
        Explicit identifiers. When ``None``, fall back to the values stored by
        :func:`set_panel` (if any).
    require_entity, require_time
        When ``True``, raise :class:`ValueError` if the corresponding id cannot be resolved.

    Returns
    -------
    tuple of (str or None, str or None)
        The resolved ``(entity, time)`` column names.

    Raises
    ------
    ValueError
        If a resolved column is not present in ``df``, or a required id is unresolved.
    """
    df = ensure_dataframe(df)
    stored = df.attrs.get(_PANEL_KEY, {})
    entity = entity if entity is not None else stored.get("entity")
    time = time if time is not None else stored.get("time")

    for label, col in (("entity", entity), ("time", time)):
        if col is not None and col not in df.columns:
            raise ValueError(f"{label} column {col!r} is not in df")
    if require_entity and entity is None:
        raise ValueError(
            "an entity (unit) id is required — pass entity=... or call set_panel(df, "
            "entity=...)"
        )
    if require_time and time is None:
        raise ValueError(
            "a time id is required — pass time=... or call set_panel(df, time=...)"
        )
    return entity, time
