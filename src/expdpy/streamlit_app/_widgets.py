"""Small Streamlit widget helpers with safe ``session_state`` handling.

Selection widgets bind to ``st.session_state`` under stable input keys
(``hist_var``, ``scatter_x``, ``reg_x`` …) so that selections persist across pages and a
loaded configuration round-trips for free. The helpers guarantee the stored value is always a
valid option *before* the widget is instantiated — otherwise Streamlit raises when a loaded
config references a column that no longer exists in the current data.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from expdpy.streamlit_app._appcore import _cluster_vars as cluster_vars

__all__ = [
    "selectbox",
    "multiselect",
    "slider",
    "checkbox",
    "cluster_vars",
]

#: Stored values that are *not* column names and so are shown verbatim (never relabelled).
_FMT_SKIP = {"None", "All", "Full Sample"}


def _label_for(value: Any, label_map: dict[str, str]) -> str:
    """Format a raw option as ``"Label (var_name)"`` using ``label_map``.

    Sentinels, numeric codes and non-strings, and columns without a distinct label, fall back
    to the bare value — so the stored/selected value always stays the raw name.
    """
    if not isinstance(value, str):
        return str(value)
    if value in _FMT_SKIP or value.isdigit():
        return value
    label = label_map.get(value)
    return f"{label} ({value})" if label and label != value else value


def _fmt(relabel: bool) -> Any:
    """Build the ``format_func`` for a selector, capturing the label map *now*.

    The map is read from ``st.session_state`` at widget-creation time (during the run, when
    session state is live) and baked into the returned closure, so the formatter stays correct
    even when Streamlit's test harness re-evaluates it outside a run (where session state is
    unavailable). ``relabel=False`` yields ``str`` (raw display).
    """
    if not relabel:
        return str
    label_map = dict(st.session_state.get("_label_map", {}))
    return lambda value: _label_for(value, label_map)


def selectbox(
    label: str,
    options: list[str],
    key: str,
    *,
    none: bool = False,
    default: str | None = None,
    help: str | None = None,
    relabel: bool = True,
) -> Any:
    """Render a ``st.selectbox`` whose stored value is coerced into ``options``.

    When ``none`` is true a leading ``"None"`` sentinel is prepended (for optional
    selectors). When ``relabel`` is true (the default) options display as ``"Label (name)"``
    via :func:`_label_for`, while the stored/selected value stays the raw column name.
    """
    opts = (["None", *options] if none else list(options)) or ["None"]
    fmt = _fmt(relabel)
    if key in st.session_state:
        if st.session_state[key] not in opts:
            st.session_state[key] = opts[0]
        return st.selectbox(label, opts, key=key, help=help, format_func=fmt)
    init = default if default in opts else opts[0]
    return st.selectbox(
        label, opts, index=opts.index(init), key=key, help=help, format_func=fmt
    )


def multiselect(
    label: str,
    options: list[str],
    key: str,
    *,
    default: list[str] | None = None,
    help: str | None = None,
    relabel: bool = True,
) -> list[str]:
    """Render a ``st.multiselect`` that drops values no longer present in ``options``.

    When ``relabel`` is true (the default) options display as ``"Label (name)"`` while the
    stored/selected values stay the raw column names.
    """
    opts = list(options)
    fmt = _fmt(relabel)
    if key in st.session_state:
        st.session_state[key] = [v for v in st.session_state[key] if v in opts]
        return st.multiselect(label, opts, key=key, help=help, format_func=fmt)
    init = [v for v in (default or []) if v in opts]
    return st.multiselect(
        label, opts, default=init, key=key, help=help, format_func=fmt
    )


def slider(
    label: str,
    min_value: int,
    max_value: int,
    key: str,
    *,
    default: int,
    help: str | None = None,
) -> int:
    """Render an integer ``st.slider`` that respects a stored ``session_state`` value."""
    if key in st.session_state:
        return st.slider(label, min_value, max_value, key=key, help=help)
    return st.slider(
        label, min_value, max_value, value=int(default), key=key, help=help
    )


def float_slider(
    label: str,
    min_value: float,
    max_value: float,
    key: str,
    *,
    default: float,
    step: float = 0.1,
    help: str | None = None,
) -> float:
    """Render a float ``st.slider`` that respects a stored ``session_state`` value."""
    if key in st.session_state:
        return st.slider(label, min_value, max_value, step=step, key=key, help=help)
    return st.slider(
        label, min_value, max_value, value=float(default), step=step, key=key, help=help
    )


def checkbox(
    label: str,
    key: str,
    *,
    default: bool = False,
    help: str | None = None,
) -> bool:
    """Render a ``st.checkbox`` that respects a stored ``session_state`` value."""
    if key in st.session_state:
        return st.checkbox(label, key=key, help=help)
    return st.checkbox(label, value=bool(default), key=key, help=help)
