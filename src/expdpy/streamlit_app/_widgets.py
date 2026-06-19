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


def selectbox(
    label: str,
    options: list[str],
    key: str,
    *,
    none: bool = False,
    default: str | None = None,
    help: str | None = None,
) -> Any:
    """Render a ``st.selectbox`` whose stored value is coerced into ``options``.

    When ``none`` is true a leading ``"None"`` sentinel is prepended (for optional
    selectors).
    """
    opts = (["None", *options] if none else list(options)) or ["None"]
    if key in st.session_state:
        if st.session_state[key] not in opts:
            st.session_state[key] = opts[0]
        return st.selectbox(label, opts, key=key, help=help)
    init = default if default in opts else opts[0]
    return st.selectbox(label, opts, index=opts.index(init), key=key, help=help)


def multiselect(
    label: str,
    options: list[str],
    key: str,
    *,
    default: list[str] | None = None,
    help: str | None = None,
) -> list[str]:
    """Render a ``st.multiselect`` that drops values no longer present in ``options``."""
    opts = list(options)
    if key in st.session_state:
        st.session_state[key] = [v for v in st.session_state[key] if v in opts]
        return st.multiselect(label, opts, key=key, help=help)
    init = [v for v in (default or []) if v in opts]
    return st.multiselect(label, opts, default=init, key=key, help=help)


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
