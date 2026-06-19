"""The app body executed on every Streamlit rerun.

``run_app`` is import-safe (it has no module-level side effects), so it can be driven both by
Streamlit's script runner (via :mod:`expdpy.streamlit_app._run`) and by
``streamlit.testing.v1.AppTest`` in the test-suite. A pre-built :class:`AppContext` may be
injected through ``_ctx`` to bypass the bundle/dataset resolution in tests.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from expdpy.streamlit_app._context import AppContext, resolve_context
from expdpy.streamlit_app._pages import build_pages
from expdpy.streamlit_app._sidebar import apply_pending_config, render_sidebar
from expdpy.streamlit_app._state import parse_config

__all__ = ["run_app"]

_EMOJI_ICON = "📊"


def _page_icon() -> Any:
    """Return the packaged expdpy logo for the browser tab, falling back to an emoji."""
    try:
        from importlib.resources import files
        from io import BytesIO

        from PIL import Image

        data = files("expdpy").joinpath("_assets/favicon.png").read_bytes()
        return Image.open(BytesIO(data))
    except Exception:
        return _EMOJI_ICON


def run_app(_ctx: AppContext | None = None) -> None:
    """Resolve the data source, render the sidebar, and run the multipage navigation."""
    if _ctx is not None:
        st.session_state["_ctx"] = _ctx
        ctx = _ctx
    else:
        ctx = resolve_context()

    st.set_page_config(page_title=ctx.title, page_icon=_page_icon(), layout="wide")
    # On the first run of a session, apply the startup configuration passed at launch
    # (``config_list`` / the bundle's ``base_cfg``) so the app opens on the preset selections
    # — e.g. the kuznets N-curve. Seeded once, so later user edits are never reset.
    if not st.session_state.get("_base_cfg_seeded"):
        st.session_state["_base_cfg_seeded"] = True
        if ctx.base_cfg:
            st.session_state.setdefault("_pending_cfg", parse_config(ctx.base_cfg))
    apply_pending_config(ctx)
    active = render_sidebar(ctx)
    st.navigation(build_pages(active)).run()
