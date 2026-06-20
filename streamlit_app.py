"""Streamlit Community Cloud entry point: a chooser for the three ExPdPy apps.

expdpy ships three module apps — **Explore**, **Analyze** and **Learn**. Deploy one of them
directly with its dedicated script (``app_explore.py`` / ``app_analyze.py`` /
``app_learn.py``), or point Streamlit Community Cloud at this file to get a small landing
page that lets the user pick a module. You can also pin this entry to a single module by
setting the ``EXPDPY_MODULE`` environment variable (``explore`` / ``analyze`` / ``learn``).

Run locally with::

    streamlit run streamlit_app.py
"""

import os

import streamlit as st

from expdpy.streamlit_app._entry import run_app

_MODULES = {
    "explore": "🔍 Explore — exploratory analysis of your panel data",
    "analyze": "🧮 Analyze — panel estimators (FE / RE / CRE, FWL, event study)",
    "learn": "📚 Learn — concept sandboxes and explainers",
}

_module = os.environ.get("EXPDPY_MODULE") or st.query_params.get("module")

if _module in _MODULES:
    run_app(module=_module)
else:
    st.set_page_config(page_title="ExPdPy", page_icon="📊", layout="centered")
    st.title("ExPdPy")
    st.write("Explore, Analyze and Learn panel data. Choose a module to open:")
    for key, label in _MODULES.items():
        if st.button(label, use_container_width=True):
            st.query_params["module"] = key
            st.rerun()
    st.caption(
        "Tip: deploy one app directly with `app_explore.py` / `app_analyze.py` / "
        "`app_learn.py`, or set the `EXPDPY_MODULE` environment variable."
    )
