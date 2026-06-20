"""Streamlit Community Cloud entry point for the ExPdPy **Learn** app.

Point Streamlit Community Cloud at this file, or run it locally with::

    streamlit run app_learn.py

It is the teaching surface: interactive concept sandboxes (omitted-variable bias, pooled vs
fixed effects, clustered SEs, first differences, within vs LSDV) and a browsable index of
the concept explainers. To launch it from Python, use
:func:`expdpy.streamlit_app.LearnApp`.
"""

from expdpy.streamlit_app._entry import run_app

run_app(module="learn")
