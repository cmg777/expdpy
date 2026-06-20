"""Streamlit Community Cloud entry point for the ExPdPy **Analyze** app.

Point Streamlit Community Cloud at this file, or run it locally with::

    streamlit run app_analyze.py

It exposes the panel estimators — regression with fixed effects and clustered SEs, the
Frisch-Waugh-Lovell plot, panel models (pooled / between / fixed / random effects), the
Hausman test, correlated random effects (Mundlak) and event studies. To launch on your own
in-memory DataFrame, use :func:`expdpy.streamlit_app.AnalyzeApp` from Python.
"""

from expdpy.streamlit_app._entry import run_app

run_app(module="analyze")
