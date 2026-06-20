"""Streamlit Community Cloud entry point for the ExPdPy **Explore** app.

Point Streamlit Community Cloud at this file, or run it locally with::

    streamlit run app_explore.py

It starts with the bundled-dataset picker (Kuznets / Gapminder); users can also upload their
own CSV / Excel / Parquet file from the sidebar. To launch on your own in-memory DataFrame,
use :func:`expdpy.streamlit_app.ExploreApp` from Python.
"""

from expdpy.streamlit_app._entry import run_app

run_app(module="explore")
