"""Streamlit Community Cloud entry point for the ExPdPy app.

Point Streamlit Community Cloud at this file, or run it locally with::

    streamlit run streamlit_app.py

It starts with the bundled-dataset picker (defaulting to Kuznets, the synthetic panel with an
N-shaped regional Kuznets curve; also Gapminder / World Bank / Russell 3000); users can also
upload their own CSV / Excel / Parquet file from the sidebar. To launch the app on your own
in-memory DataFrame instead, use :func:`expdpy.streamlit_app.ExPdPy` from Python.
"""

from expdpy.streamlit_app._entry import run_app

run_app()
