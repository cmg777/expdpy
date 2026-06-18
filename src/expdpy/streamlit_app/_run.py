"""Runnable entry script for ``streamlit run`` (packaged launcher target).

This file is *executed* by Streamlit's script runner (never imported as a module), so it
unconditionally invokes :func:`expdpy.streamlit_app._entry.run_app`.
"""

from expdpy.streamlit_app._entry import run_app

run_app()
