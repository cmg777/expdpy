"""Concept-explainer content.

Importing this package registers every shipped topic into the
:mod:`expdpy.pedagogy._registry`. Each submodule calls ``register_topic`` at import time.
"""

from __future__ import annotations

from expdpy.pedagogy._text import (
    causal,
    correlation,
    cre,
    learn,
    outliers,
    regression,
    tables,
)

__all__ = [
    "causal",
    "correlation",
    "cre",
    "learn",
    "outliers",
    "regression",
    "tables",
]
