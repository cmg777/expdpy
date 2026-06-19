"""expdpy's pedagogy layer: concept explainers + plain-language result interpretation.

Two complementary pieces:

* **Explainers** — data-independent teaching content (`explain("fixed_effects")`,
  `list_topics()`), reusable in notebooks, docs and both apps.
* **Interpretation** — data-dependent prose attached to result objects via the
  :class:`Interpretable` mixin (``result.interpret()`` / ``result.explain()``).

Importing this package also registers the shipped explainer topics (via :mod:`._text`).
"""

from __future__ import annotations

from expdpy.pedagogy import _text  # noqa: F401  (import registers the explainer topics)
from expdpy.pedagogy._interpret import (
    interpret_correlation,
    interpret_descriptive,
    interpret_estimation,
    interpret_event_study,
    interpret_fwl,
    interpret_regression,
    interpret_sandbox,
    interpret_trend,
)
from expdpy.pedagogy._mixin import Interpretable
from expdpy.pedagogy._registry import Explainer, explain, list_topics, register_topic

__all__ = [
    "Explainer",
    "Interpretable",
    "explain",
    "interpret_correlation",
    "interpret_descriptive",
    "interpret_estimation",
    "interpret_event_study",
    "interpret_fwl",
    "interpret_regression",
    "interpret_sandbox",
    "interpret_trend",
    "list_topics",
    "register_topic",
]
