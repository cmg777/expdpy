"""Shared estimation engine for expdpy's regression-style functions.

This private package holds the building blocks that every estimator plugs into:

* :mod:`._spec` — the normalized :class:`ModelSpec` / :class:`VCovSpec` dataclasses,
* :mod:`._formula` — a pure pyfixest-formula builder,
* :mod:`._vcov` — a pure ``(vcov, vcov_kwargs)`` builder,
* :mod:`._fit` — dispatch to ``feols`` / ``fepois`` / ``feglm`` (+ the SSC default),
* :mod:`._tidy` — the tidy-coefficient-frame helper,
* :mod:`._capture` — a stdout-capture context manager.

``expdpy.regression`` is a thin adapter over this engine; keeping the engine separate
lets future estimators (IV, Poisson, GLM, model comparison) reuse one tested core.
"""

from __future__ import annotations

from expdpy._estimation._capture import capture_stdout
from expdpy._estimation._fit import SSC, fit_model
from expdpy._estimation._formula import build_formula
from expdpy._estimation._results import coerce_models, first_model
from expdpy._estimation._spec import ModelSpec, VCovSpec, as_list
from expdpy._estimation._tidy import tidy_model
from expdpy._estimation._vcov import build_vcov

__all__ = [
    "SSC",
    "ModelSpec",
    "VCovSpec",
    "as_list",
    "build_formula",
    "build_vcov",
    "capture_stdout",
    "coerce_models",
    "first_model",
    "fit_model",
    "tidy_model",
]
