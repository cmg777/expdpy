"""Helpers for accepting a fitted model, a list of them, or an expdpy result object."""

from __future__ import annotations

from typing import Any

__all__ = ["coerce_models", "first_model"]


def coerce_models(obj: Any) -> list[Any]:
    """Return a flat list of fitted models from a model, a list, or a result object.

    Accepts a single fitted pyfixest model, a ``list``/``tuple`` of them, or any expdpy
    result object that carries a ``.models`` list (e.g. ``RegressionTableResult`` /
    ``EstimationResult``).
    """
    if hasattr(obj, "models"):
        out = list(obj.models)
    elif isinstance(obj, (list, tuple)):
        out = list(obj)
    else:
        out = [obj]
    if not out:
        raise ValueError("no models found")
    return out


def first_model(obj: Any) -> Any:
    """Return the first fitted model from a model, a list, or a result object."""
    return coerce_models(obj)[0]
