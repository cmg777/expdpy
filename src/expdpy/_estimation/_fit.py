"""Fit dispatcher: build the formula + vcov and call the right pyfixest entrypoint."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pyfixest as pf

from expdpy._estimation._formula import build_formula
from expdpy._estimation._spec import ModelSpec
from expdpy._estimation._vcov import build_vcov

__all__ = ["SSC", "fit_model"]

# Stata 'reghdfe'-consistent small-sample correction (matches lfe::felm cmethod='reghdfe').
SSC = pf.ssc(k_adj=True, G_adj=True)


def fit_model(data: pd.DataFrame, spec: ModelSpec, *, ssc: Any = SSC) -> Any:
    """Fit ``spec`` on ``data`` via ``pyfixest.feols``.

    The caller is responsible for column selection, NA handling and casting fixed effects to
    ``category`` (so behavior matches the historical implementation).

    Parameters
    ----------
    data
        The (already cleaned) estimation frame.
    spec
        The normalized model specification.
    ssc
        The small-sample-correction object (defaults to the module-level :data:`SSC`).

    Returns
    -------
    Any
        A fitted pyfixest ``Feols`` model, or a ``FixestMulti`` when ``spec`` requests
        stepwise or multiple outcomes.
    """
    fml = build_formula(spec)
    vcov, vcov_kwargs = build_vcov(spec.vcov)
    kwargs: dict[str, Any] = {"vcov": vcov, "ssc": ssc}
    if vcov_kwargs is not None:
        kwargs["vcov_kwargs"] = vcov_kwargs
    if spec.weights:
        kwargs["weights"] = spec.weights

    return pf.feols(fml, data=data, **kwargs)
