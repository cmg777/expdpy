"""Normalized model + variance-covariance specifications for the estimation engine.

These small, frozen, hashable dataclasses sit between the friendly public function
signatures and pyfixest. Keeping the spec normalized in one place means the formula
builder, the vcov builder and the fit dispatcher never have to re-parse user input.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

__all__ = [
    "ModelKind",
    "ModelSpec",
    "Stepwise",
    "VCovKind",
    "VCovSpec",
    "as_list",
]

VCovKind = Literal["iid", "hetero", "HC1", "HC2", "HC3", "CRV1", "CRV3", "NW", "DK"]
ModelKind = Literal["ols", "iv", "poisson", "logit", "probit"]
Stepwise = Literal["sw", "sw0", "csw", "csw0"]


def as_list(value: Any) -> list[str]:
    """Normalize ``None`` / ``""`` / str / sequence into a flat list of non-empty strings.

    Parameters
    ----------
    value
        ``None``, an empty string, a single variable name, or a sequence of names.

    Returns
    -------
    list of str
        The non-empty names, in order.
    """
    if value is None or (isinstance(value, str) and value == ""):
        return []
    if isinstance(value, str):
        return [value]
    return [v for v in value if v]


@dataclass(frozen=True)
class VCovSpec:
    """A normalized variance-covariance (standard-error) specification.

    Parameters
    ----------
    kind
        The estimator: ``"iid"``, ``"hetero"`` (alias of ``"HC1"``), ``"HC1"``/``"HC2"``/
        ``"HC3"`` (HC2/HC3 are unavailable with fixed effects), ``"CRV1"``/``"CRV3"``
        (cluster-robust) or ``"NW"``/``"DK"`` (Newey-West / Driscoll-Kraay).
    cluster
        Cluster variable name(s); required for ``"CRV1"``/``"CRV3"``.
    time_id
        Time identifier; required for ``"NW"``/``"DK"``.
    panel_id
        Panel (unit) identifier; required for ``"NW"``/``"DK"``.
    lag
        Lag truncation for ``"NW"``/``"DK"`` (pyfixest picks a default when ``None``).
    """

    kind: VCovKind = "iid"
    cluster: tuple[str, ...] = ()
    time_id: str | None = None
    panel_id: str | None = None
    lag: int | None = None


@dataclass(frozen=True)
class ModelSpec:
    """A normalized specification of a single (or stepwise/multi-outcome) model.

    Parameters
    ----------
    dv
        Dependent-variable name(s). More than one name builds a multi-outcome formula.
    idvs
        Independent (exogenous) regressor names.
    feffects
        Fixed-effect variable names absorbed by pyfixest.
    endog
        Endogenous regressors (instrumental-variables models only).
    instruments
        Excluded instruments (instrumental-variables models only).
    model
        Estimator family: ``"ols"``, ``"iv"``, ``"poisson"``, ``"logit"`` or ``"probit"``.
    stepwise
        Optional stepwise wrapper (``"sw"``, ``"sw0"``, ``"csw"`` or ``"csw0"``) applied to
        ``idvs`` to estimate a sequence of nested models in one call.
    vcov
        The variance-covariance specification.
    weights
        Optional weights column name.
    offset
        Optional offset column name (Poisson models).
    """

    dv: tuple[str, ...]
    idvs: tuple[str, ...]
    feffects: tuple[str, ...] = ()
    endog: tuple[str, ...] = ()
    instruments: tuple[str, ...] = ()
    model: ModelKind = "ols"
    stepwise: Stepwise | None = None
    vcov: VCovSpec = VCovSpec()
    weights: str | None = None
    offset: str | None = None
