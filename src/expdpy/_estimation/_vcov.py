"""Pure builder of pyfixest's ``(vcov, vcov_kwargs)`` pair from a :class:`VCovSpec`."""

from __future__ import annotations

from typing import Any

from expdpy._estimation._spec import VCovSpec

__all__ = ["build_vcov"]


def build_vcov(spec: VCovSpec) -> tuple[Any, dict[str, Any] | None]:
    """Translate a :class:`VCovSpec` into pyfixest's ``vcov`` / ``vcov_kwargs`` arguments.

    Parameters
    ----------
    spec
        The normalized variance-covariance specification.

    Returns
    -------
    tuple
        ``(vcov, vcov_kwargs)``. ``vcov`` is a string (``"iid"``, ``"hetero"``, ``"HC1"``…)
        or a ``{"CRV1"/"CRV3": "a + b"}`` dict; ``vcov_kwargs`` is ``None`` except for the
        serial-correlation-robust estimators (``"NW"``/``"DK"``), which need ``time_id`` /
        ``panel_id`` (and optionally ``lag``).

    Examples
    --------
    >>> from expdpy._estimation import VCovSpec, build_vcov
    >>> build_vcov(VCovSpec(kind="CRV1", cluster=("firm", "year")))
    ({'CRV1': 'firm + year'}, None)
    >>> build_vcov(VCovSpec(kind="iid"))
    ('iid', None)
    """
    kind = spec.kind
    if kind in ("CRV1", "CRV3"):
        if not spec.cluster:
            raise ValueError(
                f"{kind} standard errors require at least one cluster variable"
            )
        return {kind: " + ".join(spec.cluster)}, None
    if kind in ("NW", "DK"):
        if spec.time_id is None or spec.panel_id is None:
            raise ValueError(
                f"{kind} standard errors require both 'time_id' and 'panel_id'"
            )
        kwargs: dict[str, Any] = {"time_id": spec.time_id, "panel_id": spec.panel_id}
        if spec.lag is not None:
            kwargs["lag"] = spec.lag
        return kind, kwargs
    return kind, None
