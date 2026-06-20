"""Pure builder of the pyfixest formula string from a :class:`ModelSpec`."""

from __future__ import annotations

from expdpy._estimation._spec import ModelSpec

__all__ = ["build_formula"]


def build_formula(spec: ModelSpec) -> str:
    """Return the pyfixest formula string for ``spec``.

    Handles plain OLS (``"dv ~ x1 + x2"``), fixed effects (``"| f1 + f2"``), stepwise
    sequences (``"csw(x1, x2, x3)"``) and multiple outcomes (``"y1 + y2 ~ ..."``).

    Parameters
    ----------
    spec
        The normalized model specification.

    Returns
    -------
    str
        A formula string accepted by ``pyfixest.feols``.

    Examples
    --------
    >>> from expdpy._estimation import ModelSpec, build_formula
    >>> build_formula(ModelSpec(dv=("y",), idvs=("x1", "x2"), feffects=("firm",)))
    'y ~ x1 + x2 | firm'
    """
    lhs = " + ".join(spec.dv)
    if spec.stepwise and spec.idvs:
        rhs = f"{spec.stepwise}({', '.join(spec.idvs)})"
    elif spec.idvs:
        rhs = " + ".join(spec.idvs)
    else:
        rhs = "1"
    fml = f"{lhs} ~ {rhs}"
    if spec.feffects:
        fml += " | " + " + ".join(spec.feffects)
    return fml
