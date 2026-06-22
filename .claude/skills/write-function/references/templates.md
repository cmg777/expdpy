# Templates — annotated skeletons

Adapt these to the specific function. They are distilled from the exemplar
`src/expdpy/convergence.py` and `tests/test_convergence.py` — read those in full for a complete
worked example. Keep source ASCII (write `rho`/`lambda`, use `-` not the Unicode minus).

## Function module — `src/expdpy/<module>.py`

```python
"""<One-line summary>. <2-4 lines: what it estimates, the method, and any key contract
(e.g. 'the variable is used as supplied; no transformation is applied').>"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pyfixest as pf
from pandas.api import types as pdt

from expdpy._labels import resolve_label
from expdpy._panel import resolve_panel
from expdpy._theme import apply_default_layout, color_for
from expdpy._types import <Result>
from expdpy._validation import ensure_dataframe
from expdpy.regression import _SSC, _as_list  # shared small-sample correction + list-normalize

__all__ = ["<fn>"]


def _metric(...) -> float:
    """Pure, deterministic math helper with explicit NaN/domain guards.

    Keep the math transparent and unit-testable in isolation. Return ``nan`` (not an
    exception) for out-of-domain inputs so callers can decide how to present them.
    """
    if <out of domain>:
        return float("nan")
    return <closed-form expression>


def <fn>(
    df: pd.DataFrame,
    var: str,
    controls: Sequence[str] | str | None = None,
    *,
    entity: str | None = None,
    time: str | None = None,
    vcov: Literal["iid", "hetero"] = "hetero",
    title: str | None = None,
) -> <Result>:
    """<One-line imperative summary>.

    <Paragraph: the estimand and the method, in plain terms. State whether the framing is
    associational. Name the key assumptions.>

    Parameters
    ----------
    df
        <Panel/cross-section> data frame.
    var
        <Numeric variable; state the expected scale/units and that it is used as-is.>
    controls
        Optional control name(s); <how they enter the model>.
    entity, time
        Panel identifiers. Default to those declared via :func:`expdpy.set_panel`.
    vcov
        Standard-error type: ``"hetero"`` (HC1, default) or ``"iid"``. Does not change the
        point estimate.
    title
        Title for the primary figure.

    Returns
    -------
    <Result>
        <Enumerate every field the result exposes: ``df``, ``fig``/``gt``, named scalars,
        and what ``.interpret()`` reports.>

    Notes
    -----
    The math, stated transparently. For example, the estimand is

    .. math:: <formula>

    with assumptions <...>. <Cite the canonical reference.>

    Examples
    --------
    ```python
    import expdpy as ex
    res = ex.<fn>(df, "<var>", entity="<entity>", time="<time>")
    res.fig
    res.gt
    ```
    """
    # --- 1. Defensive validation, before any computation ---
    df = ensure_dataframe(df)
    controls = _as_list(controls)
    entity, time = resolve_panel(df, entity, time, require_entity=True, require_time=True)
    assert entity is not None and time is not None  # guaranteed by require_*

    missing = [c for c in [var, *controls] if c not in df.columns]
    if missing:
        raise KeyError(f"columns not found in df: {missing}")
    if not pdt.is_numeric_dtype(df[var]):
        raise TypeError(f"var {var!r} must be numeric")
    # ... per-control numeric checks, complete-case counts, etc.

    work = df[[...]].dropna(...)
    if len(work) < <min>:
        raise ValueError(f"too few complete-case rows ({len(work)}); need >= <min>")
    x = work[var].to_numpy(dtype=float)
    if np.ptp(x) <= 1e-10 * (1.0 + float(np.abs(x).max())):
        raise ValueError(f"{var!r} has (near) zero variance; the estimate is not identified")

    # --- 2. Estimate through the engine (never hand-rolled) ---
    model = pf.feols(f"<dv> ~ <rhs>", data=work, vcov=vcov, ssc=_SSC)
    coef = float(model.coef()["<term>"])
    se = float(model.se()["<term>"])
    # ... derive metrics via the pure helpers; track NaNs ...

    # --- 3. Build the figure(s)/table(s) with the shared theme ---
    fig = _build_fig(..., title=title)

    # --- 4. Return the frozen result dataclass ---
    return <Result>(df=work, fig=fig, ..., notes=tuple(notes))


def _build_fig(...) -> go.Figure:
    """Assemble the themed Plotly figure (markers with entity hover + annotation box)."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=..., y=..., mode="markers",
        customdata=<entity values>,
        hovertemplate="%{customdata}<br>x=%{x:.4g}, y=%{y:.4g}<extra></extra>",
    ))
    fig.add_annotation(xref="paper", yref="paper", x=0.02, y=0.98, xanchor="left",
                       yanchor="top", showarrow=False, align="left",
                       bgcolor="rgba(255,255,255,0.7)", bordercolor="rgba(0,0,0,0.2)",
                       borderwidth=1, text="<br>".join([f"coef = {coef:.4g}", ...]))
    apply_default_layout(fig, xaxis={"title": ...}, yaxis={"title": ...})
    if title:
        fig.update_layout(title=title)
    return fig
```

## Result dataclass — in `src/expdpy/_types.py`

```python
@dataclass(frozen=True)
class <Result>(Interpretable):
    """Result of :func:`expdpy.<fn>`. <Describe each field.>"""

    df: pd.DataFrame
    fig: go.Figure
    gt: GT | None = None
    # ... named scalar statistics ...
    notes: tuple[str, ...] = ()

    def interpret(self, *, lang: str = "en") -> str:
        return interpret_<topic>(self, lang=lang)

    def explain(self, *, lang: str = "en") -> Explainer:
        return _explain("<topic>", lang=lang)

    def tidy(self) -> pd.DataFrame:   # if meaningful
        return self.df

    def glance(self) -> pd.DataFrame:  # one-row scalar summary, if meaningful
        import pandas as pd
        return pd.DataFrame([{ "coef": self.coef, ... }])
```

## Test suite — `tests/test_<module>.py`

```python
"""Tests for :func:`expdpy.<fn>`.

The verification backbone is <closed-form / DGP> whose true result is known: <state the
identity, e.g. 'beta = (rho**T - 1)/T'>. Simulate it, run the function, assert recovery.
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

import expdpy as ex
from expdpy import <Result>
from expdpy.<module> import _metric  # private helpers are import-tested directly

pytestmark = pytest.mark.panel  # if a panel function


def _dgp(*, n=120, rho=0.9, seed=0) -> pd.DataFrame:
    """Simulate data whose true parameter is known in closed form."""
    rng = np.random.default_rng(seed)
    ...
    return pd.DataFrame(...)


def _closed_form(rho, T) -> float:
    return <expression>


# 1. Pure-helper / closed-form unit tests
def test_metric_formula_and_domain_guards():
    assert _metric(<known input>) == pytest.approx(<known output>, abs=1e-12)
    assert math.isnan(_metric(<out-of-domain>))


# 2. Mathematical-validity: recovery of the known baseline (the key test)
def test_recovers_known_parameter():
    df = _dgp(rho=0.9, seed=1)
    res = ex.<fn>(df, "x", entity="country", time="year")
    assert res.coef == pytest.approx(_closed_form(0.9, res.horizon), abs=2e-3)


# 3. Result-surface / expected-use
def test_result_surface():
    res = ex.<fn>(_dgp(seed=2), "x", entity="country", time="year")
    assert isinstance(res, <Result>) and isinstance(res.fig, go.Figure)
    markers = next(t for t in res.fig.data if t.mode == "markers")
    assert markers.customdata is not None and len(markers.customdata) == res.n_obs
    txt = res.interpret()
    assert "causes" not in txt.lower() and "effect of" not in txt.lower()
    assert "association" in txt.lower()  # the shared closing note
    assert res.explain().topic == "<topic>"


# 4. Edge cases - assert the right exception type/message
def test_missing_time_raises():
    with pytest.raises(ValueError):
        ex.<fn>(_dgp(seed=3), "x", entity="country")  # no time id

def test_non_numeric_var_raises():
    with pytest.raises(TypeError):
        ex.<fn>(_dgp(seed=4), "country", entity="country", time="year")

def test_zero_variance_raises():
    df = _dgp(seed=5)
    df.loc[df["year"] == df["year"].min(), "x"] = 10.0  # kill variance in the regressor
    with pytest.raises(ValueError, match="not identified"):
        ex.<fn>(df, "x", entity="country", time="year")
```
