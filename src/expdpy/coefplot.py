"""Coefficient plots: point estimates with confidence intervals, one or more models.

A beginner-friendly alternative to reading a regression table — the figure shows each
coefficient as a point with a confidence-interval whisker and a dashed line at zero, so the
sign, magnitude and significance of every term are visible at a glance. Multiple models are
dodged side by side, which makes "how does this coefficient move as I change the
specification?" immediately legible.

The numbers come straight from each fitted model's ``tidy(alpha=...)`` (the same source
pyfixest's own ``coefplot`` uses); the figure is rebuilt in Plotly so it carries the expdpy
theme.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd
import plotly.graph_objects as go
from pyfixest.utils.dev_utils import _select_order_coefs

from expdpy._estimation import as_list
from expdpy._theme import apply_default_layout, color_for
from expdpy._types import CoefficientPlotResult

__all__ = ["prepare_coefficient_plot"]


def _coerce_models(models: Any) -> list[Any]:
    """Return a flat list of fitted models from a model, a list, or a result object.

    A list may mix bare fitted models and result objects (anything carrying ``.models``,
    e.g. a :class:`~expdpy.RegressionTableResult`); result objects are expanded in place so
    ``prepare_coefficient_plot([pooled, fe])`` works as readily as passing the raw models.
    """
    if hasattr(models, "models"):  # e.g. a RegressionTableResult
        out = list(models.models)
    elif isinstance(models, (list, tuple)):
        out = []
        for item in models:
            if hasattr(item, "models"):  # a result object nested in the list
                out.extend(item.models)
            else:
                out.append(item)
    else:
        out = [models]
    if not out:
        raise ValueError("no models to plot")
    return out


def _ci_columns(alpha: float) -> tuple[str, str]:
    """Return the lower/upper CI column names ``tidy(alpha=...)`` produces for ``alpha``."""
    return f"{alpha / 2 * 100:.1f}%", f"{(1 - alpha / 2) * 100:.1f}%"


def prepare_coefficient_plot(
    models: Any,
    *,
    keep: Sequence[str] | str | None = None,
    drop: Sequence[str] | str | None = None,
    coef_labels: Mapping[str, str] | None = None,
    model_labels: Sequence[str] | None = None,
    alpha: float = 0.05,
    joint: bool = False,
    horizontal: bool = True,
    drop_intercept: bool = True,
    title: str | None = None,
) -> CoefficientPlotResult:
    """Plot coefficient estimates with confidence intervals for one or more models.

    Parameters
    ----------
    models
        A fitted pyfixest model, a list of fitted models, or a
        :class:`~expdpy.RegressionTableResult` (its ``.models`` are used). This lets you do
        ``prepare_coefficient_plot(prepare_regression_table(...))`` directly.
    keep, drop
        Optional regular-expression patterns selecting which coefficients to show (and in
        which order). ``keep`` whitelists, ``drop`` blacklists; both use pyfixest's own
        coefficient-selection semantics.
    coef_labels
        Optional mapping from raw coefficient names to display labels.
    model_labels
        Optional legend labels, one per model (defaults to ``"Model 1"``, ``"Model 2"``…).
    alpha
        Significance level for the confidence intervals (default ``0.05`` → 95% intervals).
    joint
        If ``True``, draw simultaneous (joint) confidence bands via ``model.confint(joint=
        True)`` instead of pointwise intervals.
    horizontal
        If ``True`` (default), estimates run along the x-axis with coefficients listed down
        the y-axis (the most readable layout for many terms).
    drop_intercept
        If ``True`` (default), omit the intercept term.
    title
        Optional figure title.

    Returns
    -------
    CoefficientPlotResult
        ``df`` (tidy long frame: ``model``, ``term``, ``estimate``, ``se``, ``ci_lower``,
        ``ci_upper``) and ``fig`` (the Plotly figure).

    Examples
    --------
    Basic — plot a single fitted model's coefficients:

    ```python
    import expdpy as ex
    from expdpy.data import load_kuznets

    df = load_kuznets()
    result = ex.prepare_regression_table(
        df, dvs="gini_regional", idvs=["log_gdp_pc", "log_gdp_pc_sq", "log_gdp_pc_cu"]
    )
    ex.prepare_coefficient_plot(result).fig
    ```

    Advanced — compare several models side by side with custom labels:

    ```python
    pooled = ex.prepare_regression_table(
        df, dvs="gini_regional", idvs=["log_gdp_pc"]
    )
    fe = ex.prepare_regression_table(
        df, dvs="gini_regional", idvs=["log_gdp_pc"], feffects=["country", "year"]
    )
    ex.prepare_coefficient_plot(
        [pooled, fe],
        model_labels=["Pooled OLS", "Two-way FE"],
        keep=["log_gdp_pc"],
    ).fig
    ```
    """
    model_list = _coerce_models(models)
    keep_list = as_list(keep)
    drop_list = as_list(drop)
    if drop_intercept and "Intercept" not in drop_list:
        drop_list = [*drop_list, "Intercept"]

    if model_labels is None:
        labels = [f"Model {i + 1}" for i in range(len(model_list))]
    else:
        labels = list(model_labels)
        if len(labels) != len(model_list):
            raise ValueError(
                f"model_labels has {len(labels)} entries but there are "
                f"{len(model_list)} models"
            )

    lo_col, hi_col = _ci_columns(alpha)
    frames: list[pd.DataFrame] = []
    for label, model in zip(labels, model_list, strict=True):
        tidy = model.tidy(alpha=alpha).reset_index()
        tidy = tidy.rename(columns={tidy.columns[0]: "term"})
        if joint:
            ci = model.confint(alpha=alpha, joint=True)
            tidy[lo_col] = tidy["term"].map(ci.iloc[:, 0])
            tidy[hi_col] = tidy["term"].map(ci.iloc[:, 1])
        order = _select_order_coefs(list(tidy["term"]), keep=keep_list, drop=drop_list)
        tidy = tidy.set_index("term").loc[order].reset_index()
        frames.append(
            pd.DataFrame(
                {
                    "model": label,
                    "term": tidy["term"],
                    "estimate": tidy["Estimate"].astype(float),
                    "se": tidy["Std. Error"].astype(float),
                    "ci_lower": tidy[lo_col].astype(float),
                    "ci_upper": tidy[hi_col].astype(float),
                }
            )
        )

    df = pd.concat(frames, ignore_index=True)
    fig = _build_fig(df, labels, coef_labels, horizontal, title)
    return CoefficientPlotResult(df=df, fig=fig)


def _build_fig(
    df: pd.DataFrame,
    labels: list[str],
    coef_labels: Mapping[str, str] | None,
    horizontal: bool,
    title: str | None,
) -> go.Figure:
    """Assemble the dodged coefficient Plotly figure with whisker CIs and a zero line."""
    terms_order: list[str] = []
    for term in df["term"]:
        if term not in terms_order:
            terms_order.append(term)
    pos = {term: i for i, term in enumerate(terms_order)}

    n_models = len(labels)
    dodge_step = 0.0 if n_models == 1 else min(0.6 / n_models, 0.18)

    fig = go.Figure()
    for j, label in enumerate(labels):
        sub = df[df["model"] == label]
        if sub.empty:
            continue
        offset = (j - (n_models - 1) / 2) * dodge_step
        coords = [pos[term] + offset for term in sub["term"]]
        est = sub["estimate"]
        err = {
            "type": "data",
            "symmetric": False,
            "array": (sub["ci_upper"] - est).to_numpy(),
            "arrayminus": (est - sub["ci_lower"]).to_numpy(),
            "thickness": 1.5,
            "color": color_for(j),
        }
        marker = {"color": color_for(j), "size": 9}
        common = {"mode": "markers", "name": label, "showlegend": n_models > 1}
        if horizontal:
            fig.add_trace(
                go.Scatter(x=est, y=coords, error_x=err, marker=marker, **common)
            )
        else:
            fig.add_trace(
                go.Scatter(x=coords, y=est, error_y=err, marker=marker, **common)
            )

    ticktext = [(coef_labels.get(t, t) if coef_labels else t) for t in terms_order]
    tickvals = [pos[t] for t in terms_order]
    axis = {
        "tickmode": "array",
        "tickvals": tickvals,
        "ticktext": ticktext,
        "title": "",
    }
    if horizontal:
        fig.add_vline(x=0, line_dash="dash", line_color="rgba(0,0,0,0.4)")
        apply_default_layout(
            fig,
            yaxis={**axis, "autorange": "reversed"},
            xaxis={"title": "Coefficient estimate"},
        )
    else:
        fig.add_hline(y=0, line_dash="dash", line_color="rgba(0,0,0,0.4)")
        apply_default_layout(fig, xaxis=axis, yaxis={"title": "Coefficient estimate"})
    if title:
        fig.update_layout(title=title)
    return fig
