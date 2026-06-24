"""Post-estimation helpers: visualize fixed effects, predict, and joint-test coefficients.

These operate on a fitted model (or any expdpy result that carries ``.models``), so they
compose with both :func:`expdpy.analyze_regression_table` and
:func:`expdpy.analyze_estimation`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from expdpy._estimation import as_list, first_model
from expdpy._theme import apply_default_layout, color_for
from expdpy._types import FixefPlotResult, JointTestResult, PredictionResult

__all__ = [
    "analyze_fixef_plot",
    "analyze_joint_test",
    "analyze_predictions",
]


def _strip_fe(key: str) -> str:
    """Turn pyfixest's ``"C(firm)"`` fixed-effect key into a plain name (``"firm"``)."""
    k = str(key)
    return k[2:-1] if k.startswith("C(") and k.endswith(")") else k


def analyze_fixef_plot(
    result_or_model: Any,
    *,
    fixef: str | None = None,
    top_n: int | None = 30,
    title: str | None = None,
) -> FixefPlotResult:
    """Plot the estimated group intercepts (fixed effects) of a model, ranked by value.

    Parameters
    ----------
    result_or_model
        A fitted model or a result object carrying ``.models``.
    fixef
        Which fixed-effect dimension to plot (e.g. ``"country"``). Defaults to the first.
    top_n
        Show at most this many levels; when there are more, the most extreme (lowest and
        highest) are kept. ``None`` shows every level.
    title
        Optional figure title.

    Returns
    -------
    FixefPlotResult
        ``df`` (``fixef``, ``level``, ``value``) and the Plotly figure.

    Examples
    --------
    Basic — plot the country fixed effects of a Kuznets-curve model:

    ```python
    import expdpy as ex
    from expdpy.data import load_kuznets, load_kuznets_data_def

    df = ex.set_labels(load_kuznets(), load_kuznets_data_def(), set_panel=True)
    model = ex.analyze_regression_table(
        df,
        dvs="gini_regional",
        idvs=["log_gdp_pc", "log_gdp_pc_sq", "log_gdp_pc_cu"],
        feffects=["country"],
    )
    ex.analyze_fixef_plot(model).fig
    ```

    Advanced — show only the most extreme levels with a custom title:

    ```python
    import expdpy as ex
    from expdpy.data import load_kuznets, load_kuznets_data_def

    df = ex.set_labels(load_kuznets(), load_kuznets_data_def(), set_panel=True)
    model = ex.analyze_regression_table(
        df,
        dvs="gini_regional",
        idvs=["log_gdp_pc", "log_gdp_pc_sq", "log_gdp_pc_cu"],
        feffects=["country"],
    )
    ex.analyze_fixef_plot(
        model, fixef="country", top_n=10, title="Country intercepts"
    ).fig
    ```
    """
    model = first_model(result_or_model)
    try:
        effects = model.fixef()
    except ValueError as exc:  # pyfixest raises when the model has no fixed effects
        raise ValueError("the model has no fixed effects to plot") from exc
    if not effects:
        raise ValueError("the model has no fixed effects to plot")

    keys = list(effects.keys())
    if fixef is None:
        key = keys[0]
    else:
        match = [k for k in keys if _strip_fe(k) == str(fixef) or k == str(fixef)]
        if not match:
            raise KeyError(
                f"fixed effect {fixef!r} not found; available: "
                f"{[_strip_fe(k) for k in keys]}"
            )
        key = match[0]

    items = sorted(effects[key].items(), key=lambda kv: kv[1])
    truncated = False
    if top_n is not None and len(items) > top_n:
        truncated = True
        half = max(1, top_n // 2)
        items = items[:half] + items[-half:]

    label = _strip_fe(key)
    df = pd.DataFrame(
        {
            "fixef": label,
            "level": [str(k) for k, _ in items],
            "value": [float(v) for _, v in items],
        }
    )

    fig = go.Figure(
        go.Bar(
            x=df["value"],
            y=df["level"],
            orientation="h",
            marker={"color": color_for(0)},
            name=label,
        )
    )
    fig.add_vline(x=0, line_dash="dash", line_color="rgba(0,0,0,0.4)")
    apply_default_layout(
        fig,
        xaxis={"title": f"Estimated {label} fixed effect"},
        yaxis={"title": label, "type": "category"},
    )
    if truncated:
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=0.98,
            y=0.02,
            xanchor="right",
            yanchor="bottom",
            showarrow=False,
            text=f"showing {len(items)} most extreme levels",
            font={"size": 12, "color": "#666"},
        )
    if title:
        fig.update_layout(title=title)
    return FixefPlotResult(df=df, fig=fig)


def analyze_predictions(
    result_or_model: Any, newdata: pd.DataFrame | None = None
) -> PredictionResult:
    """Return fitted values from a model (and residuals/actuals on the estimation sample).

    Parameters
    ----------
    result_or_model
        A fitted model or a result object carrying ``.models``.
    newdata
        Optional data frame to predict on. When omitted, predictions are for the estimation
        sample and the frame also includes the ``actual`` outcome and the ``residual``.

    Returns
    -------
    PredictionResult
        ``df`` with a ``predicted`` column (plus ``actual`` and ``residual`` in-sample).

    Examples
    --------
    Basic — in-sample fitted values, actuals, and residuals:

    ```python
    import expdpy as ex
    from expdpy.data import load_kuznets, load_kuznets_data_def

    df = ex.set_labels(load_kuznets(), load_kuznets_data_def(), set_panel=True)
    model = ex.analyze_regression_table(
        df,
        dvs="gini_regional",
        idvs=["log_gdp_pc", "log_gdp_pc_sq", "log_gdp_pc_cu"],
        feffects=["country"],
    )
    ex.analyze_predictions(model).df.head()
    ```

    Advanced — predict on new data (here a fresh slice of the panel):

    ```python
    import expdpy as ex
    from expdpy.data import load_kuznets, load_kuznets_data_def

    df = ex.set_labels(load_kuznets(), load_kuznets_data_def(), set_panel=True)
    model = ex.analyze_regression_table(
        df,
        dvs="gini_regional",
        idvs=["log_gdp_pc", "log_gdp_pc_sq", "log_gdp_pc_cu"],
        feffects=["country"],
    )
    newdata = df[df["year"] == df["year"].max()]
    ex.analyze_predictions(model, newdata=newdata).df.head()
    ```
    """
    model = first_model(result_or_model)
    if newdata is not None:
        pred = np.asarray(model.predict(newdata=newdata), dtype=float)
        return PredictionResult(df=pd.DataFrame({"predicted": pred}))
    pred = np.asarray(model.predict(), dtype=float)
    resid = np.asarray(model.resid(), dtype=float)
    return PredictionResult(
        df=pd.DataFrame({"actual": pred + resid, "predicted": pred, "residual": resid})
    )


def analyze_joint_test(
    result_or_model: Any,
    hypotheses: Sequence[str] | str | None = None,
    *,
    distribution: str = "F",
) -> JointTestResult:
    """Run a Wald joint-significance test that a set of coefficients are all zero.

    Parameters
    ----------
    result_or_model
        A fitted model or a result object carrying ``.models``.
    hypotheses
        Coefficient name(s) to test jointly. ``None`` tests all coefficients at once.
    distribution
        Reference distribution: ``"F"`` (default) or ``"chi2"``.

    Returns
    -------
    JointTestResult
        ``statistic``, ``p_value``, the tested ``hypotheses`` and the ``distribution``.

    Examples
    --------
    Basic — jointly test the two nonlinear Kuznets terms:

    ```python
    import expdpy as ex
    from expdpy.data import load_kuznets, load_kuznets_data_def

    df = ex.set_labels(load_kuznets(), load_kuznets_data_def(), set_panel=True)
    model = ex.analyze_regression_table(
        df,
        dvs="gini_regional",
        idvs=["log_gdp_pc", "log_gdp_pc_sq", "log_gdp_pc_cu"],
        feffects=["country"],
    )
    test = ex.analyze_joint_test(model, ["log_gdp_pc_sq", "log_gdp_pc_cu"])
    test.statistic, test.p_value
    ```

    Advanced — test all slope coefficients at once against a chi-square reference:

    ```python
    import expdpy as ex
    from expdpy.data import load_kuznets, load_kuznets_data_def

    df = ex.set_labels(load_kuznets(), load_kuznets_data_def(), set_panel=True)
    model = ex.analyze_regression_table(
        df,
        dvs="gini_regional",
        idvs=["log_gdp_pc", "log_gdp_pc_sq", "log_gdp_pc_cu"],
        feffects=["country"],
    )
    ex.analyze_joint_test(model, distribution="chi2").p_value
    ```
    """
    model = first_model(result_or_model)
    coefs = [str(c) for c in model.coef().index]

    if hypotheses is None:
        result = model.wald_test(distribution=distribution)
        names: tuple[str, ...] = tuple(coefs)
    else:
        names = tuple(as_list(hypotheses))
        missing = [n for n in names if n not in coefs]
        if missing:
            raise KeyError(f"coefficients not in model: {missing}")
        restriction = np.zeros((len(names), len(coefs)))
        for i, name in enumerate(names):
            restriction[i, coefs.index(name)] = 1.0
        result = model.wald_test(
            R=restriction, q=np.zeros(len(names)), distribution=distribution
        )

    return JointTestResult(
        statistic=float(result["statistic"]),
        p_value=float(result["pvalue"]),
        hypotheses=names,
        distribution=distribution,
    )
