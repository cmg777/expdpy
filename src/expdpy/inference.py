"""Robust inference: randomization inference and the wild cluster bootstrap.

Both are alternatives to relying on large-sample cluster-robust standard errors when the
number of clusters is small — a common situation in panel data and a frequent source of
over-confident p-values.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

from expdpy._estimation import first_model
from expdpy._types import RobustInferenceResult

__all__ = ["analyze_robust_inference"]


def analyze_robust_inference(
    result_or_model: Any,
    param: str,
    *,
    method: Literal["ritest", "wildboot"] = "ritest",
    reps: int = 1000,
    cluster: str | None = None,
    seed: int = 0,
) -> RobustInferenceResult:
    """Run robust inference on one coefficient via randomization inference or wild bootstrap.

    Parameters
    ----------
    result_or_model
        A fitted model or a result object carrying ``.models``.
    param
        The coefficient name to test (``H0: param = 0``).
    method
        ``"ritest"`` (randomization inference, native to pyfixest) or ``"wildboot"`` (wild
        cluster bootstrap, which requires the optional ``wildboottest`` package).
    reps
        Number of resamples / bootstrap replications.
    cluster
        Optional cluster variable for the resampling.
    seed
        Seed for reproducibility.

    Returns
    -------
    RobustInferenceResult
        ``method``, ``param``, ``estimate``, ``p_value``, ``conf_int``, ``reps`` and the
        underlying ``raw`` pyfixest output.

    Examples
    --------
    Basic — fit a regression (the data dictionary supplies the readable labels) and test
    the slope on trade openness via randomization inference:

    ```python
    import expdpy as ex
    from expdpy.data import load_kuznets, load_kuznets_data_def

    df = ex.set_labels(load_kuznets(), load_kuznets_data_def(), set_panel=True)
    model = ex.analyze_regression_table(
        df,
        dvs="gini_regional",
        idvs=["log_gdp_pc", "trade_share"],
    )
    result = ex.analyze_robust_inference(model, "trade_share", reps=200, seed=0)
    result.p_value
    ```

    Advanced — cluster the randomization inference by country (resampling permutes
    treatment within clusters; the cluster column must be numeric, so encode the
    country labels as integer codes first), then read the estimate and the raw
    pyfixest output:

    ```python
    import expdpy as ex
    from expdpy.data import load_kuznets, load_kuznets_data_def

    df = ex.set_labels(load_kuznets(), load_kuznets_data_def(), set_panel=True)
    df["country_id"] = df["country"].factorize()[0]
    model = ex.analyze_regression_table(
        df,
        dvs="gini_regional",
        idvs=["log_gdp_pc", "trade_share"],
        clusters=["country_id"],
    )
    result = ex.analyze_robust_inference(
        model, "trade_share", cluster="country_id", reps=200, seed=0
    )
    result.estimate
    result.raw
    ```
    """
    model = first_model(result_or_model)

    if method == "ritest":
        # pyfixest's numba-compiled resampler returns a float array in its clustered branch
        # but the resampvar's own dtype otherwise; an integer/bool resampvar therefore makes
        # numba fail to unify the two return types (a TypingError that only surfaces where
        # numba is installed, e.g. Google Colab). Casting the column to float is
        # value-preserving (0/1 -> 0.0/1.0, identical estimate and p-value). Restore the
        # caller's column afterwards so the model object is not left mutated.
        col = param.split("=", 1)[0].strip()
        data = getattr(model, "_data", None)
        cast_back: pd.Series | None = None
        if (
            isinstance(data, pd.DataFrame)
            and col in data.columns
            and not pd.api.types.is_float_dtype(data[col])
        ):
            cast_back = data[col]
            data[col] = data[col].astype(float)
        try:
            series = model.ritest(
                resampvar=param,
                reps=reps,
                cluster=cluster,
                rng=np.random.default_rng(seed),
            )
        finally:
            if cast_back is not None and isinstance(data, pd.DataFrame):
                data[col] = cast_back
        return RobustInferenceResult(
            method="ritest",
            param=param,
            estimate=float(series["Estimate"]),
            p_value=float(series["Pr(>|t|)"]),
            conf_int=(
                float(series["2.5% (Pr(>|t|))"]),
                float(series["97.5% (Pr(>|t|))"]),
            ),
            reps=reps,
            raw=series,
        )

    if method == "wildboot":
        try:
            import wildboottest  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "the wild cluster bootstrap requires the optional 'wildboottest' package; "
                "install it with:  pip install wildboottest"
            ) from exc
        series = model.wildboottest(reps=reps, param=param, cluster=cluster, seed=seed)
        values = {str(k): v for k, v in series.items()}
        return RobustInferenceResult(
            method="wildboot",
            param=param,
            estimate=float(values.get("Estimate", float("nan"))),
            p_value=float(values.get("Pr(>|t|)", float("nan"))),
            conf_int=(float("nan"), float("nan")),
            reps=reps,
            raw=series,
        )

    raise ValueError(f"unknown method {method!r}; use 'ritest' or 'wildboot'")
