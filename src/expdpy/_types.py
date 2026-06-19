"""Frozen result dataclasses returned by the ``prepare_*`` functions.

The R package returns ``list(df = ..., plot = ...)`` or ``list(df = ..., kable_ret = ...)``
objects. In Python we use small, typed, immutable dataclasses that expose the underlying
``pandas.DataFrame`` alongside the rendered object (a Plotly ``Figure``, a Great Tables
``GT``, or a pyfixest ``etable`` result).

Many result types also mix in :class:`expdpy.pedagogy.Interpretable`, which adds a small
broom-style surface: ``interpret()`` (plain-language reading of the result), ``explain()``
(the concept explainer for the method) and, where meaningful, ``tidy()`` / ``glance()``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from expdpy.pedagogy import Interpretable
from expdpy.pedagogy import explain as _explain
from expdpy.pedagogy._interpret import (
    interpret_correlation,
    interpret_descriptive,
    interpret_estimation,
    interpret_event_study,
    interpret_fwl,
    interpret_regression,
    interpret_sandbox,
    interpret_trend,
)

if TYPE_CHECKING:
    import pandas as pd
    import plotly.graph_objects as go
    from great_tables import GT

    from expdpy.pedagogy import Explainer

__all__ = [
    "BarChartResult",
    "ByGroupBarGraphResult",
    "ByGroupTrendGraphResult",
    "CoefficientPlotResult",
    "CorrelationGraphResult",
    "CorrelationTableResult",
    "DescriptiveTableResult",
    "EstimationResult",
    "EventStudyResult",
    "ExtObsTableResult",
    "FWLPlotResult",
    "FixefPlotResult",
    "HausmanTestResult",
    "HistogramResult",
    "JointTestResult",
    "PanelViewResult",
    "PredictionResult",
    "QuantileTrendGraphResult",
    "RegressionTableResult",
    "RobustInferenceResult",
    "SandboxResult",
    "TrendGraphResult",
]


@dataclass(frozen=True)
class DescriptiveTableResult(Interpretable):
    """Result of :func:`expdpy.prepare_descriptive_table`."""

    df: pd.DataFrame
    gt: GT

    def interpret(self, *, lang: str = "en") -> str:
        """Plain-language summary of central tendency, spread and skew per variable."""
        return interpret_descriptive(self, lang=lang)

    def explain(self, *, lang: str = "en") -> Explainer:
        """Concept explainer for descriptive statistics."""
        return _explain("descriptive_stats", lang=lang)

    def tidy(self) -> pd.DataFrame:
        """Return the summary frame with the variable index promoted to a column."""
        return self.df.rename_axis("variable").reset_index()


@dataclass(frozen=True)
class CorrelationTableResult(Interpretable):
    """Result of :func:`expdpy.prepare_correlation_table`.

    ``df_corr`` holds Pearson correlations above and Spearman correlations below the
    diagonal; ``df_prob`` the matching p-values; ``df_n`` the pairwise observation counts.
    """

    df_corr: pd.DataFrame
    df_prob: pd.DataFrame
    df_n: pd.DataFrame
    gt: GT

    def interpret(self, *, lang: str = "en") -> str:
        """Plain-language reading of the strongest pair and Pearson-vs-Spearman divergence."""
        return interpret_correlation(self, lang=lang)

    def explain(self, *, lang: str = "en") -> Explainer:
        """Concept explainer for correlation (Pearson; see-also Spearman and causation)."""
        return _explain("pearson", lang=lang)

    def tidy(self) -> pd.DataFrame:
        """Long-format ``(var1, var2, correlation)`` frame from the correlation matrix."""
        long = self.df_corr.stack().rename("correlation").reset_index()
        long.columns = ["var1", "var2", "correlation"]
        return long


@dataclass(frozen=True)
class CorrelationGraphResult:
    """Result of :func:`expdpy.prepare_correlation_graph`."""

    df_corr: pd.DataFrame
    df_prob: pd.DataFrame
    df_n: pd.DataFrame
    fig: go.Figure


@dataclass(frozen=True)
class ExtObsTableResult:
    """Result of :func:`expdpy.prepare_ext_obs_table`."""

    df: pd.DataFrame
    gt: GT


@dataclass(frozen=True)
class TrendGraphResult(Interpretable):
    """Result of :func:`expdpy.prepare_trend_graph`."""

    df: pd.DataFrame
    fig: go.Figure

    def interpret(self, *, lang: str = "en") -> str:
        """Plain-language reading of the direction of change for each series."""
        return interpret_trend(self, lang=lang)

    def explain(self, *, lang: str = "en") -> Explainer:
        """Concept explainer for time trends."""
        return _explain("time_trends", lang=lang)


@dataclass(frozen=True)
class QuantileTrendGraphResult:
    """Result of :func:`expdpy.prepare_quantile_trend_graph`."""

    df: pd.DataFrame
    fig: go.Figure


@dataclass(frozen=True)
class ByGroupBarGraphResult:
    """Result of :func:`expdpy.prepare_by_group_bar_graph`."""

    df: pd.DataFrame
    fig: go.Figure


@dataclass(frozen=True)
class ByGroupTrendGraphResult:
    """Result of :func:`expdpy.prepare_by_group_trend_graph`."""

    df: pd.DataFrame
    fig: go.Figure


@dataclass(frozen=True)
class HistogramResult:
    """Result of :func:`expdpy.prepare_histogram`."""

    df: pd.DataFrame
    fig: go.Figure


@dataclass(frozen=True)
class BarChartResult:
    """Result of :func:`expdpy.prepare_bar_chart`."""

    df: pd.DataFrame
    fig: go.Figure


@dataclass(frozen=True)
class CoefficientPlotResult:
    """Result of :func:`expdpy.prepare_coefficient_plot`.

    ``df`` is a tidy long frame with columns ``model``, ``term``, ``estimate``, ``se``,
    ``ci_lower`` and ``ci_upper``; ``fig`` is the Plotly coefficient plot.
    """

    df: pd.DataFrame
    fig: go.Figure


@dataclass(frozen=True)
class RegressionTableResult(Interpretable):
    """Result of :func:`expdpy.prepare_regression_table`.

    ``models`` is the list of fitted pyfixest models, ``etable`` the rendered regression
    table (a Great Tables object or a string depending on ``format``), and ``df`` a tidy
    coefficient frame (term, model, estimate, se, p-value).
    """

    models: list[Any]
    etable: Any
    df: pd.DataFrame

    def interpret(self, *, lang: str = "en") -> str:
        """Plain-language reading of sign, magnitude and significance per coefficient."""
        return interpret_regression(self, lang=lang)

    def explain(self, *, lang: str = "en") -> Explainer:
        """Concept explainer keyed to the design (fixed effects / clustering / OLS)."""
        model = self.models[0]
        if bool(getattr(model, "_has_fixef", False)):
            topic = "fixed_effects"
        elif bool(getattr(model, "_is_clustered", False)):
            topic = "clustered_se"
        else:
            topic = "ols"
        return _explain(topic, lang=lang)

    def tidy(self) -> pd.DataFrame:
        """Return the tidy coefficient frame (broom-style ``tidy``)."""
        return self.df

    def glance(self) -> pd.DataFrame:
        """One row per model with N, R² and within-R² (broom-style ``glance``)."""
        import math

        import pandas as pd

        rows = [
            {
                "model": i + 1,
                "depvar": getattr(m, "_depvar", None),
                "N": int(getattr(m, "_N", 0)),
                "r2": float(getattr(m, "_r2", math.nan)),
                "r2_within": float(getattr(m, "_r2_within", math.nan)),
                "has_fe": bool(getattr(m, "_has_fixef", False)),
            }
            for i, m in enumerate(self.models)
        ]
        return pd.DataFrame(rows)


@dataclass(frozen=True)
class FWLPlotResult(Interpretable):
    """Result of :func:`expdpy.prepare_fwl_plot`.

    ``df`` is the residual frame sorted by ``x_resid`` with columns ``x_resid``,
    ``y_resid``, ``fit``, ``lwr`` and ``upr`` (the OLS fit and 95% pointwise confidence band
    of ``y_resid`` on ``x_resid``). ``fig`` is the Plotly figure. ``slope`` equals the
    full-model coefficient on the focal variable (Frisch-Waugh-Lovell theorem); ``se`` is its
    standard error from the full model (clustered when clusters are given, matching
    :func:`prepare_regression_table`) — note the plotted band is the simpler residual-OLS
    confidence interval, so its implied uncertainty can differ from ``se``. ``intercept`` is
    the residual-OLS intercept (≈ 0 when residualized); ``n_obs`` is the complete-case sample
    size; ``r2_within`` is the full model's within-R² (``nan`` when there are no fixed
    effects).
    """

    df: pd.DataFrame
    fig: go.Figure
    slope: float
    se: float
    intercept: float
    n_obs: int
    r2_within: float

    def interpret(self, *, lang: str = "en") -> str:
        """Plain-language reading of the partial slope and the FWL identity."""
        return interpret_fwl(self, lang=lang)

    def explain(self, *, lang: str = "en") -> Explainer:
        """Concept explainer for the Frisch-Waugh-Lovell partial regression."""
        return _explain("fwl", lang=lang)

    def glance(self) -> pd.DataFrame:
        """One-row summary of the partial regression (broom-style ``glance``)."""
        import pandas as pd

        return pd.DataFrame(
            [
                {
                    "slope": self.slope,
                    "se": self.se,
                    "intercept": self.intercept,
                    "n_obs": self.n_obs,
                    "r2_within": self.r2_within,
                }
            ]
        )


@dataclass(frozen=True)
class EstimationResult(Interpretable):
    """Result of :func:`expdpy.prepare_estimation`.

    ``models`` are the fitted pyfixest model(s), ``etable`` the rendered table, ``df`` the
    tidy coefficient frame, ``model_kind`` the estimator (``"ols"``/``"iv"``/``"poisson"``/
    ``"logit"``/``"probit"``), ``fit_stats`` a one-row-per-model summary, and ``notes`` any
    advisory messages raised during estimation.
    """

    models: list[Any]
    etable: Any
    df: pd.DataFrame
    model_kind: str
    fit_stats: pd.DataFrame
    notes: tuple[str, ...] = ()

    def interpret(self, *, lang: str = "en") -> str:
        """Model-kind-aware plain-language reading of the coefficients."""
        return interpret_estimation(self, lang=lang)

    def explain(self, *, lang: str = "en") -> Explainer:
        """Concept explainer keyed to the estimator and design."""
        if self.model_kind == "iv":
            topic = "iv"
        elif self.model_kind in ("poisson", "logit", "probit"):
            topic = "glm"
        else:
            model = self.models[0]
            if bool(getattr(model, "_has_fixef", False)):
                topic = "fixed_effects"
            elif bool(getattr(model, "_is_clustered", False)):
                topic = "clustered_se"
            else:
                topic = "ols"
        return _explain(topic, lang=lang)

    def tidy(self) -> pd.DataFrame:
        """Return the tidy coefficient frame (broom-style ``tidy``)."""
        return self.df

    def glance(self) -> pd.DataFrame:
        """Return the per-model fit-statistics frame (broom-style ``glance``)."""
        return self.fit_stats


@dataclass(frozen=True)
class FixefPlotResult:
    """Result of :func:`expdpy.prepare_fixef_plot`.

    ``df`` has columns ``fixef`` (the fixed-effect dimension), ``level`` and ``value`` (the
    estimated group intercept); ``fig`` is the Plotly figure.
    """

    df: pd.DataFrame
    fig: go.Figure


@dataclass(frozen=True)
class PredictionResult:
    """Result of :func:`expdpy.prepare_predictions`.

    ``df`` holds the fitted ``predicted`` values, plus ``actual`` and ``residual`` columns
    when predicting on the estimation sample (no ``newdata``).
    """

    df: pd.DataFrame


@dataclass(frozen=True)
class JointTestResult:
    """Result of :func:`expdpy.prepare_joint_test` (a Wald joint-significance test)."""

    statistic: float
    p_value: float
    hypotheses: tuple[str, ...]
    distribution: str

    def summary(self) -> str:
        """Return a one-line plain-language verdict for the joint test."""
        terms = ", ".join(self.hypotheses)
        verdict = (
            "jointly statistically significant"
            if self.p_value < 0.05
            else "not jointly statistically significant at the 5% level"
        )
        return (
            f"Joint {self.distribution}-test that [{terms}] are all zero: "
            f"statistic = {self.statistic:.4g}, p = {self.p_value:.4g} — {verdict}."
        )


@dataclass(frozen=True)
class EventStudyResult(Interpretable):
    """Result of :func:`expdpy.prepare_event_study`.

    ``df`` is the tidy event-time path (columns ``event_time``, ``estimate``, ``se``,
    ``ci_lower``, ``ci_upper`` and ``cohort`` — ``cohort`` is filled only for the
    Sun-Abraham ``"saturated"`` estimator). ``fig`` is the Plotly event-study plot,
    ``model`` the fitted pyfixest object, and ``estimator`` the chosen method.
    """

    df: pd.DataFrame
    fig: go.Figure
    model: Any
    estimator: str

    def interpret(self, *, lang: str = "en") -> str:
        """Plain-language pre-trend diagnostic and dynamic-effect summary."""
        return interpret_event_study(self, lang=lang)

    def explain(self, *, lang: str = "en") -> Explainer:
        """Concept explainer for event studies / staggered difference-in-differences."""
        return _explain("event_study", lang=lang)

    def tidy(self) -> pd.DataFrame:
        """Return the tidy event-time path (broom-style ``tidy``)."""
        return self.df


@dataclass(frozen=True)
class PanelViewResult:
    """Result of :func:`expdpy.prepare_panel_view`.

    ``df`` is the treatment quilt (units by periods, 0/1) or, when an ``outcome`` is given,
    the tidy outcome frame; ``fig`` is the Plotly figure.
    """

    df: pd.DataFrame
    fig: go.Figure


@dataclass(frozen=True)
class HausmanTestResult(Interpretable):
    """Result of :func:`expdpy.prepare_hausman_test` (fixed vs random effects).

    ``statistic`` is the Hausman chi-squared statistic, ``df_test`` its degrees of freedom,
    ``p_value`` the p-value, and ``fe_coefs`` / ``re_coefs`` the compared coefficients.
    """

    statistic: float
    df_test: int
    p_value: float
    fe_coefs: pd.DataFrame
    re_coefs: pd.DataFrame

    def interpret(self, *, lang: str = "en") -> str:
        """Plain-language verdict on the fixed-vs-random-effects choice."""
        if self.p_value < 0.05:
            verdict = (
                "**reject** the null — the random-effects assumption is violated, so prefer "
                "**fixed effects**"
            )
        else:
            verdict = (
                "**fail to reject** the null — **random effects** is admissible (and more "
                "efficient than fixed effects)"
            )
        return (
            f"Hausman test (χ²({self.df_test}) = {self.statistic:.3f}, "
            f"p = {self.p_value:.4g}): {verdict}. Note that failing to reject reflects a lack "
            "of evidence against random effects, not proof that it is correct."
        )

    def explain(self, *, lang: str = "en") -> Explainer:
        """Concept explainer for the Hausman test."""
        return _explain("hausman", lang=lang)


@dataclass(frozen=True)
class SandboxResult(Interpretable):
    """Result of an ``expdpy.sandbox_*`` teaching demonstration.

    ``df`` is the comparison table, ``fig`` the headline figure, ``summary`` the scalar facts
    the demonstration turns on, and ``topic`` the concept it illustrates.
    """

    df: pd.DataFrame
    fig: go.Figure
    summary: dict[str, float]
    topic: str

    def interpret(self, *, lang: str = "en") -> str:
        """Plain-language takeaway of the demonstration."""
        return interpret_sandbox(self, lang=lang)

    def explain(self, *, lang: str = "en") -> Explainer:
        """Concept explainer for the demonstrated topic."""
        return _explain(self.topic, lang=lang)


@dataclass(frozen=True)
class RobustInferenceResult:
    """Result of :func:`expdpy.prepare_robust_inference`.

    ``method`` is ``"ritest"`` (randomization inference) or ``"wildboot"`` (wild cluster
    bootstrap); ``estimate`` and ``p_value`` are for the tested ``param``; ``conf_int`` is
    the (lower, upper) interval; ``raw`` is the underlying pyfixest result series.
    """

    method: str
    param: str
    estimate: float
    p_value: float
    conf_int: tuple[float, float]
    reps: int
    raw: Any
