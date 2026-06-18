"""Frozen result dataclasses returned by the ``prepare_*`` functions.

The R package returns ``list(df = ..., plot = ...)`` or ``list(df = ..., kable_ret = ...)``
objects. In Python we use small, typed, immutable dataclasses that expose the underlying
``pandas.DataFrame`` alongside the rendered object (a Plotly ``Figure``, a Great Tables
``GT``, or a pyfixest ``etable`` result).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd
    import plotly.graph_objects as go
    from great_tables import GT

__all__ = [
    "BarChartResult",
    "ByGroupBarGraphResult",
    "ByGroupTrendGraphResult",
    "CorrelationGraphResult",
    "CorrelationTableResult",
    "DescriptiveTableResult",
    "ExtObsTableResult",
    "HistogramResult",
    "QuantileTrendGraphResult",
    "RegressionTableResult",
    "TrendGraphResult",
]


@dataclass(frozen=True)
class DescriptiveTableResult:
    """Result of :func:`expdpy.prepare_descriptive_table`."""

    df: pd.DataFrame
    gt: GT


@dataclass(frozen=True)
class CorrelationTableResult:
    """Result of :func:`expdpy.prepare_correlation_table`.

    ``df_corr`` holds Pearson correlations above and Spearman correlations below the
    diagonal; ``df_prob`` the matching p-values; ``df_n`` the pairwise observation counts.
    """

    df_corr: pd.DataFrame
    df_prob: pd.DataFrame
    df_n: pd.DataFrame
    gt: GT


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
class TrendGraphResult:
    """Result of :func:`expdpy.prepare_trend_graph`."""

    df: pd.DataFrame
    fig: go.Figure


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
class RegressionTableResult:
    """Result of :func:`expdpy.prepare_regression_table`.

    ``models`` is the list of fitted pyfixest models, ``etable`` the rendered regression
    table (a Great Tables object or a string depending on ``format``), and ``df`` a tidy
    coefficient frame (term, model, estimate, se, p-value).
    """

    models: list[Any]
    etable: Any
    df: pd.DataFrame
