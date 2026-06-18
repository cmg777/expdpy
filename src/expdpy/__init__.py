"""expdpy — Explore your panel data interactively.

A Python port of the ExPanDaR R package (Joachim Gassen, TRR 266). Provides a set
of analytical functions for exploratory analysis of panel and cross-sectional data
(descriptive tables, correlations, time trends, scatter plots, regression tables)
returning interactive Plotly figures and Great Tables / pyfixest output, plus the
``ExPdPy`` interactive app (Shiny for Python).
"""

from __future__ import annotations

from expdpy._types import (
    BarChartResult,
    ByGroupBarGraphResult,
    ByGroupTrendGraphResult,
    CorrelationGraphResult,
    CorrelationTableResult,
    DescriptiveTableResult,
    ExtObsTableResult,
    HistogramResult,
    QuantileTrendGraphResult,
    RegressionTableResult,
    TrendGraphResult,
)
from expdpy.by_group import (
    prepare_by_group_bar_graph,
    prepare_by_group_trend_graph,
    prepare_by_group_violin_graph,
)
from expdpy.correlation import prepare_correlation_graph
from expdpy.distributions import prepare_bar_chart, prepare_histogram
from expdpy.missing import prepare_missing_values_graph
from expdpy.outliers import treat_outliers
from expdpy.regression import prepare_regression_table
from expdpy.scatter import prepare_scatter_plot
from expdpy.tables import (
    prepare_correlation_table,
    prepare_descriptive_table,
    prepare_ext_obs_table,
)
from expdpy.trends import prepare_quantile_trend_graph, prepare_trend_graph

__version__ = "0.1.0.dev0"

__all__ = [
    # outliers
    "treat_outliers",
    # tables
    "prepare_descriptive_table",
    "prepare_correlation_table",
    "prepare_ext_obs_table",
    # correlation graph
    "prepare_correlation_graph",
    # trends
    "prepare_trend_graph",
    "prepare_quantile_trend_graph",
    # by group
    "prepare_by_group_bar_graph",
    "prepare_by_group_trend_graph",
    "prepare_by_group_violin_graph",
    # distributions
    "prepare_histogram",
    "prepare_bar_chart",
    # missing
    "prepare_missing_values_graph",
    # scatter
    "prepare_scatter_plot",
    # regression
    "prepare_regression_table",
    # result types
    "DescriptiveTableResult",
    "CorrelationTableResult",
    "CorrelationGraphResult",
    "ExtObsTableResult",
    "TrendGraphResult",
    "QuantileTrendGraphResult",
    "ByGroupBarGraphResult",
    "ByGroupTrendGraphResult",
    "HistogramResult",
    "BarChartResult",
    "RegressionTableResult",
]
