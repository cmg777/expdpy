"""expdpy — Explore your panel data interactively.

A Python port of the ExPanDaR R package (Joachim Gassen, TRR 266). Provides a set
of analytical functions for exploratory analysis of panel and cross-sectional data
(descriptive tables, correlations, time trends, scatter plots, regression tables)
returning interactive Plotly figures and Great Tables / pyfixest output, plus the
``ExPdPy`` interactive app (Streamlit).
"""

from __future__ import annotations

from expdpy._types import (
    BarChartResult,
    ByGroupBarGraphResult,
    ByGroupTrendGraphResult,
    CoefficientPlotResult,
    CorrelationGraphResult,
    CorrelationTableResult,
    DescriptiveTableResult,
    EstimationResult,
    EventStudyResult,
    ExtObsTableResult,
    FixefPlotResult,
    FWLPlotResult,
    HausmanTestResult,
    HistogramResult,
    JointTestResult,
    PanelViewResult,
    PredictionResult,
    QuantileTrendGraphResult,
    RegressionTableResult,
    RobustInferenceResult,
    SandboxResult,
    TrendGraphResult,
)
from expdpy.by_group import (
    prepare_by_group_bar_graph,
    prepare_by_group_trend_graph,
    prepare_by_group_violin_graph,
)
from expdpy.coefplot import prepare_coefficient_plot
from expdpy.correlation import prepare_correlation_graph
from expdpy.did import prepare_event_study, prepare_panel_view
from expdpy.distributions import prepare_bar_chart, prepare_histogram
from expdpy.estimation import prepare_estimation
from expdpy.fwl import prepare_fwl_plot
from expdpy.inference import prepare_robust_inference
from expdpy.missing import prepare_missing_values_graph
from expdpy.outliers import treat_outliers
from expdpy.panel_models import prepare_hausman_test, prepare_panel_table
from expdpy.pedagogy import Explainer, explain, list_topics
from expdpy.postestimation import (
    prepare_fixef_plot,
    prepare_joint_test,
    prepare_predictions,
)
from expdpy.regression import prepare_regression_table
from expdpy.sandbox import (
    sandbox_clustering_se,
    sandbox_omitted_variable_bias,
    sandbox_pooled_vs_fixed_effects,
)
from expdpy.scatter import prepare_scatter_plot
from expdpy.tables import (
    prepare_correlation_table,
    prepare_descriptive_table,
    prepare_ext_obs_table,
)
from expdpy.trends import prepare_quantile_trend_graph, prepare_trend_graph

__version__ = "0.3.0"

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
    # estimation (IV / Poisson / GLM / model comparison)
    "prepare_estimation",
    # post-estimation
    "prepare_fixef_plot",
    "prepare_predictions",
    "prepare_joint_test",
    # robust inference
    "prepare_robust_inference",
    # fwl plot
    "prepare_fwl_plot",
    # coefficient plot
    "prepare_coefficient_plot",
    # event study / staggered DiD
    "prepare_event_study",
    "prepare_panel_view",
    # concept sandboxes
    "sandbox_omitted_variable_bias",
    "sandbox_pooled_vs_fixed_effects",
    "sandbox_clustering_se",
    # panel models (linearmodels)
    "prepare_panel_table",
    "prepare_hausman_test",
    # pedagogy
    "explain",
    "list_topics",
    "Explainer",
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
    "FWLPlotResult",
    "CoefficientPlotResult",
    "EstimationResult",
    "FixefPlotResult",
    "PredictionResult",
    "JointTestResult",
    "RobustInferenceResult",
    "EventStudyResult",
    "PanelViewResult",
    "SandboxResult",
    "HausmanTestResult",
]
