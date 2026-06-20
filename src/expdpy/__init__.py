"""expdpy — Explore, Analyze and Learn panel data.

A Python port of the ExPanDaR R package (Joachim Gassen, TRR 266), organized around three
conceptual modules:

* **Explore** — exploratory analysis of panel and cross-sectional data: descriptive and
  correlation tables, distributions, time trends, group comparisons, missing-value maps and
  scatter plots, returning interactive Plotly figures and Great Tables output.
* **Analyze** — panel estimators: OLS / fixed effects, random effects, correlated random
  effects (Mundlak), the Frisch-Waugh-Lovell plot, pooled / between models, the Hausman test,
  post-estimation, robust inference and event-study / difference-in-differences.
* **Learn** — a teaching layer: concept explainers (``explain`` / ``list_topics``),
  plain-language ``.interpret()`` on every result, and runnable concept sandboxes.

Three no-code ``ExPdPy`` apps (one per module) build on the same functions — see
:mod:`expdpy.streamlit_app`.
"""

from __future__ import annotations

from expdpy._types import (
    BarChartResult,
    ByGroupBarGraphResult,
    ByGroupTrendGraphResult,
    CoefficientPlotResult,
    CorrelationGraphResult,
    CorrelationTableResult,
    CRETableResult,
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
from expdpy.cre import prepare_cre_table
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
    sandbox_first_differences,
    sandbox_omitted_variable_bias,
    sandbox_pooled_vs_fixed_effects,
    sandbox_within_vs_lsdv,
)
from expdpy.scatter import prepare_scatter_plot
from expdpy.tables import (
    prepare_correlation_table,
    prepare_descriptive_table,
    prepare_ext_obs_table,
)
from expdpy.trends import prepare_quantile_trend_graph, prepare_trend_graph

__version__ = "0.4.0"

__all__ = [
    # ===== EXPLORE =====
    # outlier treatment
    "treat_outliers",
    # tables
    "prepare_descriptive_table",
    "prepare_correlation_table",
    "prepare_ext_obs_table",
    # distributions
    "prepare_histogram",
    "prepare_bar_chart",
    # correlation graph
    "prepare_correlation_graph",
    # trends
    "prepare_trend_graph",
    "prepare_quantile_trend_graph",
    # by group
    "prepare_by_group_bar_graph",
    "prepare_by_group_trend_graph",
    "prepare_by_group_violin_graph",
    # missing values
    "prepare_missing_values_graph",
    # scatter
    "prepare_scatter_plot",
    # ===== ANALYZE =====
    # regression table (OLS / fixed effects / clustered SEs)
    "prepare_regression_table",
    # estimation (OLS + stepwise + Newey-West / Driscoll-Kraay + weights)
    "prepare_estimation",
    # diagnostic plots
    "prepare_fwl_plot",
    "prepare_coefficient_plot",
    # panel models (pooled / between / fixed / random effects, CRE) + Hausman
    "prepare_panel_table",
    "prepare_cre_table",
    "prepare_hausman_test",
    # post-estimation
    "prepare_fixef_plot",
    "prepare_predictions",
    "prepare_joint_test",
    # robust inference
    "prepare_robust_inference",
    # event study / staggered DiD
    "prepare_event_study",
    "prepare_panel_view",
    # ===== LEARN =====
    # concept sandboxes
    "sandbox_omitted_variable_bias",
    "sandbox_pooled_vs_fixed_effects",
    "sandbox_clustering_se",
    "sandbox_first_differences",
    "sandbox_within_vs_lsdv",
    # pedagogy
    "explain",
    "list_topics",
    "Explainer",
    # ===== RESULT TYPES =====
    # explore
    "DescriptiveTableResult",
    "CorrelationTableResult",
    "ExtObsTableResult",
    "HistogramResult",
    "BarChartResult",
    "CorrelationGraphResult",
    "TrendGraphResult",
    "QuantileTrendGraphResult",
    "ByGroupBarGraphResult",
    "ByGroupTrendGraphResult",
    # analyze
    "RegressionTableResult",
    "EstimationResult",
    "FWLPlotResult",
    "CoefficientPlotResult",
    "CRETableResult",
    "HausmanTestResult",
    "FixefPlotResult",
    "PredictionResult",
    "JointTestResult",
    "RobustInferenceResult",
    "EventStudyResult",
    "PanelViewResult",
    # learn
    "SandboxResult",
]
