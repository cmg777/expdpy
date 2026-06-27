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
* **Utilities** — shared helpers used across modules: ``set_panel`` / ``resolve_panel``
  (declare the panel once), ``set_labels`` / ``resolve_label`` (declare human-readable
  variable labels once), ``build_data_def`` (infer a data dictionary from a raw frame),
  ``treat_outliers`` (winsorize/truncate), and the concept-explainer registry entry points
  (``explain`` / ``list_topics``).

Three no-code ``ExPdPy`` apps (one per module) build on the same functions — see
:mod:`expdpy.streamlit_app`.
"""

from __future__ import annotations

from expdpy._data_def import build_data_def
from expdpy._labels import resolve_label, set_labels
from expdpy._panel import resolve_panel, set_panel
from expdpy._types import (
    AnimatedScatterResult,
    BarChartResult,
    BetaConvergenceResult,
    ByGroupBarGraphResult,
    ByGroupTrendGraphResult,
    ByGroupViolinResult,
    CoefficientPlotResult,
    ConvergenceClubsResult,
    CorrelationGraphResult,
    CorrelationTableResult,
    CRETableResult,
    DescriptiveTableResult,
    DistributionOverTimeResult,
    EstimationResult,
    EventStudyResult,
    ExtObsTableResult,
    FixefPlotResult,
    FWLPlotResult,
    HausmanTestResult,
    HistogramResult,
    IVRegressionResult,
    JointTestResult,
    KuznetsWavesResult,
    MarginalEffectsResult,
    MissingValuesResult,
    PanelStructureResult,
    PanelViewResult,
    PredictionResult,
    QuantileTrendGraphResult,
    RegressionTableResult,
    RobustInferenceResult,
    SandboxResult,
    ScatterPlotResult,
    SigmaConvergenceResult,
    SpaghettiGraphResult,
    TransitionMatrixResult,
    TrendGraphResult,
    ValueHeatmapResult,
    WithinBetweenScatterResult,
    WithinPersistenceResult,
    XtsumTableResult,
)
from expdpy.animated_scatter import explore_animated_scatter_plot
from expdpy.by_group import (
    explore_bar_plot_by_group,
    explore_trend_plot_by_group,
    explore_violin_plot_by_group,
)
from expdpy.coefplot import analyze_coefficient_plot
from expdpy.convergence import (
    analyze_beta_convergence,
    analyze_convergence_clubs,
    analyze_sigma_convergence,
)
from expdpy.correlation import explore_correlation_plot
from expdpy.cre import analyze_cre_table
from expdpy.did import analyze_event_study, analyze_panel_view
from expdpy.distributions import explore_bar_plot, explore_histogram
from expdpy.dynamics import (
    explore_distribution_over_time,
    explore_transition_matrix,
    explore_within_persistence,
)
from expdpy.estimation import analyze_estimation
from expdpy.fwl import analyze_fwl_plot
from expdpy.inference import analyze_robust_inference
from expdpy.iv import analyze_iv_regression, analyze_panel_iv_regression
from expdpy.kuznets import analyze_kuznets_waves
from expdpy.marginal_effects import analyze_marginal_effects_plot
from expdpy.missing import explore_missing_values_plot
from expdpy.outliers import treat_outliers
from expdpy.panel_models import analyze_hausman_test, analyze_panel_table
from expdpy.panel_structure import explore_panel_structure, explore_value_heatmap
from expdpy.panel_summary import (
    explore_scatter_plot_within_between,
    explore_xtsum_table,
)
from expdpy.pedagogy import Explainer, explain, list_topics
from expdpy.postestimation import (
    analyze_fixef_plot,
    analyze_joint_test,
    analyze_predictions,
)
from expdpy.regression import analyze_regression_table
from expdpy.sandbox import (
    learn_beta_convergence,
    learn_clustering_se,
    learn_convergence_clubs,
    learn_first_differences,
    learn_kuznets_waves,
    learn_omitted_variable_bias,
    learn_pooled_vs_fixed_effects,
    learn_sigma_convergence,
    learn_within_vs_lsdv,
)
from expdpy.scatter import explore_scatter_plot
from expdpy.spaghetti import explore_spaghetti_plot
from expdpy.tables import (
    explore_correlation_table,
    explore_descriptive_table,
    explore_ext_obs_table,
)
from expdpy.trends import explore_quantile_trend_plot, explore_trend_plot

__version__ = "0.4.20"

__all__ = [
    # ===== EXPLORE =====
    # tables
    "explore_descriptive_table",
    "explore_correlation_table",
    "explore_ext_obs_table",
    # distributions
    "explore_histogram",
    "explore_bar_plot",
    # correlation graph
    "explore_correlation_plot",
    # trends
    "explore_trend_plot",
    "explore_quantile_trend_plot",
    # by group
    "explore_bar_plot_by_group",
    "explore_trend_plot_by_group",
    "explore_violin_plot_by_group",
    # missing values
    "explore_missing_values_plot",
    # scatter
    "explore_scatter_plot",
    "explore_animated_scatter_plot",
    # within/between variation
    "explore_xtsum_table",
    "explore_scatter_plot_within_between",
    # per-unit trajectories
    "explore_spaghetti_plot",
    # panel structure
    "explore_panel_structure",
    "explore_value_heatmap",
    # distribution & transition dynamics
    "explore_distribution_over_time",
    "explore_transition_matrix",
    "explore_within_persistence",
    # ===== ANALYZE =====
    # regression table (OLS / fixed effects / clustered SEs)
    "analyze_regression_table",
    # estimation (OLS + stepwise + Newey-West / Driscoll-Kraay + weights)
    "analyze_estimation",
    # instrumental variables / 2SLS (cross-section + panel) + weak-instrument F
    "analyze_iv_regression",
    "analyze_panel_iv_regression",
    # diagnostic plots
    "analyze_fwl_plot",
    "analyze_coefficient_plot",
    "analyze_marginal_effects_plot",
    # panel models (pooled / between / fixed / random effects, CRE) + Hausman
    "analyze_panel_table",
    "analyze_cre_table",
    "analyze_hausman_test",
    # post-estimation
    "analyze_fixef_plot",
    "analyze_predictions",
    "analyze_joint_test",
    # robust inference
    "analyze_robust_inference",
    # event study / staggered DiD
    "analyze_event_study",
    "analyze_panel_view",
    # beta convergence (unconditional + conditional + rolling)
    "analyze_beta_convergence",
    # sigma convergence (dispersion-over-time + log-trend)
    "analyze_sigma_convergence",
    # club convergence (Phillips-Sul log(t) test + data-driven clustering)
    "analyze_convergence_clubs",
    # Kuznets waves (extended Kuznets curve: pooled / between / within, side by side)
    "analyze_kuznets_waves",
    # ===== LEARN =====
    # concept sandboxes
    "learn_omitted_variable_bias",
    "learn_pooled_vs_fixed_effects",
    "learn_clustering_se",
    "learn_first_differences",
    "learn_within_vs_lsdv",
    "learn_beta_convergence",
    "learn_sigma_convergence",
    "learn_convergence_clubs",
    "learn_kuznets_waves",
    # concept-explainer registry type
    "Explainer",
    # ===== UTILITIES =====
    # panel declaration
    "set_panel",
    "resolve_panel",
    # variable labels (human-readable display names)
    "set_labels",
    "resolve_label",
    # data dictionary inference (df_def)
    "build_data_def",
    # outlier treatment
    "treat_outliers",
    # concept explainers (registry entry points)
    "explain",
    "list_topics",
    # ===== RESULT TYPES =====
    # explore
    "DescriptiveTableResult",
    "CorrelationTableResult",
    "ExtObsTableResult",
    "HistogramResult",
    "BarChartResult",
    "CorrelationGraphResult",
    "ScatterPlotResult",
    "AnimatedScatterResult",
    "MissingValuesResult",
    "TrendGraphResult",
    "QuantileTrendGraphResult",
    "ByGroupBarGraphResult",
    "ByGroupTrendGraphResult",
    "ByGroupViolinResult",
    "XtsumTableResult",
    "WithinBetweenScatterResult",
    "SpaghettiGraphResult",
    "PanelStructureResult",
    "ValueHeatmapResult",
    "DistributionOverTimeResult",
    "TransitionMatrixResult",
    "WithinPersistenceResult",
    # analyze
    "RegressionTableResult",
    "EstimationResult",
    "IVRegressionResult",
    "FWLPlotResult",
    "CoefficientPlotResult",
    "MarginalEffectsResult",
    "CRETableResult",
    "HausmanTestResult",
    "FixefPlotResult",
    "PredictionResult",
    "JointTestResult",
    "RobustInferenceResult",
    "EventStudyResult",
    "PanelViewResult",
    "BetaConvergenceResult",
    "SigmaConvergenceResult",
    "ConvergenceClubsResult",
    "KuznetsWavesResult",
    # learn
    "SandboxResult",
]
