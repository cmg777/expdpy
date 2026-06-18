"""App configuration state (port of ExPanDaR's ``default_config`` list)."""

from __future__ import annotations

from copy import deepcopy

__all__ = ["DEFAULT_CONFIG", "create_config", "parse_config"]

# The full set of configuration keys mirrors ExPanDaR's `default_config` (server.R).
# Values are scalars/lists; "None"/"All"/"Full Sample" are the app's sentinel strings.
DEFAULT_CONFIG: dict = {
    "sample": None,
    "subset_factor": "Full Sample",
    "subset_value": "All",
    "group_factor": "None",
    "balanced_panel": False,
    "outlier_treatment": "1",  # 1=none, 2=winsorize 1%, 3=winsorize 5%
    "outlier_factor": "None",
    "udvars": None,
    "delvars": None,
    # bar chart
    "bar_chart_var1": "None",
    "bar_chart_var2": "None",
    "bar_chart_group_by": "All",
    "bar_chart_relative": False,
    # missing values
    "missing_values_group_by": "All",
    # descriptive
    "desc_group_by": "All",
    # by-group bar
    "bgbg_var": "None",
    "bgbg_byvar": "None",
    "bgbg_stat": "mean",
    "bgbg_sort_by_stat": False,
    "bgbg_group_by": "All",
    # by-group violin
    "bgvg_var": "None",
    "bgvg_byvar": "None",
    "bgvg_sort_by_stat": False,
    "bgvg_group_by": "All",
    # histogram
    "hist_var": "None",
    "hist_group_by": "All",
    "hist_nr_of_breaks": 20,
    # extreme observations
    "ext_obs_var": "None",
    "ext_obs_group_by": "All",
    "ext_obs_period_by": "All",
    # trend
    "trend_graph_var1": "None",
    "trend_graph_var2": "None",
    "trend_graph_var3": "None",
    "trend_graph_group_by": "All",
    # quantile trend
    "quantile_trend_graph_var": "None",
    "quantile_trend_graph_quantiles": ["0.05", "0.25", "0.50", "0.75", "0.95"],
    "quantile_trend_graph_group_by": "All",
    # by-group trend
    "bgtg_var": "None",
    "bgtg_byvar": "None",
    "bgtg_group_by": "All",
    # correlation
    "corrplot_group_by": "All",
    # scatter
    "scatter_x": "None",
    "scatter_y": "None",
    "scatter_size": "None",
    "scatter_color": "None",
    "scatter_loess": True,
    "scatter_sample": True,
    "scatter_group_by": "All",
    # regression
    "reg_y": "None",
    "reg_x": "None",
    "reg_fe1": "None",
    "reg_fe2": "None",
    "reg_by": "None",
    "cluster": 1,
    "model": "ols",
}


def create_config(overrides: dict | None = None) -> dict:
    """Return a fresh copy of the default config, updated with ``overrides``."""
    cfg = deepcopy(DEFAULT_CONFIG)
    if overrides:
        cfg.update(overrides)
    return cfg


def parse_config(config_list: dict | None) -> dict:
    """Merge a user-provided ``config_list`` over the defaults (missing keys filled).

    Mirrors ExPanDaR's ``parse_config``: every default key is guaranteed present.
    """
    cfg = deepcopy(DEFAULT_CONFIG)
    if config_list:
        for key, value in config_list.items():
            cfg[key] = value
    return cfg
