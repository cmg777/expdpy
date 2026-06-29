# Choosing a function

A decision guide from intent to call. For exact signatures, see `function-catalog.md`.

## I want to look at the data first
- Summary statistics (a formatted table) -> `explore_descriptive_table(df)`
- Pairwise correlations (Pearson/Spearman) -> `explore_correlation_table(df[cols])`
- Where are the missing values? -> `explore_missing_values_plot(df)`
- Extreme / outlier observations -> `explore_ext_obs_table(df, var=...)` (and `treat_outliers` to winsorize/truncate)
- Distribution of one variable -> `explore_histogram(df, var)`

## I want to see how things move over time (panel)
- Group mean trend with SE bars -> `explore_trend_plot(df, var)`
- Every unit's trajectory -> `explore_spaghetti_plot(df, var)`
- How a distribution shifts across periods -> `explore_distribution_over_time(df, var)`
- Is the panel balanced? gaps? -> `explore_panel_structure(df)`
- Decompose variation into within/between/overall -> `explore_xtsum_table(df, vars)`

## I want to see a bivariate relationship
- Scatter (optionally with LOESS, color, size) -> `explore_scatter_plot(df, x, y)`
- Within vs between decomposition of x-y -> `explore_scatter_plot_within_between(df, x, y)`
- Animated bubble scatter over time -> `explore_animated_scatter_plot(df, x, y, size=...)`

## I want to estimate a model
- OLS, with optional fixed effects + clustered SEs -> `analyze_regression_table(df, dvs, idvs, feffects=, clusters=)`
- Pooled / between / fixed / random effects side by side -> `analyze_panel_table(df, dv, idvs)`
- Random effects vs fixed effects: which? -> `analyze_hausman_test(df, dv, idvs)`
- Mundlak / correlated random effects -> `analyze_cre_table(df, dv, idvs)`
- Endogenous regressor (2SLS / IV) -> `analyze_iv_regression(df, dv, endog, instruments)` (panel: `analyze_panel_iv_regression`)
- Interaction / marginal effects -> `analyze_marginal_effects_plot(...)`
- Frisch-Waugh-Lovell partial relationship -> `analyze_fwl_plot(df, dv, var)`

## I want a causal / treatment design
- Staggered adoption event study / DiD -> `analyze_event_study(df, outcome=, cohort=, unit=, time=)`
- Visualize the treatment structure -> `analyze_panel_view(df, ...)`
- Robust inference (randomization / wild bootstrap) -> `analyze_robust_inference(...)`

## I want a growth / convergence question
- Are poorer units catching up? (beta-convergence, speed, half-life) -> `analyze_beta_convergence(df, var)`
- Is cross-sectional dispersion shrinking? (sigma-convergence) -> `analyze_sigma_convergence(df, var)`
- Are there convergence clubs? (Phillips-Sul) -> `analyze_convergence_clubs(df, var)`
- Inequality vs development (Kuznets curve/waves) -> `analyze_kuznets_waves(df, inequality, development)`

## I want to teach / understand a concept
- Run a simulated demonstration -> `learn_*` (e.g. `learn_omitted_variable_bias`, `learn_pooled_vs_fixed_effects`)
- Read the theory -> `explain("<topic>")`; list keys with `list_topics()`
