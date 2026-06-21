

<p align="center">
  <img src="https://raw.githubusercontent.com/cmg777/expdpy/main/docs/images/hero.webp" alt="expdpy — A Python library to explore panel data interactively" width="100%">
</p>

<!-- badges: start -->
[![CI](https://github.com/cmg777/expdpy/actions/workflows/ci.yml/badge.svg)](https://github.com/cmg777/expdpy/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/cmg777/expdpy/branch/main/graph/badge.svg)](https://codecov.io/gh/cmg777/expdpy)
[![Docs](https://github.com/cmg777/expdpy/actions/workflows/docs.yml/badge.svg)](https://cmg777.github.io/expdpy/)
[![PyPI](https://img.shields.io/pypi/v/expdpy.svg)](https://pypi.org/project/expdpy/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://pypi.org/project/expdpy/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
<!-- badges: end -->

**expdpy** is a Python library for interactive analysis of **panel and cross-sectional data**,
organized around three modules — **Explore**, **Analyze** and **Learn**. It pairs composable
functions — that return interactive [Plotly](https://plotly.com/python/) figures and
publication-quality [Great Tables](https://posit-dev.github.io/great-tables/) — with
**fixest-style econometrics**, a built-in **teaching layer** that explains and interprets every
result, and **three no-code `ExPdPy` apps** (one per module). It is built for beginners and
applied researchers alike.

It is built on the modern Python data and econometrics stack:

- **[Plotly](https://plotly.com/python/)** — interactive figures
- **[pyfixest](https://github.com/py-econometrics/pyfixest)** — fixed-effects and difference-in-differences estimators
- **[Great Tables](https://posit-dev.github.io/great-tables/)** — publication-quality tables
- **[linearmodels](https://bashtage.github.io/linearmodels/)** — random effects, between, correlated random effects, and the Hausman test
- **[Streamlit](https://streamlit.io/)** — the no-code `ExPdPy` apps

## Features

### Explore panel data

Descriptive, correlation and extreme-observation tables; histograms and category bar charts;
time trends and quantile trends; by-group bar, violin and trend views; scatter plots with an
optional LOESS smoother; a missing-value heatmap across the panel; and outlier treatment with
`treat_outliers`. Each function takes a `pandas` DataFrame and returns an interactive Plotly
figure or a Great Tables object you can drop straight into a notebook or report.

A dedicated set of **panel-aware** views makes the cross-unit-vs-over-time structure explicit:
the **within/between variation** table `explore_xtsum_table` (Stata `xtsum`-style) and the
**within-vs-between scatter** `explore_scatter_plot_within_between`; **per-unit trajectories**
(`explore_spaghetti_plot`); **panel-structure diagnostics** — a balance/gaps summary and
unit-by-period presence grid (`explore_panel_structure`) plus a unit-by-time value heatmap
(`explore_value_heatmap`); and **distribution & transition dynamics** — `explore_distribution_over_time`
(ridgeline or animated), `explore_transition_matrix`, and within-unit serial-correlation via
`explore_within_persistence`. Panel functions take an `entity` (unit) and a `time` id; declare
them once with `set_panel(df, entity=..., time=...)` and the rest of Explore can omit them.

### Analyze panel data

OLS with **multi-way fixed effects** and **clustered standard errors** via
[pyfixest](https://github.com/py-econometrics/pyfixest), plus a richer `analyze_estimation`
adding **stepwise / multiple-outcome** comparison, serial-correlation-robust standard errors
(**Newey–West**, **Driscoll–Kraay**) and weights. Estimate **pooled / between / fixed / random
effects** with `analyze_panel_table`, bring within estimates into a random-effects frame with
the **correlated-random-effects (Mundlak)** estimator `analyze_cre_table`, and choose between
fixed and random effects with the **Hausman test**. Round it out with post-estimation tools
(fixed-effect plots, predictions, Wald joint tests), **robust inference** (randomization
inference and the wild cluster bootstrap), **Frisch–Waugh–Lovell** and **coefficient** plots,
and modern **event-study / staggered difference-in-differences** estimators (Gardner's `did2s`,
Sun–Abraham, local-projections DiD, dynamic TWFE) with a built-in pre-trend diagnostic and a
treatment-structure `analyze_panel_view`.

### Learn panel data

Every result speaks plain language. `.interpret()` gives an **associational** reading of the
output (never a causal claim unless the design supports it); `.explain()`, together with
`explain(topic)` and `list_topics()`, provides concept explainers for fixed effects,
clustering, random effects, the Mundlak device, first differences, demeaning, dummy variables,
event studies, omitted-variable bias and more. Result objects also expose broom-style
`.tidy()` / `.glance()`. **Concept sandboxes** simulate data so a learner can *see* and tune a
concept — `learn_omitted_variable_bias`, `learn_pooled_vs_fixed_effects`,
`learn_clustering_se`, `learn_first_differences`, and `learn_within_vs_lsdv` (which shows
first differences ≈ demeaning ≈ least-squares dummy variables).

### Three no-code apps — Streamlit

The whole workflow without writing code, in three apps — **Explore**, **Analyze** and
**Learn** — that share a sidebar **sample pipeline** (subset filters, outlier treatment,
user-defined variables) and differ only in which pages they expose. The
[apps](https://cmg777.github.io/expdpy/streamlit.html) deploy to
[Streamlit Community Cloud](https://streamlit.io/cloud) in one click.

### Reproducibility & safety

Any in-app exploration exports to a **runnable bundle** — a Jupyter notebook, a `.py` script
and the prepared data (parquet) — that recreates every displayed result with `expdpy` calls.
Analysis configurations **save and load** as JSON. New variables can be defined live through a
**restricted-AST expression evaluator** (never `eval`/`exec`) with **panel-aware `lag`/`lead`**
that shift within each cross-section.

### Bundled datasets

`expdpy.data` ships ready-to-explore panels — **`kuznets`** (the flagship N-shaped
Kuznets-curve demo), `gapminder`, **`staggered_did`** (a synthetic staggered-adoption panel
for the event-study / DiD tools), and **`firms`** (a small *unbalanced* panel — staggered
entry/exit, interior gaps, a discrete size class and persistent revenue — for the
panel-structure, transition and persistence views). See the
[kuznets dataset](https://cmg777.github.io/expdpy/explanation/kuznets-dataset.html) page for the
data dictionary.

## Installation

Install the latest release from PyPI (random effects, CRE and the Hausman test work out of the
box; the apps need the `streamlit` extra):

```bash
pip install expdpy
pip install "expdpy[streamlit]"   # the no-code ExPdPy apps (Streamlit)
```

Using [uv](https://docs.astral.sh/uv/):

```bash
uv pip install expdpy
uv pip install "expdpy[streamlit]"
```

### Development version (latest from GitHub)

For the most up-to-date, unreleased version, install straight from the `main` branch:

```bash
pip install "git+https://github.com/cmg777/expdpy.git"
pip install "expdpy[streamlit] @ git+https://github.com/cmg777/expdpy.git"
```

Pin to a release, branch, or commit for reproducible installs:

```bash
pip install "expdpy==0.5.0"
pip install "git+https://github.com/cmg777/expdpy.git@v0.5.0"
pip install "git+https://github.com/cmg777/expdpy.git@main"
```

Requires Python 3.10+.

> **Upgrading from 0.4.x?** In **0.5.0** every analysis function gained a module prefix:
> `prepare_*` → `explore_*` / `analyze_*` and `sandbox_*` → `learn_*`, with figures ending in
> `_plot`, tables in `_table`, and scope qualifiers moved to the end (e.g.
> `prepare_by_group_violin_graph` → `explore_violin_plot_by_group`). The utilities `set_panel`,
> `resolve_panel`, `treat_outliers`, `explain` and `list_topics` keep their names. See the
> [changelog](https://cmg777.github.io/expdpy/changelog.html) for the full rename map.

## At a glance

The lead example throughout the docs is the bundled `kuznets` panel (80 countries ×
2015–2025): a synthetic dataset whose regional inequality traces an **N-shaped Kuznets curve**
in income — it rises, falls, then rises again at very high income.

```python
import expdpy as ex
from expdpy.data import load_kuznets

df = load_kuznets()
# The N-shaped regional Kuznets curve: regional inequality vs (log) GDP per capita
ex.explore_scatter_plot(
    df, x="log_gdp_pc", y="gini_regional", color="continent", size="population", loess=1
).fig
```

**Explore the panel structure.** Declare the panel once, then split a variable's variation
into across-unit (between) and over-time (within) parts, or trace every unit at once:

```python
df = ex.set_panel(load_kuznets(), entity="country", time="year")

ex.explore_xtsum_table(df, var=["gini_regional", "log_gdp_pc"]).gt   # within/between table
ex.explore_spaghetti_plot(df, var="gini_regional").fig              # one line per country
ex.explore_scatter_plot_within_between(df, x="log_gdp_pc", y="gini_regional").fig
```

**Run a regression and let it explain itself.** Two-way fixed effects, clustered standard
errors, a plain-language reading, and a coefficient plot:

```python
res = ex.analyze_regression_table(
    df,
    dvs="gini_regional",
    idvs=["log_gdp_pc", "log_gdp_pc_sq", "log_gdp_pc_cu"],
    feffects=["country", "year"],
    clusters=["country"],
)
print(res.interpret())            # plain-language, associational reading
ex.analyze_coefficient_plot(res)  # themed coefficient plot with confidence intervals
```

**Bring within estimates into a random-effects frame** with the correlated-random-effects
(Mundlak) estimator — its slope equals the fixed-effects estimate, and a joint test on the
unit-mean terms is the regression-form Hausman test:

```python
ex.analyze_cre_table(
    df, dv="gini_regional", idvs=["log_gdp_pc"], entity="country", time="year"
).etable
```

**Event study & staggered difference-in-differences** on the bundled treated panel:

```python
from expdpy.data import load_staggered_did

did = load_staggered_did()
ex.analyze_panel_view(did, unit="unit", time="year", cohort="cohort")   # treatment structure
ex.analyze_event_study(                                                  # dynamic effects
    did, outcome="outcome", unit="unit", time="year", cohort="cohort", estimator="did2s"
).fig
```

**Classic panel models and the Hausman test:**

```python
ex.analyze_panel_table(did, dv="outcome", idvs=["treated"], entity="unit", time="year").etable
print(ex.analyze_hausman_test(did, dv="outcome", idvs=["treated"], entity="unit", time="year").interpret())
```

**Learn as you go** — concept sandboxes and explainers:

```python
ex.learn_first_differences()      # first differences ≈ demeaning ≈ dummy variables
print(ex.explain("fixed_effects"))  # a concept explainer; ex.list_topics() lists them all
```

Launch the Explore app on this data, pre-configured to open on the curve:

```python
from expdpy.streamlit_app import ExploreApp
from expdpy.data import load_kuznets, load_kuznets_data_def, get_config

ExploreApp(load_kuznets(), df_def=load_kuznets_data_def(), config_list=get_config("kuznets"))
```

Head to [Explore](https://cmg777.github.io/expdpy/explore.html),
[Analyze](https://cmg777.github.io/expdpy/analyze.html) and
[Learn](https://cmg777.github.io/expdpy/learn.html) to see every function in action, the
[kuznets dataset](https://cmg777.github.io/expdpy/explanation/kuznets-dataset.html) page for
the data dictionary, or the [app guide](https://cmg777.github.io/expdpy/streamlit.html) to
launch the interactive apps.

## Documentation

Full documentation, tutorials, and the API reference live at
**https://cmg777.github.io/expdpy/**.

## Acknowledgements

expdpy began as a Python port of the excellent
[ExPanDaR](https://github.com/trr266/ExPanDaR) package by Joachim Gassen and the
TRR 266 Accounting for Transparency project, and its foundations remain deeply inspired by
that work. Over time it has grown well beyond the original — three no-code **Streamlit** apps;
fixest-style estimators (fixed effects, event study and staggered difference-in-differences)
with coefficient and Frisch–Waugh–Lovell plots; random-effects, correlated-random-effects and
Hausman panel models; a built-in pedagogy layer that interprets and explains results; a
restricted-AST expression evaluator with panel-aware `lag`/`lead`; and reproducible notebook /
script / data export — and it will keep evolving. We are grateful to the ExPanDaR authors;
please cite the original work when using `expdpy` in research (see
[`CITATION.cff`](CITATION.cff)).

## License

MIT — see [`LICENSE`](LICENSE).
