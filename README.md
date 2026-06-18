<p align="center">
  <img src="https://raw.githubusercontent.com/cmg777/expdpy/main/docs/images/logo.png" alt="expdpy logo" width="96" height="96">
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/cmg777/expdpy/main/docs/images/hero.webp" alt="expdpy — A Python library to explore panel data interactively" width="100%">
</p>

<!-- badges: start -->
[![CI](https://github.com/cmg777/expdpy/actions/workflows/ci.yml/badge.svg)](https://github.com/cmg777/expdpy/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/cmg777/expdpy/branch/main/graph/badge.svg)](https://codecov.io/gh/cmg777/expdpy)
[![Docs](https://github.com/cmg777/expdpy/actions/workflows/docs.yml/badge.svg)](https://cmg777.github.io/expdpy/)
[![GitHub release](https://img.shields.io/github/v/release/cmg777/expdpy?include_prereleases&label=release)](https://github.com/cmg777/expdpy/releases)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://github.com/cmg777/expdpy)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
<!-- badges: end -->

**expdpy** is a Python library for interactive, exploratory analysis of **panel and
cross-sectional data**. It helps you explore your data through a set of composable
analytical functions and two interactive, no-code `ExPdPy` web apps (Streamlit and Shiny) —
built on the modern Python data/econometrics stack:

- **[Plotly](https://plotly.com/python/)** for interactive figures
- **[pyfixest](https://github.com/py-econometrics/pyfixest)** for fixed-effects / clustered regressions
- **[Great Tables](https://posit-dev.github.io/great-tables/)** for publication-quality tables
- **[Streamlit](https://streamlit.io/)** and **[Shiny for Python](https://shiny.posit.co/py/)** for the two no-code `ExPdPy` apps

## Installation

The package is not on PyPI yet — install the latest version straight from GitHub:

```bash
# Core analytical functions:
pip install "git+https://github.com/cmg777/expdpy.git"

# ...with the interactive ExPdPy app (Streamlit):
pip install "expdpy[streamlit] @ git+https://github.com/cmg777/expdpy.git"

# ...with the interactive ExPdPy app (Shiny):
pip install "expdpy[app] @ git+https://github.com/cmg777/expdpy.git"
```

Using [uv](https://docs.astral.sh/uv/):

```bash
uv pip install "git+https://github.com/cmg777/expdpy.git"
uv pip install "expdpy[streamlit] @ git+https://github.com/cmg777/expdpy.git"    # Streamlit
uv pip install "expdpy[app] @ git+https://github.com/cmg777/expdpy.git"          # Shiny
```

Pin to a branch, tag, or commit for reproducible installs:

```bash
pip install "git+https://github.com/cmg777/expdpy.git@main"
# pip install "git+https://github.com/cmg777/expdpy.git@v0.1.0"   # once a release is tagged
```

> **Coming soon (PyPI):** once published, `pip install expdpy` /
> `pip install "expdpy[streamlit]"` / `pip install "expdpy[app]"` will work directly.

## Quickstart

```python
import expdpy as ex
from expdpy.data import load_kuznets

df = load_kuznets()  # 80 countries x 2015-2025; an N-shaped regional Kuznets curve

# Descriptive statistics (returns a DataFrame + a Great Tables object)
desc = ex.prepare_descriptive_table(df[["gini_regional", "gdp_pc", "school_enrollment"]])
desc.gt  # renders in a notebook

# Winsorize outliers at the 1st/99th percentile
clean = ex.treat_outliers(df[["gini_regional", "gdp_pc", "population", "area"]], percentile=0.01)

# Correlation table (Pearson above, Spearman below the diagonal)
corr = ex.prepare_correlation_table(clean)
corr.gt

# Time trend of a variable across the panel
ex.prepare_trend_graph(df, ts_id="year", var=["gini_regional"]).fig.show()

# The N-shaped Kuznets curve: regional inequality vs (log) GDP per capita
ex.prepare_scatter_plot(
    df, x="log_gdp_pc", y="gini_regional", color="continent", size="population", loess=1
).show()

# Cubic regression recovers the N (significant positive cubic term), controlling for
# country and year (two-way) fixed effects, with SEs clustered by country
reg = ex.prepare_regression_table(
    df,
    dvs="gini_regional",
    idvs=["log_gdp_pc", "log_gdp_pc_sq", "log_gdp_pc_cu"],
    feffects=["country", "year"],
    clusters=["country"],
)
reg.etable

# Frisch-Waugh-Lovell plot: the partial relationship between gini and log GDP per capita,
# net of the other terms AND the country and year fixed effects (the fitted slope equals
# the coefficient)
ex.prepare_fwl_plot(
    df,
    dv="gini_regional",
    var="log_gdp_pc",
    controls=["log_gdp_pc_sq", "log_gdp_pc_cu"],
    feffects=["country", "year"],
    clusters=["country"],
).fig.show()
```

Launch the interactive app — **Streamlit** — pre-configured to open on the N-shaped curve
(multipage UI with native tables):

```python
from expdpy.streamlit_app import ExPdPy
from expdpy.data import load_kuznets, load_kuznets_data_def, get_config

ExPdPy(load_kuznets(), df_def=load_kuznets_data_def(), config_list=get_config("kuznets"))
```

…or the **Shiny** version (same data, a single-window card stack):

```python
from expdpy.app import ExPdPy
from expdpy.data import load_kuznets, load_kuznets_data_def, get_config

ExPdPy(load_kuznets(), df_def=load_kuznets_data_def(), config_list=get_config("kuznets"))
```

To deploy on [Streamlit Community Cloud](https://streamlit.io/cloud) (or run it without
passing data), point Streamlit at the bundled entry script — it starts with a dataset picker
(defaulting to Kuznets) and an upload box:

```bash
streamlit run streamlit_app.py
```

## Example datasets

`expdpy.data` ships several panel datasets for teaching and demos:

| Loader | Panel | Notes |
|---|---|---|
| `load_kuznets` | 80 countries × 2015–2025 | **The default showcase.** Synthetic; rich in control variables; regional inequality traces an **N-shaped** Kuznets curve in income (rises, falls, then rises again at very high income). Built to exercise every feature and ships a preset config (`get_config("kuznets")`) that opens the app directly on the curve. |
| `load_gapminder` | country × year | Life expectancy, population, GDP per capita. |

## Functions

| Function | Purpose |
|---|---|
| `prepare_descriptive_table` | Summary statistics (N, mean, sd, quartiles) |
| `prepare_correlation_table` / `prepare_correlation_graph` | Pearson/Spearman correlations (table / heatmap) |
| `prepare_ext_obs_table` | Extreme (top/bottom) observations |
| `prepare_trend_graph` / `prepare_quantile_trend_graph` | Variable means / quantiles over time |
| `prepare_by_group_bar_graph` / `_trend_graph` / `_violin_graph` | Statistics by group |
| `prepare_histogram` / `prepare_bar_chart` | Distribution / category counts |
| `prepare_missing_values_graph` | Missing-value heatmap across the panel |
| `prepare_scatter_plot` | Scatter with optional size/color/LOESS |
| `prepare_regression_table` | OLS with fixed effects + clustered SEs (pyfixest) |
| `prepare_fwl_plot` | Frisch-Waugh-Lovell partial-regression plot (residualized on controls + fixed effects) |
| `treat_outliers` | Winsorize or truncate outliers |
| `expdpy.streamlit_app.ExPdPy` | Interactive Streamlit exploration app (multipage, native tables) |
| `expdpy.app.ExPdPy` | Interactive Shiny-for-Python exploration app |

## Documentation

Full documentation, tutorials, and the API reference live at
**https://cmg777.github.io/expdpy/**.

## Acknowledgements

expdpy began as a Python port of the excellent
[ExPanDaR](https://github.com/trr266/ExPanDaR) package by Joachim Gassen and the
TRR 266 Accounting for Transparency project, and its foundations remain deeply inspired by
that work. Over time it has grown functionality beyond the original — two interactive
frontends (Streamlit **and** Shiny), a restricted-AST expression evaluator with panel-aware
`lag`/`lead`, pyfixest-based fixed-effects regressions with Frisch–Waugh–Lovell plots, and
reproducible notebook / script / data export — and it will keep evolving. We are grateful to
the ExPanDaR authors; please cite the original work when using `expdpy` in research (see
[`CITATION.cff`](CITATION.cff)).

## License

MIT — see [`LICENSE`](LICENSE).
