# expdpy

<!-- badges: start -->
[![CI](https://github.com/cmg777/expdpy/actions/workflows/ci.yml/badge.svg)](https://github.com/cmg777/expdpy/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/cmg777/expdpy/branch/main/graph/badge.svg)](https://codecov.io/gh/cmg777/expdpy)
[![Docs](https://github.com/cmg777/expdpy/actions/workflows/docs.yml/badge.svg)](https://cmg777.github.io/expdpy/)
[![GitHub release](https://img.shields.io/github/v/release/cmg777/expdpy?include_prereleases&label=release)](https://github.com/cmg777/expdpy/releases)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://github.com/cmg777/expdpy)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
<!-- badges: end -->

**expdpy** is a Python port of the [ExPanDaR](https://github.com/trr266/ExPanDaR) R
package (Joachim Gassen, TRR 266). It helps you explore panel and cross-sectional data
through a set of composable analytical functions and an interactive web app — built on
the modern Python data/econometrics stack:

- **[Plotly](https://plotly.com/python/)** for interactive figures
- **[pyfixest](https://github.com/py-econometrics/pyfixest)** for fixed-effects / clustered regressions
- **[Great Tables](https://posit-dev.github.io/great-tables/)** for publication-quality tables
- **[Shiny for Python](https://shiny.posit.co/py/)** for the no-code `ExPanD` app

## Installation

The package is not on PyPI yet — install the latest version straight from GitHub:

```bash
# Core analytical functions:
pip install "git+https://github.com/cmg777/expdpy.git"

# ...with the interactive ExPanD app:
pip install "expdpy[app] @ git+https://github.com/cmg777/expdpy.git"
```

Using [uv](https://docs.astral.sh/uv/):

```bash
uv pip install "git+https://github.com/cmg777/expdpy.git"
uv pip install "expdpy[app] @ git+https://github.com/cmg777/expdpy.git"
```

Pin to a branch, tag, or commit for reproducible installs:

```bash
pip install "git+https://github.com/cmg777/expdpy.git@main"
# pip install "git+https://github.com/cmg777/expdpy.git@v0.1.0"   # once a release is tagged
```

> **Coming soon (PyPI):** once published, `pip install expdpy` /
> `pip install "expdpy[app]"` will work directly.

## Quickstart

```python
import expdpy as ex
from expdpy.data import load_gapminder

df = load_gapminder()  # country x year panel (life expectancy, population, GDP per capita)

# Descriptive statistics (returns a DataFrame + a Great Tables object)
desc = ex.prepare_descriptive_table(df[["lifeExp", "gdpPercap", "pop"]])
desc.gt  # renders in a notebook

# Winsorize outliers at the 1st/99th percentile
clean = ex.treat_outliers(df[["lifeExp", "gdpPercap", "pop"]], percentile=0.01)

# Correlation table (Pearson above, Spearman below the diagonal)
corr = ex.prepare_correlation_table(clean)
corr.gt

# Time trend of a variable across the panel
ex.prepare_trend_graph(df, ts_id="year", var=["lifeExp"]).fig.show()

# Regression with country fixed effects and clustered standard errors (pyfixest)
reg = ex.prepare_regression_table(
    df, dvs="lifeExp", idvs=["gdpPercap"], feffects=["country"], clusters=["country"]
)
reg.etable
```

Launch the interactive app:

```python
from expdpy.app import ExPanD
from expdpy.data import load_gapminder, load_gapminder_data_def

ExPanD(load_gapminder(), df_def=load_gapminder_data_def())
```

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
| `treat_outliers` | Winsorize or truncate outliers |
| `ExPanD` | Interactive Shiny-for-Python exploration app |

## Documentation

Full documentation, tutorials, and the API reference live at
**https://cmg777.github.io/expdpy/**.

## Acknowledgements

This package is a Python port of the excellent
[ExPanDaR](https://github.com/trr266/ExPanDaR) package by Joachim Gassen and the
TRR 266 Accounting for Transparency project. Please cite the original work when using
`expdpy` in research (see [`CITATION.cff`](CITATION.cff)).

## License

MIT — see [`LICENSE`](LICENSE).
