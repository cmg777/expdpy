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
cross-sectional data**. It pairs a set of composable analytical functions — that return
interactive [Plotly](https://plotly.com/python/) figures and publication-quality
[Great Tables](https://posit-dev.github.io/great-tables/) — with **two no-code `ExPdPy` web
apps** that bring the same workflow to the browser, no code required.

It is built on the modern Python data and econometrics stack:

- **[Plotly](https://plotly.com/python/)** — interactive figures
- **[pyfixest](https://github.com/py-econometrics/pyfixest)** — fixed-effects / clustered regressions
- **[Great Tables](https://posit-dev.github.io/great-tables/)** — publication-quality tables
- **[Streamlit](https://streamlit.io/)** and **[Shiny for Python](https://shiny.posit.co/py/)** — the two no-code `ExPdPy` apps

## Features

### Composable analytical functions

Descriptive, correlation and extreme-observation tables; histograms and category bar charts;
time trends and quantile trends; by-group bar, violin and trend views; scatter plots with an
optional LOESS smoother; and a missing-value heatmap across the panel. Each function takes a
`pandas` DataFrame and returns an interactive Plotly figure or a Great Tables object you can
drop straight into a notebook or report.

### Panel-aware econometrics

OLS with **multi-way fixed effects** and **clustered standard errors** via
[pyfixest](https://github.com/py-econometrics/pyfixest), publication-ready coefficient tables,
and **Frisch–Waugh–Lovell** partial-regression plots that show a single coefficient net of all
controls and fixed effects. Winsorize or truncate outliers with `treat_outliers`.

### Two no-code apps — Streamlit & Shiny

The same exploration workflow in two frontends: a sidebar **sample pipeline** (subset filters,
outlier treatment), component selection and ordering, and live, point-and-click analysis. The
[Streamlit app](https://cmg777.github.io/expdpy/streamlit.html) organises the components into a
multipage layout with native, sortable tables and deploys to
[Streamlit Community Cloud](https://streamlit.io/cloud) in one click; the
[Shiny app](https://cmg777.github.io/expdpy/shiny.html) stacks every component in one scrolling
view. See [Streamlit vs Shiny](https://cmg777.github.io/expdpy/explanation/streamlit-vs-shiny.html)
for a side-by-side comparison.

### Reproducibility & safety

Any in-app exploration exports to a **runnable bundle** — a Jupyter notebook, a `.py` script
and the prepared data (parquet) — that recreates every displayed result with `expdpy` calls.
Analysis configurations **save, load and interchange between the two apps**. New variables can
be defined live through a **restricted-AST expression evaluator** (never `eval`/`exec`) with
**panel-aware `lag`/`lead`** that shift within each cross-section.

### Bundled panel datasets

`expdpy.data` ships ready-to-explore panels — **`kuznets`** (the flagship N-shaped
Kuznets-curve demo) and `gapminder` — with `kuznets` shipping a preset configuration that
opens an app directly on the worked example. See the
[kuznets dataset](https://cmg777.github.io/expdpy/explanation/kuznets-dataset.html) page for
the data dictionary.

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

## At a glance

The lead example throughout the docs is the bundled `kuznets` panel (80 countries ×
2015–2025): a synthetic dataset, rich in control variables, whose regional inequality traces
an **N-shaped Kuznets curve** in income — it rises, falls, then rises again at very high
income.

```python
import expdpy as ex
from expdpy.data import load_kuznets

df = load_kuznets()
# The N-shaped regional Kuznets curve: regional inequality vs (log) GDP per capita
ex.prepare_scatter_plot(
    df, x="log_gdp_pc", y="gini_regional", color="continent", size="population", loess=1
)
```

Launch the same data in the interactive app, pre-configured to open on the curve:

```python
from expdpy.streamlit_app import ExPdPy   # or: from expdpy.app import ExPdPy
from expdpy.data import load_kuznets, load_kuznets_data_def, get_config

ExPdPy(load_kuznets(), df_def=load_kuznets_data_def(), config_list=get_config("kuznets"))
```

Head to the [Quickstart](https://cmg777.github.io/expdpy/quickstart.html) to see every
function in action, the
[kuznets dataset](https://cmg777.github.io/expdpy/explanation/kuznets-dataset.html) page for
the data dictionary, or the [Streamlit](https://cmg777.github.io/expdpy/streamlit.html) /
[Shiny](https://cmg777.github.io/expdpy/shiny.html) guides to launch the interactive apps.

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
