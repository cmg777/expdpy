# Project status

_Last updated: 2026-06-24._

**expdpy** is a Python library for interactive analysis of **panel and cross-sectional data** — a
port of the ExPanDaR R package — organized as three modules (**Explore**, **Analyze**, **Learn**),
three no-code Streamlit apps, and a Quarto + quartodoc documentation site
(<https://cmg777.github.io/expdpy/>).

Current version: **0.4.13** — on `main` and **released to PyPI**.

## Recently shipped — data-dictionary-driven readability

This batch makes the library read better by leaning on the data dictionary (`df_def`) while still
working without it:

- **Panel-aware descriptive table.** `explore_descriptive_table` now reflects the panel structure:
  when a `time` column is known it shows each statistic **by period** (first and last by default,
  `periods=` to override) under a spanning column header; otherwise it falls back to a flat table.
  The default statistics are **Mean, Std. dev., Median, Min., Max.**, rows are labelled from the
  dictionary, and the notes report the observation count and any variable with missing data. The
  result gains a tidy `.by_period` frame, while `.df` keeps all eight pooled statistics.
  **Breaking:** the old length-8 `digits` vector is replaced by `stats=` / `digits=` (scalar or
  per-statistic mapping) / `periods=`.
- **Histogram density overlays.** `explore_histogram` gains opt-in `kde=` and `normal=` flags that
  draw a kernel-density estimate and/or a normal curve on the Density scale (off by default; the
  Count/Density toggle hides them in Count view).
- **No more range sliders** on the time-series plots (`explore_trend_plot`,
  `explore_quantile_trend_plot`, `explore_spaghetti_plot`).
- **`df_def` everywhere.** Regression / estimation / CRE tables relabel their coefficient and
  dependent-variable rows from the dictionary (the tidy `.df` keeps raw term names); the panel
  estimators (`analyze_panel_table`, `analyze_hausman_test`, `analyze_cre_table`) and DiD views
  (`analyze_event_study`, `analyze_panel_view`) now resolve `entity` / `time` / `unit` from the
  declared panel, so those arguments can be omitted after `set_panel` / `set_labels`; and every
  function's docstring `Examples` now illustrate
  `df = ex.set_labels(load_kuznets(), load_kuznets_data_def(), set_panel=True)`.

An adversarial review of the diff surfaced and fixed two issues in the new descriptive table: a
tolerant-fallback guard that wrongly re-raised when a valid explicit id accompanied a
stored-missing one, and a period-range note that did not sort chronologically.

## Status of checks

- **Tests** — `pixi run pytest` green (439 passed, 4 skipped).
- **Lint / types** — `ruff check`, `ruff format --check` and `mypy src` clean.
- **Docs** — `pixi run -e docs docs-build` renders the full site with every example executing
  cleanly; the quickstart/feature notebooks were regenerated and are drift-check fresh.

## Open items / next steps

- **CI docs-build time.** The reference build executes every documented example. If CI gets slow,
  commit `docs/_freeze/` with `execute: freeze: auto` to cache execution between builds.
- **Typing nit.** A minor mypy warning in `tools/build_source_pages.py` (the heterogeneous
  `by_module` dict); `tools/` is outside CI's `mypy src` scope, so tidy when convenient.
