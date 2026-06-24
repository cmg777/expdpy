# Project status

_Last updated: 2026-06-24._

**expdpy** is a Python library for interactive analysis of **panel and cross-sectional data** — a
port of the ExPanDaR R package — organized as three modules (**Explore**, **Analyze**, **Learn**),
three no-code Streamlit apps, and a Quarto + quartodoc documentation site
(<https://cmg777.github.io/expdpy/>).

Current version: **0.4.12** — on `main`, **not yet released to PyPI** (this batch is
documentation/tooling-only).

## Recently shipped — API reference documentation overhaul

The API reference now presents each function like the splot reference:

- **Live example output.** Every documented function's docstring `Examples` are executed at
  docs-build time, so the real **interactive Plotly figure / Great Table / DataFrame** renders
  directly below the code on its reference page.
- **Source code on the site.** Each reference page carries a **`[source]`** link to a generated,
  splot-style `reference/modules/<module>.qmd` page that lists the full, syntax-highlighted
  module source, anchored at the function (with a "↩ docs" back-link and a GitHub link).
- **Scannable index.** The reference index shows a brief, splot-style argument signature beside
  each function name (`name(req[, opt, …])`) and drops the distracting link underline.
- **Build tooling.** Two steps — `tools/build_source_pages.py` and
  `tools/build_reference_enrichment.py` — run inside `docs-build` between `quartodoc build` and
  `quarto render`. Their pure helpers are unit-tested in `tests/test_docs_tooling.py`.
- **`write-function` skill updated.** New functions now ship documentation that works under this
  workflow: registering a function in `docs/_quarto.yml` quartodoc `contents` is a **core** step,
  and docstring `Examples` must be self-contained and runnable (they execute at build).

Two latent docstring bugs surfaced by the now-executed examples were fixed:
`explore_spaghetti_plot` was missing `time=`, and `explain` used `>>>` doctest style.

## Status of checks

- **Tests** — `pixi run pytest` green, including `tests/test_docs_tooling.py`.
- **Lint / types** — `ruff check` and `mypy src` clean.
- **Docs** — `pixi run -e docs docs-build` renders the full site (reference + source pages) with
  every example executing cleanly.

## Open items / next steps

- **Release.** 0.4.12 is documentation/tooling-only, so no PyPI release was cut. Tag and publish
  with `gh release create v0.4.12 --target main ...` if/when a release is wanted.
- **CI docs-build time.** The reference build now executes ~40 examples. If CI gets slow, commit
  `docs/_freeze/` with `execute: freeze: auto` to cache execution between builds.
- **Typing nit.** A minor mypy warning in `tools/build_source_pages.py` (the heterogeneous
  `by_module` dict); `tools/` is outside CI's `mypy src` scope, so tidy when convenient.
