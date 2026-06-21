# CLAUDE.md

Guidance for working in this repository. Keep it current when commands or conventions change.

## Overview

**expdpy** is a Python library for interactive analysis of **panel and cross-sectional data**
— a port of the ExPanDaR R package — organized around three conceptual modules:

- **Explore** — descriptive/correlation/extreme tables, distributions, trends, by-group views,
  scatter, missing-value maps, and panel-aware views (within/between, spaghetti, panel
  structure, distribution/transition dynamics). Returns interactive **Plotly** figures and
  **Great Tables**.
- **Analyze** — OLS / fixed effects / clustered SEs (**pyfixest**), pooled/between/random
  effects + Hausman + CRE (**linearmodels**), FWL & coefficient plots, post-estimation, robust
  inference, and event-study / staggered DiD.
- **Learn** — concept explainers, runnable sandboxes, and a plain-language `.interpret()` on
  every result.

Three no-code **Streamlit** apps (one per module). `src/` layout, Python ≥ 3.10, managed with
**pixi**. Current version: 0.4.2.

## Commands

Everything runs through pixi. The `default` env is Python 3.12; lint/types/docs/R use their own
`-e <env>` (matching CI in `.github/workflows/ci.yml`).

```bash
pixi run pytest -q                          # fast test run (default env)
pixi run pytest tests/test_panel.py -q      # one file;  add -k <expr> for one test
pixi run test                               # full task: coverage + -n auto, excludes R parity
pixi run -e lint ruff check src tests       # lint
pixi run -e lint ruff format src tests      # format (CI uses `ruff format --check`)
pixi run -e lint mypy src                   # type check  (alias: pixi run typecheck)
pixi run -e lint pre-commit run --all-files # all pre-commit hooks
pixi run -e r test-r                        # R numerical-parity tests (needs R + ExPanDaR)
pixi run -e docs docs-build                 # build the Quarto docs site
pixi run streamlit run app_explore.py       # run an app (also app_analyze.py / app_learn.py)
python tools/build_<name>.py                # regenerate a bundled dataset (kuznets/firms/...)
```

## Architecture

Each module is a set of **flat modules** under `src/expdpy/` (no per-module subpackage). The
public API is curated in `src/expdpy/__init__.py` (`__all__` grouped by module).

- **Explore**: `tables.py`, `distributions.py`, `correlation.py`, `trends.py`, `by_group.py`,
  `missing.py`, `scatter.py`, `outliers.py`; panel-aware `panel_summary.py` (xtsum +
  within/between scatter), `spaghetti.py`, `panel_structure.py`, `dynamics.py`. Helpers:
  `_panel.py` (`set_panel`/`resolve_panel`), `_panel_math.py` (the xtsum within/between
  decomposition).
- **Analyze**: `regression.py`, `estimation.py`, `fwl.py`, `coefplot.py`, `panel_models.py`,
  `cre.py`, `postestimation.py`, `inference.py`, `did.py`. Shared estimation engine in
  `_estimation/` (pyfixest wrapper: `_spec`, `_formula`, `_vcov`, `_fit`, `_tidy`, `_results`).
- **Learn**: `sandbox.py` + `pedagogy/` (`_registry` `Explainer`/`explain`/`list_topics`,
  `_interpret`, `_mixin` `Interpretable`, `_format`, and `_text/*` topic registrations).
- **Infra**: `_types.py` (frozen result dataclasses), `_theme.py` (shared Plotly theme),
  `_validation.py`, `_corr.py`; `data/` (bundled parquets + loaders); `streamlit_app/` (the
  three apps — subprocess bundle handoff in `_handoff`/`_launcher`/`_context`, the sample
  pipeline, and `_pages`).

## Conventions (follow these)

- **Panel vocabulary is `entity` (unit) + `time`** everywhere — function params, launcher
  kwargs, and the `df_def` `type` metadata. `set_panel(df, entity=, time=)` declares the panel
  once (stored on `df.attrs`); explicit per-call args always win (`resolve_panel`). New
  functions take `entity`/`time` keyword-only.
- **Public functions are module-prefixed** (renamed in 0.4.2, replacing the old `prepare_*` /
  `sandbox_*` names): `explore_*` (Explore), `analyze_*` (Analyze), `learn_*` (Learn).
  Plotly-figure functions end in `_plot`, Great-Tables functions in `_table`, and scope
  qualifiers go last (e.g. `explore_violin_plot_by_group`). The cross-cutting helpers
  `set_panel` / `resolve_panel` / `treat_outliers` / `explain` / `list_topics` are
  **unprefixed** and grouped as "Utilities" in `__all__`.
- **Every `explore_*` / `analyze_*` / `learn_*` function returns a frozen result dataclass**
  (defined in `_types.py`) exposing `.df` plus `.fig` (Plotly) or `.gt` (Great Tables) — never
  a bare figure. Many mix in `Interpretable`, adding
  `.interpret()` / `.explain()` / `.tidy()` / `.glance()`.
- **`.interpret()` describes associations, never causation** — the words "causes" / "effect of"
  must not appear, and the text ends with the shared `_ASSOC_NOTE` pointing to
  `correlation_vs_causation`. Interpretation logic lives in `pedagogy/_interpret.py`, which is
  **duck-typed and must not import `_types`** (avoids an import cycle).
- **Concept explainers** register at import time via `register_topic(Explainer(...))` in
  `pedagogy/_text/<area>.py` (wired through `pedagogy/_text/__init__.py`).
- **Plotting**: build Plotly `go.Figure`s and style them with `apply_default_layout`,
  `color_for`, and the shared scales from `_theme.py`; never call `.show()`. Tables use
  `great_tables.GT`.
- **Estimation goes through `_estimation/` (pyfixest) and linearmodels** — don't hand-roll
  OLS/FE/clustering.
- **Code style**: `from __future__ import annotations`; PEP 604 unions (`str | None`); NumPy
  docstrings (ruff `D` is enforced — public functions need them); line length 88; private
  modules are `_`-prefixed; `__all__` is grouped by topic, not sorted (RUF022 ignored).
- **Determinism / R goldens**: numeric results are cross-checked against R
  (`tests/fixtures/goldens.json`, generated by `make_goldens.R`; the shared deterministic panel
  is `tests/fixtures/sample.csv`). Do **not** regenerate the bundled data parquets casually — it
  drifts the goldens; prefer surgical edits to `df_def` metadata.

## Testing

- pytest markers: `against_r` (R parity, **excluded by default** via `-m 'not against_r'`),
  `streamlit`, `panel`. Streamlit pages are tested with `streamlit.testing.v1.AppTest`.
- New panel-exploration functions are covered in `tests/test_panel.py` (incl. known-answer toys
  and direct comparisons against pandas).

## Data & apps

- Bundled datasets (in `expdpy.data`): `kuznets` (flagship N-shaped curve), `gapminder`,
  `staggered_did` (event study / DiD), `firms` (a small **unbalanced** panel for the
  structure/transition/persistence views). Each has a `load_*()` + `load_*_data_def()`.
- `df_def.type ∈ {entity, time, factor, logical, numeric}`. Build scripts live in `tools/`.
- Launch apps in-process with `ExploreApp(df, entity=, time=, df_def=...)` /
  `AnalyzeApp` / `LearnApp` (kwargs are `entity`/`time`, renamed from `cs_id`/`ts_id` in 0.4.1).

## Versioning

Version bumps are **gradual by default**: increment the patch component by **+0.0.1** (e.g.
0.4.2 → 0.4.3) on any releasable change — you don't need to be asked. Make a larger jump
(minor `+0.1.0` / major `+1.0.0`) **only when the user explicitly asks**; a breaking change on
its own does *not* justify a bigger bump. Keep the version in sync across `pyproject.toml`
(`[project] version`), `src/expdpy/__init__.py` (`__version__`), and the "Current version"
line in the Overview above; for a user-facing release also add a dated `## <version>` entry to
`docs/changelog.qmd` and update the install pins in `README.md`.

## Gotchas

- The `ruff-format` pre-commit hook rewrites files and **aborts the first commit**; re-stage
  (`git add -A`) and commit again.
- `pixi run <task>` uses the `default` (py312) env; lint/mypy/docs/R parity each need their own
  `-e lint` / `-e docs` / `-e r`.
