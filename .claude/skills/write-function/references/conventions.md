# expdpy conventions — the rulebook for a generated function

Distilled from `CLAUDE.md`. A function produced by this skill must obey every rule here. When
in doubt, open `CLAUDE.md` and the exemplar `src/expdpy/convergence.py`.

## Naming

- Public functions are **module-prefixed**: `explore_*` (Explore), `analyze_*` (Analyze),
  `learn_*` (Learn). Plotly-figure functions end in `_plot`, Great-Tables functions in
  `_table`, and scope qualifiers go last (e.g. `explore_violin_plot_by_group`).
- The cross-cutting helpers `set_panel` / `resolve_panel` / `set_labels` / `resolve_label` /
  `treat_outliers` / `explain` / `list_topics` are **unprefixed** ("Utilities").
- New keyword-only panel params are `entity` / `time` (never `cs_id`/`ts_id`). Declare the
  panel once with `set_panel(df, entity=, time=)`; `resolve_panel` lets explicit per-call args
  win over `df.attrs`.
- Private modules and helpers are `_`-prefixed.

## Return type

- **Every** `explore_*` / `analyze_*` / `learn_*` function returns a **frozen result
  dataclass** defined in `src/expdpy/_types.py` — never a bare figure or frame.
- The dataclass exposes `.df` plus `.fig` (Plotly) and/or `.gt` (Great Tables), plus named
  scalar statistics where relevant.
- Most mix in `Interpretable` (from `expdpy.pedagogy`), adding `.interpret()` / `.explain()` /
  `.tidy()` / `.glance()`. Implement the ones that are meaningful; the base raises a clear
  `NotImplementedError` for the rest.
- `_types.py` imports `interpret_*` functions and `_explain` at module top; the dataclass
  methods delegate to them. The dataclass is added to `_types.__all__` (grouped, not sorted).

## `.interpret()` describes associations, never causation

- Interpretation text **must not** contain the word "causes" or the phrase "effect of".
- It ends with the shared `_ASSOC_NOTE` (pointing to `explain('correlation_vs_causation')`)
  whenever it describes a relationship between variables.
- Interpretation logic lives in `pedagogy/_interpret.py`, is **duck-typed** (reads `.df` /
  `.models` / scalar fields), and **must not import `_types`** (avoids an import cycle).
- Reuse the formatting helpers in `pedagogy/_format.py`: `fmt_num`, `direction_word`,
  `significance_phrase`, `sign_word`, `is_significant`.

## Concept explainers

- Register at import time via `register_topic(Explainer(...))` in
  `pedagogy/_text/<area>.py`, wired through `pedagogy/_text/__init__.py`.
- `Explainer` fields: `topic`, `title`, `what`, `when_to_use`, `caveats` (tuple), `see_also`
  (tuple of existing topic keys), `references` (tuple). `aliases=` is a `register_topic` kwarg.
- Confirm `see_also` keys exist (`grep 'topic="' src/expdpy/pedagogy/_text/`).

## Estimation engine

- OLS / fixed effects / clustered SEs go through `_estimation/` (the pyfixest wrapper) or
  through `pf.feols` directly with the shared small-sample correction `_SSC` (re-exported from
  `regression.py`). Pooled/between/random effects + Hausman + CRE go through linearmodels.
- **Never hand-roll** OLS/FE/clustering for the *reported* coefficients. (statsmodels is fine
  for an auxiliary fit/confidence band on a residual scatter, as in `fwl.py`.)
- Reuse `_residualize` (`fwl.py`) for Frisch–Waugh–Lovell partialling-out.
- Default vcov is `iid`; expose a `vcov=` arg (e.g. `"hetero"` for HC1) when robust SEs matter.

## Plotting

- Build Plotly `go.Figure`s and style them with `apply_default_layout`, `color_for`, and the
  shared scales (`SEQUENTIAL_SCALE`, `DIVERGING_SCALE`) from `_theme.py`. Never call `.show()`.
- For entity-aware hover, set `customdata=<entity series>` and a `hovertemplate` using
  `%{customdata}` (see `panel_summary.py` / `convergence.py`).
- Tables use `great_tables.GT` (`GT(df, rowname_col=, groupname_col=).tab_header().fmt_*()`).

## Defensive validation (do this first, with clear errors)

- `df = ensure_dataframe(df)` (`_validation.py`).
- `entity, time = resolve_panel(df, entity, time, require_entity=True, require_time=True)`
  for panel functions.
- Missing columns ⇒ `KeyError`; non-numeric focal/control ⇒ `TypeError`; too-few complete-case
  rows ⇒ `ValueError`; degenerate input such as (near-)zero variance in the regressor ⇒
  `ValueError` stating the estimate is "not identified". Drop NaNs explicitly and, where it
  matters, report what was dropped rather than silently truncating.

## Code style

- `from __future__ import annotations`; PEP 604 unions (`str | None`); **NumPy docstrings**
  (ruff `D` is enforced — public functions need full docstrings with `Parameters`, `Returns`,
  and a `Notes`/summary stating the math); line length 88.
- Keep ASCII in code/strings: avoid the Unicode minus `−` (use `-`) and Greek letters in
  source strings (write `rho`, `lambda`), or ruff `RUF001/RUF002` will flag them. Math symbols
  are fine in Markdown docs and notebooks.
- `__all__` is grouped by topic, not sorted (RUF022 is ignored).
- Avoid the single-element slice `[0]` on a generator/list comprehension (`RUF015`); use
  `next(...)`.

## Determinism & R goldens

- Numeric results may be cross-checked against R (`tests/fixtures/goldens.json`, generated by
  `make_goldens.R`; shared deterministic panel `tests/fixtures/sample.csv`). R-parity tests
  carry the `against_r` marker and are excluded by default.
- Do **not** regenerate the bundled data parquets casually — it drifts the goldens.

## Versioning

- Bump the patch component **+0.0.1** on any releasable change (no need to ask). Keep the
  version in sync across `pyproject.toml`, `src/expdpy/__init__.py` (`__version__`), and the
  CLAUDE.md "Current version" line; add a dated `## <version>` entry to `docs/changelog.qmd`
  and update README install pins.

## Quality gate (commands)

- `pixi run -e lint ruff format src tests` / `pixi run -e lint ruff check src tests`
- `pixi run -e lint mypy src`
- `pixi run pytest -q` (fast) / `pixi run pytest tests/test_<fn>.py -q -k <expr>`
- Docs/notebooks: `pixi run -e docs python tools/build_quickstart_notebook.py`,
  `pixi run -e docs docs-build`.
