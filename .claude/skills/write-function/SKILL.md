---
name: write-function
description: >-
  This skill should be used whenever the user wants to create, add, write, scaffold, or
  implement a new computational/econometrics function in the expdpy library — e.g. "write a
  function for X", "add an analyze_/explore_/learn_ function", "implement an estimator",
  "create a new panel-data routine", "scaffold a convergence/regression/decomposition
  function", or "build out a new statistic". It produces production-ready functions to
  expdpy's standard: module-prefixed public functions returning frozen Interpretable result
  dataclasses, pyfixest/linearmodels estimation, transparent NumPy docstrings, defensive input
  validation, and a fault-tolerant pytest suite that includes a mathematical-validity test
  against a known baseline. Use this skill even when the user does not name expdpy explicitly
  but is clearly asking for a new analysis function in this repo, and even for "just a small
  helper" — the rigor and wiring are the point.
---

# write-function

Create production-ready econometrics/data-science functions for **expdpy** to the same
standard as the worked exemplar `analyze_beta_convergence`
(`src/expdpy/convergence.py` + `tests/test_convergence.py`). The value of this skill is
**rigor and consistency**: every function ships mathematically transparent, defensively
validated, fully tested, and correctly wired into the public API — without re-deriving the
repo's conventions each time.

Errors in scientific software cascade. Favor explicitness over cleverness, validate inputs
before computing, never black-box the math, and prove correctness against a known baseline.

## Interaction protocol (blocking — do this first)

Do **not** write any code until ~95% confident the requirements are understood. Never assume
the mathematical approach, the estimator, the data structure, or the outputs. Ask the user a
**numbered list of clarifying questions**, then wait for answers. Prefer the `AskUserQuestion`
tool when running interactively.

Cover at least these, skipping only what the user already answered:

1. **Mathematical objective / estimator.** What exactly is computed or estimated (e.g. an ATE,
   a β-convergence slope, a variance decomposition, a spatial-FE model)? What is the precise
   formula, and what are the explicit assumptions? Is it associational or causal in framing?
2. **Inputs.** Data shape (panel / cross-section / time-series), the required columns and
   parameters, their types, expected dimensions, and constraints (e.g. balanced vs unbalanced,
   minimum N, allowed ranges). What entity/time columns apply?
3. **Outputs.** What the result must expose: `.df`, a Plotly `.fig` and/or a Great-Tables
   `.gt`, named scalar statistics, and what `.interpret()` should say.
4. **Known baseline for the math-validity test.** A closed-form result, a simulated DGP with
   known parameters (the strongest option — see the AR(1) recovery in `tests/test_convergence.py`),
   a trusted reference implementation, or an R-parity golden. This is mandatory; without a
   baseline the function cannot be proven correct.
5. **Integration layers (adaptive scope).** The core is always produced (function module +
   docstring + validation + result dataclass + pytest + public export). Ask which extras to
   add: a pedagogy concept explainer + `.interpret()`, a runnable `learn_*` sandbox, a
   Streamlit app tab, and a paired Quarto→Colab notebook (extended guide + verification harness).
6. **Public name + module.** Must obey the naming rule (`analyze_*` / `explore_*` / `learn_*`,
   `_plot` / `_table` suffixes, scope qualifiers last). Confirm the new module filename.

When confident, restate the locked decisions in one short list, then proceed.

## Workflow

Read `references/conventions.md` before implementing — it is the rulebook a generated function
must obey. Read `references/wiring-checklist.md` before Phase 4 and `references/templates.md`
before Phases 2–3.

### Phase 1 — Explore & reuse
Search `src/expdpy/` for helpers to reuse before writing anything new — reinventing existing
machinery is the most common failure. Common reuse points: `resolve_panel`/`set_panel`
(`_panel.py`), `resolve_label` (`_labels.py`), `_residualize` + `_SSC`/`_as_list`
(`fwl.py`/`regression.py`), the `_estimation/` pyfixest wrapper, `apply_default_layout`/
`color_for`/scales (`_theme.py`), `ensure_dataframe` (`_validation.py`), and the pedagogy
registry. Read the exemplar `src/expdpy/convergence.py` to anchor the structure.

### Phase 2 — Implement the function
Follow the function skeleton in `references/templates.md`. Required shape:
- `from __future__ import annotations`; PEP 604 unions; private helpers `_`-prefixed; line
  length 88; module docstring explaining the method and stating that the variable is used as
  supplied where relevant.
- **Defensive validation first**, before any computation, with clear messages: `ensure_dataframe`;
  `resolve_panel(..., require_entity=, require_time=)` when panel; check columns exist
  (`KeyError`), numeric dtype (`TypeError`), enough complete-case rows, and guard degenerate
  inputs (e.g. (near-)zero variance ⇒ `ValueError` "not identified"). Drop/track NaNs explicitly.
- **Transparent math**: the NumPy docstring's `Notes`/summary states the estimand, the formula,
  and the assumptions — no black-boxing. Document every parameter (type, dims, constraints) and
  every returned field.
- **Runnable docstring `Examples`**: the reference page **executes** them at docs-build time, so
  make the example self-contained — load a bundled `expdpy.data` dataset and use real
  columns/keys; plain ` ```python ` fences, never `>>>` doctest prompts (`references/templates.md`).
- **Estimation through the engine**: pyfixest (`pf.feols` with the shared `_SSC`) or
  linearmodels — never hand-roll OLS/FE/clustering. Plot with `_theme` helpers; never `.show()`.
- **Return a frozen result dataclass** that mixes in `Interpretable` (defined in `_types.py`),
  exposing `.df` plus `.fig`/`.gt` and named scalars, with `.interpret()`/`.explain()`/
  `.tidy()`/`.glance()` as meaningful.

### Phase 3 — Generate the test suite
Mirror `tests/test_convergence.py` (see `references/templates.md`). Include all four categories:
- A **known-answer DGP / closed-form helper** with parameters whose true result is computable.
- **Expected-use** tests asserting the documented behavior and result surface (figs present,
  hover/customdata, tables populated, `.interpret()` association-only with no "causes"/"effect
  of").
- **Edge cases**: missing entity/time, non-numeric inputs, NaNs, collinearity, zero variance,
  too-few-observations — assert the right exception type and message.
- The **mathematical-validity** test: assert the function recovers the baseline within a
  tolerance justified by the injected noise. Apply the correct pytest markers (`panel`, etc.).

### Phase 4 — Wire the chosen layers
Use `references/wiring-checklist.md` for the exact files per layer. Core always: result
dataclass in `_types.py` (+ `__all__`), exports in `__init__.py` (`__all__` grouped by module),
`interpret_*` import wiring if Interpretable, and registering `<fn>` in `docs/_quarto.yml`
quartodoc `contents` (the step that creates the reference page — with its auto-executed example
output and `[source]` link). Optional layers only if requested: the
pedagogy explainer (`pedagogy/_text/<topic>.py`) + `interpret_*` (`pedagogy/_interpret.py`,
association-only, ending in `_ASSOC_NOTE`, no "causes"/"effect of") + both `__all__`s +
`pedagogy/__init__.py`; a `learn_*` sandbox; a Streamlit tab; a Quarto→Colab notebook
(`docs/<fn>.qmd` + `MODULES` in `tools/build_quickstart_notebook.py` + `docs/_quarto.yml` nav).
Always finish with the release bump (`pyproject.toml`, `__init__.py __version__`, CLAUDE.md
"Current version", a dated `docs/changelog.qmd` entry, README pins) — patch +0.0.1 by default.

### Phase 5 — Verify
Run the quality gate, then the adversarial review:
- `bash .claude/skills/write-function/scripts/verify.sh [tests/test_<fn>.py]` — ruff
  format + check, mypy, pytest. Fix everything it surfaces.
- **Docs check** — render the new reference page so its docstring example actually executes
  (this is what the CI docs build does, and a broken example turns it red):
  `pixi run -e docs bash -c "quartodoc build --config docs/_quarto.yml >/dev/null && python
  tools/build_source_pages.py && python tools/build_reference_enrichment.py && quarto render
  docs/reference/<fn>.qmd"`. A clean render confirms the example runs and the `[source]`
  link/anchor resolve. Fix any failure by making the example self-contained.
- If a notebook was added, regenerate and run the drift-check
  (`pixi run -e docs python tools/build_quickstart_notebook.py`; then `git diff` ignoring the
  build-timestamp line) and render the `.qmd` so its asserts execute.
- Run the adversarial multi-agent review and fix every independently-confirmed finding:
  `Workflow({ scriptPath: ".claude/skills/write-function/scripts/review_workflow.js",
  args: { files: ["src/expdpy/<fn>.py", "tests/test_<fn>.py"], spec: "<one-paragraph spec of
  the estimand, assumptions, and conventions>" } })`.

Report results faithfully: if a test fails or a step was skipped, say so with the output.

## Resources

- `references/conventions.md` — the expdpy rulebook a generated function must obey.
- `references/wiring-checklist.md` — file-by-file integration steps, grouped by layer.
- `references/templates.md` — annotated function-module and pytest skeletons.
- `scripts/verify.sh` — the deterministic ruff/mypy/pytest gate.
- `scripts/review_workflow.js` — the parameterized adversarial find→verify review (run via the
  `Workflow` tool with `scriptPath` + `args`).
