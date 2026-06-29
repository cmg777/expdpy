# Wiring checklist — files to touch per layer

The exact integration steps, grouped by layer, derived from the `analyze_beta_convergence`
change. The **Core** and **Release** layers always apply; the rest are optional and added only
when the user requests them (adaptive scope). Replace `<fn>` with the function name (e.g.
`analyze_beta_convergence`), `<module>` with its file (e.g. `convergence`), `<Result>` with the
dataclass (e.g. `BetaConvergenceResult`), and `<topic>` with the explainer key.

## Core (always)

- **`src/expdpy/<module>.py`** — new module: the public function + `_`-prefixed helpers.
  `__all__ = ["<fn>"]`. See `references/templates.md`.
- **`src/expdpy/_types.py`** — add the frozen `<Result>(Interpretable)` dataclass.
  - Add `<Result>` to the module `__all__` (grouped, alphabetical within its group is fine but
    the list is topic-grouped, not globally sorted).
  - If it implements `.interpret()`, add `interpret_<topic>` to the `from
    expdpy.pedagogy._interpret import (...)` block at the top.
- **`src/expdpy/__init__.py`** —
  - Import the function: `from expdpy.<module> import <fn>`.
  - Import the result type in the `from expdpy._types import (...)` block.
  - Add `<fn>` to `__all__` under the matching `# ===== EXPLORE/ANALYZE/LEARN =====` section.
  - Add `<Result>` to `__all__` in the `# ===== RESULT TYPES =====` section.
- **`docs/_quarto.yml`** — register `<fn>` in the `quartodoc.sections[*].contents` list for the
  function's own section (Explore / Analyze / Learn). quartodoc renders **only** what is listed
  in `contents` — the `__all__` export does not auto-populate it — so this is the step that
  creates the function's reference page. The page then **automatically** gains the executed
  example output and the `[source]` link to an auto-generated source page (`build_source_pages.py`
  / `build_reference_enrichment.py` run inside `docs-build`; no manual step for either). Because
  the docstring `Examples` are executed there at build time, they must be self-contained and
  runnable — see `references/templates.md` and the Phase 5 docs check.

## LLM layer (always)

The agent-facing surface derives from the public API (`__all__` + docstrings) via
`src/expdpy/_meta.py` — see CLAUDE.md **Keeping the LLM layer in sync**. Two parts:

- **Auto-coverage (mandatory).** The moment `<fn>` is exported and in the quarto `contents` (the
  Core step), it flows into `llms.txt` / `llms-full.txt`, its per-page `.md`, and the `use-expdpy`
  `function-catalog.md`. Regenerate and commit, or CI `artifacts-fresh` fails:
  ```bash
  pixi run -e docs python tools/build_tool_schemas.py
  pixi run -e docs python tools/build_use_skill.py
  pixi run -e docs python tools/build_llms_txt.py --canonical-only
  ```
  Commit `schemas/anthropic_tools.json`, `schemas/openai_tools.json`,
  `.claude/skills/use-expdpy/references/function-catalog.md`, `docs/use-with-llms.qmd`,
  `docs/llms.txt`. (`llms-full.txt` + per-page `.html.md` are `_site`-only — not committed.)
- **Curation (optional: expose `<fn>` as an agent tool).** For a **high-value, headless-safe**
  `<fn>`, append a `ToolSpec(name="<fn>", figure_attrs=(...), tables=(...), table_methods=(...),
  see_also_topics=("<topic>", ...), forced_kwargs=...)` to `CURATED_TOOL_SPECS` in
  `src/expdpy/_meta.py` — **output-surface metadata only** (the input JSON Schema is compiled from
  the signature + docstring; use `forced_kwargs={"format": "df"}` for `_table` functions). Then
  regenerate (above) so `<fn>` appears in the MCP server + Anthropic/OpenAI schemas.
  `tests/test_meta.py` / `tests/test_tool_schemas.py` validate the spec; the
  `uncurated_analysis_functions` test lists what is not yet curated.

## Pedagogy (optional: concept explainer + `.interpret()`)

- **`src/expdpy/pedagogy/_text/<topic>.py`** — new file calling `register_topic(Explainer(
  topic="<topic>", title=..., what=..., when_to_use=..., caveats=(...), see_also=(...),
  references=(...)), aliases=(...))`. Verify `see_also` keys exist
  (`grep 'topic="' src/expdpy/pedagogy/_text/`).
- **`src/expdpy/pedagogy/_text/__init__.py`** — add `<topic>` to the import block and `__all__`.
- **`src/expdpy/pedagogy/_interpret.py`** — add `interpret_<topic>(result, *, lang="en") -> str`
  (duck-typed; reads `.df`/scalars; association-only; ends with `_ASSOC_NOTE`; no "causes"/
  "effect of"). Add `"interpret_<topic>"` to that module's `__all__`.
- **`src/expdpy/pedagogy/__init__.py`** — add `interpret_<topic>` to both the import block and
  `__all__`.
- Back in **`_types.py`**, the `<Result>.interpret()` delegates to `interpret_<topic>(self)`
  and `.explain()` returns `_explain("<topic>")`.

## Learn sandbox (optional: runnable `learn_*`)

- **`src/expdpy/sandbox.py`** — add `learn_<topic>(*, ..., seed=0) -> SandboxResult` that
  simulates a known-parameter DGP and (ideally) **dogfoods the new analyze function** to show
  the concept; build a comparison `df` + `fig` + a `summary` dict of the scalar facts; return
  `SandboxResult(df, fig, summary, topic="<topic>")`. Add the name to `sandbox.__all__` and
  import the analyze function if dogfooding.
- **`src/expdpy/pedagogy/_interpret.py`** — add a `if topic == "<topic>":` branch to
  `interpret_sandbox` reading the `summary` scalars (the single explainer already resolves
  `learn_<topic>().explain()`).
- **`src/expdpy/__init__.py`** — import and export `learn_<topic>` under `# ===== LEARN =====`.
- **`docs/_quarto.yml`** — this is the **Core** docs-registration step applied to the Learn
  section: add `learn_<topic>` to the quartodoc **Learn**-section `contents` (and refresh that
  section's `desc` if needed) so it gets a reference page.
- **`docs/learn.qmd`** — add a `### `learn_<topic>`` section that runs `learn_<topic>(...)`,
  prints `.interpret()`, and renders `.fig` (cross-link the analyze page if there is one). This
  source edit is what makes `notebooks/learn.ipynb` change on regeneration; commit the
  regenerated notebook too (see the Notebook layer's Regenerate step).

## Streamlit (optional: app tab)

- **`src/expdpy/streamlit_app/_pages.py`** — extend the relevant page (e.g. `page_sandboxes`):
  import the function, add a tab label, and a `with tabs[i]:` block with sliders rendering
  `.fig` + `.interpret()` (+ `.explain().to_markdown()` in an expander).
- **`tests/test_streamlit_app.py`** — update any assertion on the tab count / tab set.

## Notebook (optional: Quarto → Colab, "extended guide + verification harness")

- **`docs/<fn>.qmd`** — front matter `title: "<fn>"`, `jupyter: python3`; a `{=html}` Colab
  badge block (stripped by the build script); sections: concept + math, per-argument reference,
  the full result surface, a **testing** section whose `assert`s recover the known baseline,
  and a real-data walkthrough. (Feature notebooks are named after the function — see the memory
  note `feature-notebook-naming`.)
- **`tools/build_quickstart_notebook.py`** — add a `MODULES` entry `{"slug": "<fn>", "title":
  "<fn>", "title_md": "..."}` (custom `title_md` for a function guide vs. the module template).
- **`docs/_quarto.yml`** — add the guide page to the Articles `menu`. (`<fn>` is already in the
  quartodoc `contents` from the **Core** step — that is what gives it a reference page; this
  optional layer only adds the long-form Colab guide on top.)
- **Regenerate**: `pixi run -e docs python tools/build_quickstart_notebook.py` then
  `pixi exec --spec ruff==0.15.17 -- ruff format notebooks/`; commit `notebooks/<fn>.ipynb` (and
  any module notebook that changed because its `.qmd` was edited, e.g. `learn.ipynb`). The CI
  `notebooks-fresh` job drift-checks these against their sources, ignoring the build-timestamp
  line.

## Release (always)

- Bump the patch version **+0.0.1** in `pyproject.toml` (`[project] version`),
  `src/expdpy/__init__.py` (`__version__`), and CLAUDE.md "Current version".
- Add a dated `## <version> (YYYY-MM-DD)` entry to `docs/changelog.qmd` (Added/Fixed). If the
  feature is unreleased working-tree state from the same batch, folding into the existing
  unreleased entry is acceptable instead of a new bump.
- Update README install pins and, if user-facing, the feature blurb.
- If the new module is architecturally notable, add it to the CLAUDE.md "Architecture" list.
- Regenerate + commit the LLM artifacts (see **LLM layer (always)**) — required for the CI
  `artifacts-fresh` job.

## Tests (always — see `references/templates.md`)

- **`tests/test_<module>.py`** — new file with the DGP/closed-form helper and the four test
  categories (expected-use, edge cases, math-validity, result-surface). Markers per
  `pyproject.toml [tool.pytest.ini_options].markers` (`panel`, `streamlit`, `against_r`).
- Extend **`tests/test_sandbox.py`** if a `learn_*` sandbox was added.
