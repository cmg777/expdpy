---
name: use-expdpy
description: >-
  This skill should be used whenever the user wants to ANALYZE, EXPLORE, or EXPLAIN a panel
  or cross-sectional dataset using the installed expdpy library - e.g. "explore/describe this
  panel", "is GDP per capita converging across regions?", "run a fixed-effects / IV / 2SLS
  regression", "estimate the Kuznets curve", "make a descriptive or correlation table", "plot
  trends / missing values / a coefficient plot", "interpret this result in plain language", or
  "explain omitted-variable bias / fixed effects / convergence". It teaches the end-user
  workflow: declare the panel once with set_panel(entity=, time=), pick the right
  explore_/analyze_/learn_ function, read the frozen result object (.df, .fig, .gt) and its
  .interpret()/.explain()/.tidy()/.glance(), and obey expdpy's econometric guardrails -
  results are ASSOCIATIONS, not causation, and entity/time is the panel vocabulary. Use this to
  DO analysis WITH expdpy; to ADD a new function to the library, use the write-function skill
  instead.
---

# use-expdpy

Use **expdpy** - Explore / Analyze / Learn for panel and cross-sectional data - to do real
analysis: declare the panel, call one module-prefixed function, and read the frozen result it
returns. Every function returns a typed result object with a uniform surface (`.df`, `.fig`,
`.gt`) and, on most, plain-language `.interpret()` / `.explain()`. The value of this skill is
**doing the analysis the way the library intends** - panel-aware, association-only, and
self-documenting - instead of reaching for raw pandas/statsmodels.

expdpy is a Python port of the R package ExPanDaR. Import it as `import expdpy as xp`. The full,
always-correct list of functions is in `references/function-catalog.md` (generated from the
installed library, so it never lies). Read that before guessing a name.

## The four-step workflow

### Step 1 - Declare the panel once
Panel data has two index columns: an **entity** (the unit - country, region, firm) and a
**time** column. Declare them once and every later call inherits them:

```python
import expdpy as xp
from expdpy.data import load_kuznets, load_kuznets_data_def

df = xp.set_labels(load_kuznets(), load_kuznets_data_def(), set_panel=True)
# or, without a data dictionary:
df = xp.set_panel(load_kuznets(), entity="country", time="year")
```

`set_panel` writes the panel onto `df.attrs`; `resolve_panel` reads it back. You may always
override per call with the keyword-only `entity=` / `time=` arguments - explicit args win. The
vocabulary is **always `entity` and `time`** - never `id`, `unit`, `cs_id`, or `ts_id`. If you
are unsure which columns qualify, run
`python .claude/skills/use-expdpy/scripts/check_panel.py yourfile.csv` for a read-only report of
candidate index columns and dtypes.

Cross-sectional data (one row per entity, no time) needs no `set_panel`; pass it straight to
the cross-sectional functions (e.g. `analyze_iv_regression`).

### Step 2 - Pick the module
- **Explore** (`explore_*`) - describe and visualize *before* modeling:
  `explore_descriptive_table`, `explore_correlation_table`, `explore_missing_values_plot`,
  `explore_trend_plot`, `explore_scatter_plot`, `explore_spaghetti_plot`,
  `explore_xtsum_table`, `explore_panel_structure`. Functions ending `_plot` return a Plotly
  `.fig`; functions ending `_table` return a Great Tables `.gt`.
- **Analyze** (`analyze_*`) - estimate: `analyze_regression_table` / `analyze_panel_table`
  (OLS / fixed / random effects), `analyze_iv_regression` (2SLS), `analyze_beta_convergence` /
  `analyze_sigma_convergence` / `analyze_convergence_clubs`, `analyze_kuznets_waves`,
  `analyze_event_study` (staggered DiD).
- **Learn** (`learn_*`) - runnable concept sandboxes on simulated data
  (`learn_omitted_variable_bias`, `learn_pooled_vs_fixed_effects`, ...) for *teaching* a concept,
  not for estimating on the user's data.

When the intent is fuzzy, open `references/choosing-a-function.md`.

### Step 3 - Read the result object (the universal contract)
Every `explore_*` / `analyze_*` / `learn_*` function returns a **frozen result dataclass**, never
a bare figure or frame. The surface is uniform:

| Attribute      | What it is                                                            |
|----------------|-----------------------------------------------------------------------|
| `.df`          | the tidy underlying data / coefficient frame (a `pandas.DataFrame`)    |
| `.fig`         | an interactive Plotly figure (functions ending `_plot`)               |
| `.gt`          | a Great Tables object (functions ending `_table`)                     |
| `.interpret()` | one paragraph of plain-language, **association-only** interpretation   |
| `.explain()`   | the relevant concept explainer from the pedagogy registry             |
| `.tidy()`      | one row per coefficient/term (estimation results)                     |
| `.glance()`    | one row of model-level scalars (N, R-squared, F, ...)                 |

Not every method is meaningful on every result. Never call `.show()` on `.fig` in a script - to
persist a figure, write it: `result.fig.write_html("fig.html")` (or `.write_image("fig.png")`).
Full surface and idioms in `references/result-contract.md`.

### Step 4 - Interpret responsibly (READ THIS BEFORE WRITING ANY CONCLUSION)
expdpy is deliberately built so the library never overclaims, and **you must match it**:

- **Associations, not causation.** `.interpret()` describes how variables *move together*. It
  never contains the word "causes" or the phrase "effect of", and it ends with a note pointing to
  `xp.explain("correlation_vs_causation")`. When you summarize a result, keep that framing: write
  "is associated with" / "is higher when", not "causes" / "the effect of". The one place a causal
  claim is legitimate is instrumental variables - and even there the causal argument is *yours*
  (you must defend the instrument's relevance and exclusion), while `.interpret()` still reports
  the association.
- **Lead with `.interpret()`.** Prefer quoting the library's own interpretation over inventing
  your own narrative; it is calibrated to the library's standards.
- **Reach for `explain(topic)` when a concept needs grounding.** `xp.list_topics()` returns the
  concept keys (e.g. `fixed_effects`, `omitted_variable_bias`, `instrumental_variables`,
  `beta_convergence`). `xp.explain("fixed_effects")` returns an `Explainer` with `what` /
  `when_to_use` / `caveats` / `references`. Use it to justify a method or warn about a pitfall,
  instead of writing econometrics from memory.

The full guardrail cheat-sheet is in `references/guardrails.md`.

## Recipes (runnable, end-to-end, bundled data)

### Recipe A - Descriptive + correlation overview of a panel
```python
import expdpy as xp
from expdpy.data import load_kuznets, load_kuznets_data_def

df = xp.set_labels(load_kuznets(), load_kuznets_data_def(), set_panel=True)

desc = xp.explore_descriptive_table(df)                 # summarizes all numeric columns
corr = xp.explore_correlation_table(df[["gini_regional", "gdp_pc", "trade_share"]])

print(desc.df)              # the numbers
desc.gt                     # the formatted Great Table (display in a notebook)
print(corr.interpret())     # association-only reading of the correlations
```

### Recipe B - Instrumental variables (AJR 2001 colonial origins, cross-section)
```python
import expdpy as xp
from expdpy.data import load_colonial_origins, load_colonial_origins_data_def

ajr = xp.set_labels(load_colonial_origins(), load_colonial_origins_data_def())
base = ajr[ajr["base_sample"] == 1]          # the 64-country base sample

iv = xp.analyze_iv_regression(
    base,
    dv="log_gdp_pc_1995",
    endog="expropriation_risk",              # institutions: endogenous
    instruments="log_settler_mortality",     # the AJR instrument
)
print(iv.tidy())            # 2SLS coefficients (the instrumented slope ~ 0.94)
print(iv.glance())          # includes the first-stage weak-instrument F (watch for < 10)
print(iv.interpret())       # association framing; the causal case is YOURS to argue
```
IV is the *one* method where a causal claim is on-topic - but only if you explicitly defend that
`log_settler_mortality` is relevant (moves institutions) and excludable (affects income only
through institutions). See `xp.explain("instrumental_variables")`.

### Recipe C - Kuznets waves across pooled / between / within estimators
```python
import expdpy as xp
from expdpy.data import load_kuznets, load_kuznets_data_def

df = xp.set_labels(load_kuznets(), load_kuznets_data_def(), set_panel=True)

kw = xp.analyze_kuznets_waves(df, inequality="gini_regional", development="log_gdp_pc")
print(kw.interpret())       # how inequality co-moves with development - associational
kw.fig                      # pooled scatter with the fitted wave
```

## Resources
- `references/function-catalog.md` - every public function, grouped, with signature + one-line
  summary (generated from the installed library; always current).
- `references/choosing-a-function.md` - "I want to ... -> call ..." decision guide.
- `references/guardrails.md` - association-not-causation, entity/time vocabulary, reading
  `.interpret()`, and the `explain()` registry.
- `references/result-contract.md` - the full `.df`/`.fig`/`.gt`/`.interpret()`/`.explain()`/
  `.tidy()`/`.glance()` surface and the figure-saving idiom.
- `references/recipes.md` - more end-to-end recipes (event-study/DiD, panel models + Hausman,
  robust inference, outlier treatment).
- `scripts/check_panel.py` - read-only helper that reports candidate entity/time columns.

## Programmatic access (MCP server + tool schemas)
To let an agent *call* expdpy directly, install the MCP server with `pip install 'expdpy[mcp]'`
and run `expdpy-mcp` (stdio). Static Anthropic/OpenAI function-calling schemas are published at
`https://cmg777.github.io/expdpy/tools/anthropic_tools.json` and `.../openai_tools.json`, and a
machine-readable map of the whole library is at `https://cmg777.github.io/expdpy/llms.txt`.

## When to use a different skill
This skill is for **using** expdpy on data. To **add or change a function in the library itself**
(write `analyze_*`/`explore_*`/`learn_*`, an estimator, a result dataclass, tests, wiring), use
the **write-function** skill instead - it is the developer counterpart.
