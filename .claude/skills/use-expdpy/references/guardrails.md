# Guardrails (read before writing any conclusion)

expdpy is built so the library never overclaims. When you report its results, match that
discipline exactly.

## 1. Associations, not causation
- Every `.interpret()` describes how variables **move together**. It never says "causes" or
  "effect of", and it ends with a pointer to `explain("correlation_vs_causation")`.
- Mirror this in your own summaries. Prefer: "is associated with", "is higher when", "co-moves
  with", "predicts". Avoid: "causes", "the effect of", "leads to", "drives" (when stated as fact).
- The one principled exception is an identified design (instrumental variables, a clean event
  study). Even then the causal claim rests on **your** defense of the assumptions (instrument
  relevance + exclusion; parallel trends), not on the regression output. State those assumptions.

## 2. Panel vocabulary is entity + time
- The unit dimension is always `entity`; the period dimension is always `time`. Never `id`,
  `unit`, `cs_id`, `ts_id`, `group`.
- Declare once: `xp.set_panel(df, entity=..., time=...)` (or `set_labels(df, data_def,
  set_panel=True)`), then call functions without repeating it. Per-call `entity=`/`time=` override.

## 3. Lead with the library's own readings
- Quote `.interpret()` rather than inventing a narrative; it is calibrated to these standards.
- Use `.explain()` (on a result) or `explain("<topic>")` (standalone) to ground a method choice or
  surface caveats. `list_topics()` lists every available concept key.

## 4. Be honest about the sample and the model
- Report N and how many rows were dropped for missing values (expdpy warns when it drops rows).
- `.glance()` carries model-level scalars (N, R-squared, F). For IV, check the first-stage
  weak-instrument F (a rule of thumb: below ~10 the instrument is weak).
- Clustering changes standard errors, not point estimates. Choose the cluster level deliberately.

## Banned phrasings (in your output text)
Do not write, as statements of fact about the data: "X causes Y", "the effect of X on Y",
"X drives/leads to Y". Reframe as association unless you are defending a specific identification
strategy.
