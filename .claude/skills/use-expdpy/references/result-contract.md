# The result contract

Every `explore_*` / `analyze_*` / `learn_*` function returns a **frozen result dataclass**, never
a bare figure or DataFrame. The surface is uniform, so you read every result the same way.

## Common attributes
- `.df` - the tidy underlying data or coefficient frame (`pandas.DataFrame`). Always present.
- `.fig` - an interactive Plotly figure (functions whose name ends `_plot`). Some results carry
  several figures (e.g. `analyze_beta_convergence` has `.fig`, `.fig_conditional`, `.fig_rolling`;
  `analyze_kuznets_waves` has `.fig`, `.fig_between`, `.fig_within`).
- `.gt` - a `great_tables.GT` object (functions whose name ends `_table`).
- Named scalars - estimation results expose the numbers directly, e.g. an IV result has
  `.first_stage_f` / `.first_stage_p`; a beta-convergence result has `.beta`, `.speed`,
  `.half_life`.

## Methods (mixed in via `Interpretable`)
- `.interpret()` -> str: one paragraph, plain language, **association-only**, ending with a
  correlation-vs-causation note. Lead with this.
- `.explain()` -> `Explainer`: the concept behind the method (`.what`, `.when_to_use`, `.caveats`,
  `.references`); also available standalone as `explain("<topic>")`.
- `.tidy()` -> DataFrame: one row per coefficient/term (estimation results).
- `.glance()` -> DataFrame: one row of model-level scalars (N, R-squared, F, ...).

Not every method is meaningful on every result type (a scatter plot has no coefficients). Calling
one that is not raises a clear error.

## Displaying and saving outputs
- In a notebook, evaluate `result.fig` or `result.gt` as the last expression in a cell to render.
- In a script, **never** call `result.fig.show()`. Persist instead:
  ```python
  result.fig.write_html("figure.html")     # always works, no extra dependency
  result.fig.write_image("figure.png")     # needs `kaleido`
  ```
- For tables: `result.df` is the data; `result.gt` is the formatted Great Table. To export the
  Great Table, use its own methods (e.g. `.as_raw_html()`).

## Reading estimation output
```python
res = xp.analyze_panel_table(df, dv="y", idvs=["x1", "x2"])
res.tidy()      # per-term estimates, SEs, p-values across the pooled/between/fe/re columns
res.glance()    # per-model N, R-squared, ...
res.interpret() # association-only summary
res.gt          # the side-by-side formatted table
```
