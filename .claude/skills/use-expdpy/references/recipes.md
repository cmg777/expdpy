# More recipes

Runnable, end-to-end, on bundled data. See `../SKILL.md` for the three core recipes
(descriptive+correlation, IV, Kuznets waves).

## Panel models side by side + Hausman
```python
import expdpy as xp
from expdpy.data import load_kuznets, load_kuznets_data_def

df = xp.set_labels(load_kuznets(), load_kuznets_data_def(), set_panel=True)

panel = xp.analyze_panel_table(df, dv="gini_regional", idvs=["log_gdp_pc", "trade_share"])
print(panel.tidy())          # pooled / between / fe / re, side by side
print(panel.interpret())

haus = xp.analyze_hausman_test(df, dv="gini_regional", idvs=["log_gdp_pc", "trade_share"])
print(haus.interpret())      # fixed vs random effects guidance
```

## OLS with fixed effects and clustered SEs
```python
import expdpy as xp
from expdpy.data import load_kuznets, load_kuznets_data_def

df = xp.set_labels(load_kuznets(), load_kuznets_data_def(), set_panel=True)

reg = xp.analyze_regression_table(
    df,
    dvs="gini_regional",
    idvs=["log_gdp_pc", "trade_share"],
    feffects=["country", "year"],   # two-way fixed effects
    clusters=["country"],           # cluster-robust SEs
)
print(reg.tidy())
reg.gt
```

## Beta- and sigma-convergence
```python
import expdpy as xp
from expdpy.data import load_bolivia112_gdppc, load_bolivia112_gdppc_data_def

df = xp.set_labels(
    load_bolivia112_gdppc(), load_bolivia112_gdppc_data_def(), set_panel=True
)

beta = xp.analyze_beta_convergence(df, var="log_gdppc")
print(beta.glance())         # beta, speed, half-life
print(beta.interpret())

sigma = xp.analyze_sigma_convergence(df, var="log_gdppc")
sigma.fig                    # dispersion over time
```

## Staggered DiD event study
```python
import expdpy as xp
from expdpy.data import load_staggered_did

df = load_staggered_did()
es = xp.analyze_event_study(
    df, outcome="y", unit="unit", time="year", cohort="cohort", estimator="did2s"
)
es.fig                       # event-time coefficients with a pre-trend check
print(es.interpret())        # parallel-trends caveats apply - state them
```

## Outlier treatment before analysis
```python
import expdpy as xp
from expdpy.data import load_kuznets

df = load_kuznets()
clean = xp.treat_outliers(df, columns=["gdp_pc"], method="winsorize", limits=(0.01, 0.99))
```
Check exact parameter names against `function-catalog.md` - some knobs differ by function.
