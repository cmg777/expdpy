"""Enrich the bundled ``*_data_def`` parquets with a ``role`` column.

The ``role`` column (``outcome`` / ``covariate`` / ``entity_name``) lets every bundled dataset
demonstrate the key-variable defaults and the ``Name (id)`` entity display out of the box — a
scatter plots the covariate against the outcome, a regression uses the outcome as the dependent
variable, panel figures label units by name, and so on.

This is a **surgical metadata edit**: it rewrites only the small ``*_data_def`` parquets (adding /
updating the ``role`` column, positioned right after ``type``), never the data parquets — so the
R numerical goldens, which read ``tests/fixtures/sample.csv`` and not the bundled dictionaries, are
untouched. It is idempotent: re-running it just re-applies the same roles. Run it again after
regenerating any dataset's dictionary with its ``tools/build_<name>.py`` script.

    python tools/add_roles_to_data_defs.py
"""

from __future__ import annotations

from importlib import resources

import pandas as pd

# Per-dataset roles. Covariates are listed in the order they should appear (the first covariate
# becomes the scatter x-axis; all of them become the default regressors). An ``entity_name`` is a
# readable label column paired with the entity id (rendered as ``Name (id)``).
ROLES: dict[str, dict[str, object]] = {
    # The flagship N-shaped Kuznets curve: gini = f(log gdp pc, its square and cube).
    "kuznets": {
        "outcome": "gini_regional",
        "covariates": ["log_gdp_pc", "log_gdp_pc_sq", "log_gdp_pc_cu"],
    },
    # The classic Gapminder relationship (life expectancy vs income).
    "gapminder": {"outcome": "lifeExp", "covariates": ["gdpPercap"]},
    # Subnational convergence: the province name labels each unit (id is ``prov_id``).
    "bolivia112_gdppc": {"outcome": "log_gdppc", "entity_name": "prov"},
    # Acemoglu-Johnson-Robinson (2001): log GDP pc 1995 on expropriation risk.
    "colonial_origins": {
        "outcome": "log_gdp_pc_1995",
        "covariates": ["expropriation_risk"],
    },
    # Club convergence on the log GDP-per-capita panel.
    "productivity": {"outcome": "log_gdppc"},
    # Event study / staggered DiD: the treatment outcome.
    "staggered_did": {"outcome": "outcome"},
    # A small unbalanced firm panel: revenue on headcount.
    "firms": {"outcome": "log_revenue", "covariates": ["employees"]},
    # Region-year conflict instrumented by night-lights.
    "regional_conflict": {"outcome": "conflict", "covariates": ["log_lights"]},
}


def _role_series(df_def: pd.DataFrame, spec: dict[str, object]) -> list[str]:
    """Return the ``role`` value for each row of ``df_def`` per ``spec``."""
    outcome = spec.get("outcome")
    covariates = set(spec.get("covariates", []) or [])
    entity_name = spec.get("entity_name")
    names = set(df_def["var_name"])
    for label, cols in (
        ("outcome", [outcome] if outcome else []),
        ("covariate", covariates),
        ("entity_name", [entity_name] if entity_name else []),
    ):
        missing = [c for c in cols if c not in names]
        if missing:
            raise ValueError(f"{label} column(s) {missing} not in df_def")

    def role_for(name: str) -> str:
        if name == outcome:
            return "outcome"
        if name in covariates:
            return "covariate"
        if name == entity_name:
            return "entity_name"
        return ""

    return [role_for(n) for n in df_def["var_name"]]


def _with_role(df_def: pd.DataFrame, roles: list[str]) -> pd.DataFrame:
    """Return ``df_def`` with the ``role`` column set, positioned just after ``type``."""
    out = df_def.drop(columns=["role"], errors="ignore").copy()
    out["role"] = roles
    cols = [c for c in out.columns if c != "role"]
    pos = cols.index("type") + 1 if "type" in cols else len(cols)
    ordered = [*cols[:pos], "role", *cols[pos:]]
    return out[ordered]


def main() -> None:
    """Patch every dataset's ``*_data_def.parquet`` in place with its roles."""
    data_dir = resources.files("expdpy.data")
    for name, spec in ROLES.items():
        with resources.as_file(data_dir / f"{name}_data_def.parquet") as path:
            df_def = pd.read_parquet(path)
            patched = _with_role(df_def, _role_series(df_def, spec))
            patched.to_parquet(path, index=False)
            tagged = list(patched.loc[patched["role"] != "", "var_name"])
            print(f"{name}_data_def.parquet — roles set on {tagged}")


if __name__ == "__main__":
    main()
