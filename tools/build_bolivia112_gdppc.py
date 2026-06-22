"""Generate the bundled `bolivia112_gdppc` subnational convergence panel.

Run with ``python tools/build_bolivia112_gdppc.py``. Reads the two source CSVs shipped under
``docs/data/`` and writes the committed parquets under ``src/expdpy/data/``: a balanced annual
panel of 112 Bolivian provinces (nested within 9 departments) over 1990-2024 with GDP per capita
and its natural log, plus the province/department grouping factors and a variable-definition
table. It is the real-world counterpart to the synthetic ``productivity`` panel -- suitable for
both the convergence workflows (beta / sigma / club) and general subnational exploration.

Because the variable-definition table is provided as a CSV alongside the data, ``build_data_def``
just reads and reorders it to the repo's df_def column order (rather than hand-authoring rows).

Source: Kummu, M., Kosonen, M. & Masoumzadeh Sayyar, S. "Downscaled gridded global dataset for
gross domestic product (GDP) per capita PPP over 1990-2022." Sci Data 12, 178 (2025).
https://doi.org/10.1038/s41597-025-04487-x

The source CSVs are the working-tree reference; this script only regenerates the committed
parquet (the data itself ships in the package).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "src" / "expdpy" / "data"
SOURCE = ROOT / "docs" / "data" / "bolivia112_gdppc.csv"
SOURCE_DEF = ROOT / "docs" / "data" / "bolivia112_gdppc_def.csv"

# Canonical column order, shared by the data frame and the var-definition table.
_COLS = ["prov_id", "prov", "dep", "dep_id", "year", "gdppc", "log_gdppc"]
# Repo df_def column order (the provided CSV is var_name,type,label,var_def,can_be_na).
_DEF_COLS = ["var_name", "var_def", "label", "type", "can_be_na"]


def build() -> pd.DataFrame:
    df = pd.read_csv(SOURCE)
    for col in ("prov_id", "dep_id", "year"):
        df[col] = df[col].astype(int)
    for col in ("prov", "dep"):
        df[col] = df[col].astype(str)
    return df[_COLS].sort_values(["prov_id", "year"]).reset_index(drop=True)


def build_data_def() -> pd.DataFrame:
    dd = pd.read_csv(SOURCE_DEF)
    return dd[_DEF_COLS].reset_index(drop=True)


def main() -> None:
    df = build()
    df.to_parquet(OUT / "bolivia112_gdppc.parquet", index=False)
    build_data_def().to_parquet(OUT / "bolivia112_gdppc_data_def.parquet", index=False)
    counts = df.groupby("prov_id")["year"].nunique()
    print(
        f"wrote {len(df)} rows ({df['prov_id'].nunique()} provinces, "
        f"{df['year'].nunique()} years) to {OUT / 'bolivia112_gdppc.parquet'}"
    )
    print(
        f"balanced: {counts.min() == counts.max()} "
        f"(obs per province: {counts.min()}-{counts.max()})"
    )


if __name__ == "__main__":
    main()
