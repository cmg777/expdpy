"""Generate the bundled `productivity` convergence-clubs dataset.

Run with ``python tools/build_productivity.py``. Distils a clean, balanced annual panel of
cross-country labor productivity and GDP per capita (Penn World Table 9.0, 1990-2014) from the
reference source ``dat.csv`` shipped with the Phillips-Sul club-convergence materials
(mendez2020-convergence-clubs). The bundled frame keeps the **raw** (unfiltered) log series so
:func:`expdpy.analyze_convergence_clubs` does the Hodrick-Prescott (lambda=400) trend filtering
itself, plus level series and grouping factors for context.

The source CSV is the working-tree reference folder; this script is only needed to regenerate
the committed parquet (the data itself ships in the package).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "src" / "expdpy" / "data"
SOURCE = (
    ROOT
    / "reference_for_convergence_clubs"
    / "visualization and export of the clubs"
    / "assets"
    / "dat.csv"
)

# Source column -> bundled column (kept lean and self-explanatory).
_RENAME = {
    "country": "country",
    "year": "year",
    "region": "region",
    "GDPpc": "gdppc",
    "lp": "lp",
    "log_GDPpc_raw": "log_gdppc",  # RAW log GDP per capita (HP-filtered inside the function)
    "log_lp_raw": "log_lp",  # RAW log labor productivity
}


def build() -> pd.DataFrame:
    raw = pd.read_csv(SOURCE, na_values=["."])
    df = raw[[*_RENAME]].rename(columns=_RENAME).copy()
    df["high_income_1990"] = raw["hi1990"].astype(str).str.strip().str.lower().eq("yes")
    df["year"] = df["year"].astype(int)
    df["region"] = df["region"].astype(str)

    # Keep a strictly balanced panel: drop any country missing a year or a log series.
    needed = ["log_gdppc", "log_lp", "gdppc", "lp"]
    df = df.dropna(subset=needed)
    n_years = df["year"].nunique()
    complete = df.groupby("country")["year"].nunique().eq(n_years)
    df = df[df["country"].isin(complete[complete].index)].copy()

    cols = [
        "country",
        "year",
        "region",
        "high_income_1990",
        "gdppc",
        "lp",
        "log_gdppc",
        "log_lp",
    ]
    return df[cols].sort_values(["country", "year"]).reset_index(drop=True)


def build_data_def() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "var_name": "country",
                "var_def": "Country (cross-section id, standardized PWT name)",
                "label": "Country",
                "type": "entity",
                "can_be_na": False,
            },
            {
                "var_name": "year",
                "var_def": "Year (time-series id, annual 1990-2014)",
                "label": "Year",
                "type": "time",
                "can_be_na": False,
            },
            {
                "var_name": "region",
                "var_def": "UN regional group",
                "label": "Region",
                "type": "factor",
                "can_be_na": False,
            },
            {
                "var_name": "high_income_1990",
                "var_def": "High-income country as of 1990 (World Bank classification)",
                "label": "High income (1990)",
                "type": "logical",
                "can_be_na": False,
            },
            {
                "var_name": "gdppc",
                "var_def": "GDP per capita (PWT9.0, constant prices)",
                "label": "GDP per capita",
                "type": "numeric",
                "can_be_na": False,
            },
            {
                "var_name": "lp",
                "var_def": "Labor productivity (GDP per worker, PWT9.0)",
                "label": "Labor productivity",
                "type": "numeric",
                "can_be_na": False,
            },
            {
                "var_name": "log_gdppc",
                "var_def": "Natural log of GDP per capita (raw, before HP filtering)",
                "label": "Log GDP per capita",
                "type": "numeric",
                "can_be_na": False,
            },
            {
                "var_name": "log_lp",
                "var_def": "Natural log of labor productivity (raw, before HP filtering)",
                "label": "Log labor productivity",
                "type": "numeric",
                "can_be_na": False,
            },
        ]
    )


def main() -> None:
    df = build()
    df.to_parquet(OUT / "productivity.parquet", index=False)
    build_data_def().to_parquet(OUT / "productivity_data_def.parquet", index=False)
    counts = df.groupby("country")["year"].nunique()
    print(
        f"wrote {len(df)} rows ({df['country'].nunique()} countries, "
        f"{df['year'].nunique()} years) to {OUT / 'productivity.parquet'}"
    )
    print(
        f"balanced: {counts.min() == counts.max()} "
        f"(obs per country: {counts.min()}-{counts.max()})"
    )


if __name__ == "__main__":
    main()
