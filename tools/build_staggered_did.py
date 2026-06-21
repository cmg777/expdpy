"""Generate the bundled synthetic staggered difference-in-differences dataset.

Run with ``python tools/build_staggered_did.py``. Produces a small, balanced panel with
several treatment cohorts (and a never-treated control group) and a known *dynamic*
treatment effect, for teaching event-study / staggered-DiD methods. Deterministic (seeded).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "src" / "expdpy" / "data"

N_UNITS = 80
YEARS = list(range(2000, 2017))  # 17 periods
COHORTS = [0, 2005, 2009, 2013]  # 0 == never treated
EFFECT_PER_YEAR = 0.25  # dynamic effect grows by this each post-treatment year


def build() -> pd.DataFrame:
    rng = np.random.default_rng(20240601)
    rows = []
    for u in range(1, N_UNITS + 1):
        cohort = COHORTS[u % len(COHORTS)]
        unit_fe = rng.normal(2.0, 1.0)
        for year in YEARS:
            treated = int(cohort != 0 and year >= cohort)
            rel = year - cohort if cohort != 0 else np.nan
            te = EFFECT_PER_YEAR * (year - cohort + 1) if treated else 0.0
            outcome = (
                unit_fe
                + 0.05 * (year - YEARS[0])  # common linear trend
                + te
                + rng.normal(0.0, 0.4)
            )
            rows.append(
                {
                    "unit": u,
                    "year": year,
                    "cohort": cohort,
                    "treated": treated,
                    "rel_year": rel,
                    "outcome": round(float(outcome), 6),
                }
            )
    return pd.DataFrame(rows)


def build_data_def() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "var_name": "unit",
                "var_def": "Unit identifier (cross-section id)",
                "type": "entity",
            },
            {"var_name": "year", "var_def": "Year (time-series id)", "type": "time"},
            {
                "var_name": "cohort",
                "var_def": "First treated year (0 = never treated)",
                "type": "factor",
            },
            {
                "var_name": "treated",
                "var_def": "Treatment status indicator (0/1)",
                "type": "numeric",
            },
            {
                "var_name": "rel_year",
                "var_def": "Event time = year - cohort (NaN if never treated)",
                "type": "numeric",
            },
            {
                "var_name": "outcome",
                "var_def": "Outcome with a dynamic treatment effect",
                "type": "numeric",
            },
        ]
    )


def main() -> None:
    df = build()
    df.to_parquet(OUT / "staggered_did.parquet", index=False)
    build_data_def().to_parquet(OUT / "staggered_did_data_def.parquet", index=False)
    print(f"wrote {len(df)} rows to {OUT / 'staggered_did.parquet'}")
    print(df.groupby("cohort").size().to_dict())


if __name__ == "__main__":
    main()
