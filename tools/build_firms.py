"""Generate the bundled synthetic *unbalanced* firms panel.

Run with ``python tools/build_firms.py``. Produces a small unbalanced firm-year panel with
staggered entry, attrition and a few interior gaps, a persistent (AR-1) log-revenue process,
a discrete size class that transitions over time, and a sector grouping factor. Built to
showcase the panel-structure, transition-matrix and within-persistence diagnostics.
Deterministic (seeded).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "src" / "expdpy" / "data"

N_FIRMS = 30
YEARS = list(range(2010, 2021))  # 11 candidate periods
SECTORS = ["Tech", "Retail", "Manufacturing", "Services"]
_SECTOR_BASE = {"Tech": 4.8, "Retail": 4.2, "Manufacturing": 4.5, "Services": 4.0}


def build() -> pd.DataFrame:
    rng = np.random.default_rng(20240620)
    rows = []
    for f in range(1, N_FIRMS + 1):
        sector = SECTORS[f % len(SECTORS)]
        entry = int(rng.integers(YEARS[0], YEARS[0] + 5))  # staggered entry 2010-2014
        exit_year = int(
            rng.integers(YEARS[-1] - 4, YEARS[-1] + 1)
        )  # attrition 2016-2020
        if exit_year < entry:
            exit_year = YEARS[-1]
        span = [y for y in YEARS if entry <= y <= exit_year]
        # a few firms skip one interior year (a genuine gap)
        if len(span) >= 5 and rng.random() < 0.4:
            drop = int(rng.choice(span[1:-1]))
            span = [y for y in span if y != drop]

        base = _SECTOR_BASE[sector] + rng.normal(0.0, 0.4)
        log_rev = base
        for y in span:
            # AR(1) within firm: persistent deviations around the firm/sector base + drift
            log_rev = (
                base
                + 0.6 * (log_rev - base)
                + 0.04 * (y - entry)
                + rng.normal(0.0, 0.18)
            )
            revenue = float(np.exp(log_rev))
            employees = int(max(1, round(revenue * rng.uniform(0.8, 1.2))))
            rows.append(
                {
                    "firm": f"F{f:02d}",
                    "year": y,
                    "sector": sector,
                    "revenue": round(revenue, 3),
                    "log_revenue": round(float(log_rev), 4),
                    "employees": employees,
                    "profitable": int(rng.random() < 0.55 + 0.1 * (log_rev - base)),
                }
            )
    df = pd.DataFrame(rows)
    # Discrete size class from global revenue terciles — a state that transitions over time.
    df["size_class"] = pd.qcut(
        df["revenue"], 3, labels=["small", "medium", "large"]
    ).astype(str)
    return df


def build_data_def() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "var_name": "firm",
                "var_def": "Firm identifier (cross-section id)",
                "type": "entity",
                "can_be_na": False,
            },
            {
                "var_name": "year",
                "var_def": "Year (time-series id)",
                "type": "time",
                "can_be_na": False,
            },
            {
                "var_name": "sector",
                "var_def": "Industry sector",
                "type": "factor",
                "can_be_na": False,
            },
            {
                "var_name": "revenue",
                "var_def": "Annual revenue (synthetic units)",
                "type": "numeric",
                "can_be_na": True,
            },
            {
                "var_name": "log_revenue",
                "var_def": "Natural log of revenue (persistent AR-1)",
                "type": "numeric",
                "can_be_na": True,
            },
            {
                "var_name": "employees",
                "var_def": "Number of employees",
                "type": "numeric",
                "can_be_na": True,
            },
            {
                "var_name": "profitable",
                "var_def": "Profitable in the year (0/1)",
                "type": "logical",
                "can_be_na": True,
            },
            {
                "var_name": "size_class",
                "var_def": "Revenue tercile: small / medium / large",
                "type": "factor",
                "can_be_na": False,
            },
        ]
    )


def main() -> None:
    df = build()
    df.to_parquet(OUT / "firms.parquet", index=False)
    build_data_def().to_parquet(OUT / "firms_data_def.parquet", index=False)
    counts = df.groupby("firm")["year"].nunique()
    print(
        f"wrote {len(df)} rows ({df['firm'].nunique()} firms) to {OUT / 'firms.parquet'}"
    )
    print(f"obs per firm: min {counts.min()}, max {counts.max()} (unbalanced)")


if __name__ == "__main__":
    main()
