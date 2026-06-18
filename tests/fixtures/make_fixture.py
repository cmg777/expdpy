"""Generate the deterministic fixture CSV shared by Python tests and the R golden script.

Run once (committed output): ``python tests/fixtures/make_fixture.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def make_panel() -> pd.DataFrame:
    """Build a small, reproducible panel data frame."""
    rng = np.random.default_rng(20240618)
    n_firms, n_years = 20, 8
    firms = np.repeat(np.arange(1, n_firms + 1), n_years)
    years = np.tile(np.arange(2014, 2014 + n_years), n_firms)
    n = n_firms * n_years
    x1 = rng.normal(0, 1, n)
    x2 = 0.6 * x1 + rng.normal(0, 1, n)
    x3 = rng.gamma(2.0, 2.0, n)
    grp = rng.choice(["A", "B", "C"], n)
    df = pd.DataFrame(
        {
            "firm": firms,
            "year": years,
            "grp": grp,
            "x1": x1,
            "x2": x2,
            "x3": x3,
        }
    )
    return df


if __name__ == "__main__":
    make_panel().to_csv(HERE / "sample.csv", index=False)
    print(f"wrote {HERE / 'sample.csv'}")
