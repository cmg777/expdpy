#!/usr/bin/env python
"""Build-time dataset conversion for expdpy.

Converts ExPanDaR's bundled R data to the parquet/JSON files shipped under
``src/expdpy/data/``:

* ExPanDaR ``.RData`` frames are read with :mod:`pyreadr` and written as parquet — but none
  are bundled at present (``russell_3000`` / ``worldbank`` were removed), so ``FRAMES`` is empty;
* ``gapminder`` is read from the vignette CSV and a small data-definition table is derived;
* any ``ExPanD_config_*`` R *lists* are converted to JSON by ``tools/convert_rdata.R``.

The synthetic ``kuznets`` dataset is *not* built here — it is generated (no R source) by
``tools/build_kuznets.py``.

Run from the repo root::

    python tools/build_datasets.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "ExPanDaR" / "data"
GAPMINDER_CSV = REPO / "ExPanDaR" / "vignettes" / "data" / "gapminder.csv"
OUT = REPO / "src" / "expdpy" / "data"

# (rdata file, R object name, output parquet stem). ExPanDaR's russell_3000 / worldbank
# frames are no longer bundled, so there are none to convert here.
FRAMES: list[tuple[str, str, str]] = []


def _read_frame(rdata: str, obj: str) -> pd.DataFrame:
    import pyreadr  # lazy: only needed for the (currently empty) RData FRAMES

    result = pyreadr.read_r(str(SRC / rdata))
    if obj in result:
        return result[obj]
    # Fall back to the single object present.
    return next(iter(result.values()))


def _infer_type(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "logical"
    if isinstance(series.dtype, pd.CategoricalDtype) or pd.api.types.is_object_dtype(
        series
    ):
        return "factor"
    return "numeric"


# Concise, human-readable display labels for the gapminder columns (the CSV ships none).
GAPMINDER_LABELS = {
    "country": "Country",
    "iso3c": "ISO3 code",
    "continent": "Continent",
    "year": "Year",
    "lifeExp": "Life expectancy (years)",
    "pop": "Population",
    "gdpPercap": "GDP per capita (USD)",
}


def build_gapminder_data_def(gap: pd.DataFrame) -> pd.DataFrame:
    """Derive the gapminder variable-definition table (with display labels) from the frame."""
    gap_def = pd.DataFrame(
        {
            "var_name": list(gap.columns),
            "var_def": list(gap.columns),
            "label": [GAPMINDER_LABELS.get(c, c) for c in gap.columns],
            "type": [_infer_type(gap[c]) for c in gap.columns],
            "can_be_na": [True] * gap.shape[1],
        }
    )
    # Mark the conventional panel identifiers when present.
    type_map = {"country": "entity", "iso3c": "entity", "year": "time"}
    gap_def["type"] = [
        type_map.get(name, t)
        for name, t in zip(gap_def["var_name"], gap_def["type"], strict=False)
    ]
    gap_def["can_be_na"] = [t not in ("entity", "time") for t in gap_def["type"]]
    return gap_def


def build() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    for rdata, obj, stem in FRAMES:
        df = _read_frame(rdata, obj)
        df.to_parquet(OUT / f"{stem}.parquet", index=False)
        print(f"wrote {stem}.parquet  ({df.shape[0]} rows x {df.shape[1]} cols)")

    # gapminder from CSV + a derived data-definition table.
    gap = pd.read_csv(GAPMINDER_CSV)
    gap.to_parquet(OUT / "gapminder.parquet", index=False)
    print(f"wrote gapminder.parquet  ({gap.shape[0]} rows x {gap.shape[1]} cols)")

    gap_def = build_gapminder_data_def(gap)
    gap_def.to_parquet(OUT / "gapminder_data_def.parquet", index=False)
    print("wrote gapminder_data_def.parquet")

    # Config lists -> JSON via R/jsonlite.
    rc = subprocess.run(
        ["Rscript", str(REPO / "tools" / "convert_rdata.R"), str(SRC), str(OUT)],
        check=False,
    )
    if rc.returncode != 0:
        print(
            "WARNING: convert_rdata.R failed; config JSON not written.", file=sys.stderr
        )


if __name__ == "__main__":
    build()
