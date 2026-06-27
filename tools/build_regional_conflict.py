"""Build the bundled African *regional-conflict* panel (a focused teaching subset).

Run with ``python tools/build_regional_conflict.py``. Downloads the regional-conflict
replication ``.dta`` (night-lights, rainfall, drought and conflict across African regions,
1994-2010), keeps the variables needed for the panel-IV example, renames them to readable
snake_case, and writes ``regional_conflict.parquet`` + ``regional_conflict_data_def.parquet``
under ``expdpy/data``.

The headline example instruments **night-time lights** (a proxy for local economic activity)
with lagged **rainfall** and **drought**, absorbing region and year fixed effects — the panel
analogue of Stata's ``xtivreg2 ... fe`` and the canonical dataset for
:func:`expdpy.analyze_panel_iv_regression`. The ``*_dt`` model variables are region-detrended
(region-specific linear trends removed), matching the published specification.
"""

from __future__ import annotations

import io
import urllib.request
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "src" / "expdpy" / "data"
URL = (
    "https://raw.githubusercontent.com/cmg777/starter-academic-v501/master/"
    "content/post/stata_iv_panel/reference/EL_regional_conflict_replication.dta"
)

# new snake_case name <- source column
_RENAME = {
    "region_id": "objectid",
    "year": "year",
    "country": "countryname",
    "country_code": "countrycode",
    "conflict": "ucdp_death_dummy_dt",
    "conflict_severe": "ucdp_25death_dummy_dt",
    "conflict_raw": "ucdp_death_dummy",
    "log_lights_lag1": "llnlight01_dt",
    "log_lights": "llnlight01",
    "rain_lag2": "l2lnrain01_dt",
    "drought_lag2": "l2meanpdsi_dt",
}


def build() -> pd.DataFrame:
    with urllib.request.urlopen(URL) as resp:
        raw = pd.read_stata(io.BytesIO(resp.read()))
    out = raw[list(_RENAME.values())].rename(columns={v: k for k, v in _RENAME.items()})
    out["region_id"] = out["region_id"].astype(int)
    out["year"] = out["year"].astype(int)
    out["conflict_raw"] = out["conflict_raw"].fillna(0).astype(int)
    out["country"] = out["country"].astype(str)
    out["country_code"] = out["country_code"].astype(str)
    for c in (
        "conflict",
        "conflict_severe",
        "log_lights_lag1",
        "log_lights",
        "rain_lag2",
        "drought_lag2",
    ):
        out[c] = out[c].astype(float).round(6)
    return out.sort_values(["region_id", "year"]).reset_index(drop=True)


def build_data_def() -> pd.DataFrame:
    rows = [
        ("region_id", "Region identifier (GADM objectid)", "Region", "entity", False),
        ("year", "Year (1994-2010)", "Year", "time", False),
        ("country", "Country name", "Country", "factor", False),
        ("country_code", "ISO3 country code", "Country code", "factor", False),
        (
            "conflict",
            "Conflict onset (>=1 battle death), region-detrended",
            "Conflict",
            "numeric",
            False,
        ),
        (
            "conflict_severe",
            "Conflict onset (>=25 battle deaths), region-detrended",
            "Severe conflict",
            "numeric",
            False,
        ),
        (
            "conflict_raw",
            "Conflict onset (>=1 battle death), raw 0/1 indicator",
            "Conflict (raw)",
            "logical",
            False,
        ),
        (
            "log_lights_lag1",
            "Log night-time lights, t-1 (region-detrended; the endogenous regressor)",
            "Log lights (t-1)",
            "numeric",
            False,
        ),
        ("log_lights", "Log night-time lights (raw)", "Log lights", "numeric", False),
        (
            "rain_lag2",
            "Log rainfall, t-2 (region-detrended; an instrument)",
            "Rainfall (t-2)",
            "numeric",
            False,
        ),
        (
            "drought_lag2",
            "Drought index PDSI, t-2 (region-detrended; an instrument)",
            "Drought (t-2)",
            "numeric",
            False,
        ),
    ]
    return pd.DataFrame(
        rows, columns=["var_name", "var_def", "label", "type", "can_be_na"]
    )


def main() -> None:
    df = build()
    df.to_parquet(OUT / "regional_conflict.parquet", index=False)
    build_data_def().to_parquet(OUT / "regional_conflict_data_def.parquet", index=False)
    print(
        f"wrote {len(df)} region-years ({df['region_id'].nunique()} regions, "
        f"{df['country'].nunique()} countries, {df.shape[1]} cols) to "
        f"{OUT / 'regional_conflict.parquet'}"
    )
    print(f"size: {(OUT / 'regional_conflict.parquet').stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
