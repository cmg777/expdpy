"""Build the bundled *Colonial Origins* cross-section (Acemoglu, Johnson & Robinson 2001).

Run with ``python tools/build_colonial_origins.py``. Downloads the canonical AJR replication
``maketable*.dta`` files, merges the variables needed for the famous settler-mortality IV
example (``log GDP per capita ~ expropriation risk``, instrumented by ``log settler
mortality``), renames them to readable snake_case, and writes
``colonial_origins.parquet`` + ``colonial_origins_data_def.parquet`` under ``expdpy/data``.

This is a country-level **cross-section** (no time dimension): the canonical teaching example
for :func:`expdpy.analyze_iv_regression`. Source: Acemoglu, Johnson & Robinson (2001), "The
Colonial Origins of Comparative Development", *American Economic Review* 91(5).
"""

from __future__ import annotations

import io
import urllib.request
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "src" / "expdpy" / "data"
BASE = (
    "https://raw.githubusercontent.com/cmg777/starter-academic-v501/master/"
    "content/post/stata_iv"
)

# new snake_case name <- (source table, source column)
_RENAME = {
    "country": ("4", "shortnam"),
    "log_gdp_pc_1995": ("4", "logpgp95"),
    "expropriation_risk": ("4", "avexpr"),
    "log_settler_mortality": ("4", "logem4"),
    "latitude": ("4", "lat_abst"),
    "africa": ("4", "africa"),
    "asia": ("4", "asia"),
    "neo_europe": ("4", "rich4"),
    "european_settlement_1900": ("8", "euro1900"),
    "british_colony": ("5", "f_brit"),
    "french_colony": ("5", "f_french"),
    "catholic_1980": ("5", "catho80"),
    "muslim_1980": ("5", "muslim80"),
    "base_sample": ("4", "baseco"),
}
# 0/1 indicator columns (missing -> 0, then int).
_DUMMIES = (
    "africa",
    "asia",
    "neo_europe",
    "base_sample",
    "british_colony",
    "french_colony",
)


def _read_dta(table: str) -> pd.DataFrame:
    with urllib.request.urlopen(f"{BASE}/maketable{table}.dta") as resp:
        return pd.read_stata(io.BytesIO(resp.read()))


def build() -> pd.DataFrame:
    tables = {t: _read_dta(t) for t in {"4", "5", "8"}}
    out = pd.DataFrame()
    for new, (table, src) in _RENAME.items():
        col = tables[table].set_index("shortnam")[src] if src != "shortnam" else None
        if src == "shortnam":
            out[new] = tables["4"]["shortnam"].astype(str)
        else:
            out[new] = tables["4"]["shortnam"].map(col)
    for d in _DUMMIES:
        out[d] = out[d].fillna(0).astype(int)
    # tidy float precision for the continuous columns
    for c in out.columns:
        if c not in (*_DUMMIES, "country"):
            out[c] = out[c].astype(float).round(6)
    return out.sort_values("country").reset_index(drop=True)


def build_data_def() -> pd.DataFrame:
    rows = [
        ("country", "Country (3-letter AJR code)", "Country", "entity", False),
        (
            "log_gdp_pc_1995",
            "Log GDP per capita (PPP, 1995)",
            "Log GDP p.c. 1995",
            "numeric",
            True,
        ),
        (
            "expropriation_risk",
            "Average protection against expropriation risk, 1985-95 (0-10)",
            "Expropriation risk",
            "numeric",
            True,
        ),
        (
            "log_settler_mortality",
            "Log settler mortality (the AJR instrument)",
            "Log settler mortality",
            "numeric",
            True,
        ),
        (
            "latitude",
            "Distance from the equator (abs. latitude / 90)",
            "Latitude",
            "numeric",
            True,
        ),
        ("africa", "Located in Africa", "Africa", "logical", False),
        ("asia", "Located in Asia", "Asia", "logical", False),
        (
            "neo_europe",
            "Neo-European country (the rich-4: USA/CAN/AUS/NZL)",
            "Neo-European",
            "logical",
            False,
        ),
        (
            "european_settlement_1900",
            "European settlement share in 1900 (%)",
            "European settlement 1900",
            "numeric",
            True,
        ),
        ("british_colony", "Former British colony", "British colony", "logical", False),
        ("french_colony", "Former French colony", "French colony", "logical", False),
        (
            "catholic_1980",
            "Catholic population share, 1980 (fraction)",
            "Catholic 1980",
            "numeric",
            True,
        ),
        (
            "muslim_1980",
            "Muslim population share, 1980 (fraction)",
            "Muslim 1980",
            "numeric",
            True,
        ),
        (
            "base_sample",
            "In the AJR 64-country base (settler-mortality) sample",
            "Base sample",
            "logical",
            False,
        ),
    ]
    return pd.DataFrame(
        rows, columns=["var_name", "var_def", "label", "type", "can_be_na"]
    )


def main() -> None:
    df = build()
    df.to_parquet(OUT / "colonial_origins.parquet", index=False)
    build_data_def().to_parquet(OUT / "colonial_origins_data_def.parquet", index=False)
    base = int(df["base_sample"].sum())
    iv_ready = int(
        df[["log_gdp_pc_1995", "expropriation_risk", "log_settler_mortality"]]
        .notna()
        .all(axis=1)
        .sum()
    )
    print(
        f"wrote {len(df)} countries ({df.shape[1]} cols) to "
        f"{OUT / 'colonial_origins.parquet'}"
    )
    print(f"base-sample countries: {base}; complete IV rows: {iv_ready}")


if __name__ == "__main__":
    main()
