#!/usr/bin/env python
"""Build the synthetic ``kuznets`` example dataset for expdpy.

Unlike the other bundled datasets (see ``tools/build_datasets.py``), ``kuznets`` is *fully
synthetic*: a country-year panel generated so that regional inequality traces an
**N-shaped** Kuznets curve in income — it rises, falls, then rises again at very high
income. The design mirrors Table 4 of the Lessmann & Seidel (2017) regional-inequality
replication (https://carlos-mendez.org/post/python_kuznets_dmsp/data/): the same variables
(renamed to clean snake_case), generic country names, 80 countries, annual data 2015-2025.

It writes three files into ``src/expdpy/data/``:

* ``kuznets.parquet``            — the panel (880 rows x 21 cols);
* ``kuznets_data_def.parquet``  — the variable-definition table (var_name/var_def/type/can_be_na);
* ``expdpy_config_kuznets.json``— a startup config that opens the app on the N-curve.

Everything is reproducible (``numpy.random.default_rng(2025)``). Run from the repo root::

    python tools/build_kuznets.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "src" / "expdpy" / "data"

SEED = 2025
N_COUNTRIES = 80
YEARS = np.arange(2015, 2026)  # 2015..2025 inclusive (11 years)
N_YEARS = YEARS.size

# Income span of the cross-section (USD GDP per capita) and the two turning points of the
# N-shape, expressed on the natural-log income axis.
GDP_LO, GDP_HI = 700.0, 110_000.0
X1, X2 = np.log(3_000.0), np.log(30_000.0)  # local max, then local min -> "N"
GINI_LO, GINI_HI = 0.08, 0.40  # structural-mean band the cubic is rescaled into

CONTINENTS = np.array(
    ["Continent A", "Continent B", "Continent C", "Continent D", "Continent E"]
)
CONTINENT_OFFSET = np.array([-0.02, -0.01, 0.0, 0.01, 0.02])

# (column, original Table-4 name, data-def type, human description, concise display label)
SCHEMA = [
    (
        "country",
        "Country_NAME",
        "entity",
        "Country identifier (synthetic, generic names)",
        "Country",
    ),
    (
        "iso",
        "Country_ISO",
        "entity",
        "Country ISO code (synthetic, generic codes)",
        "ISO code",
    ),
    ("year", "year", "time", "Calendar year", "Year"),
    (
        "continent",
        "(new)",
        "factor",
        "Synthetic continent (grouping factor)",
        "Continent",
    ),
    (
        "gini_regional",
        "GINIW_pred_GDP_pc",
        "numeric",
        "Regional inequality Gini — the N-shaped Kuznets outcome (GINIW_pred_GDP_pc)",
        "Regional inequality (Gini)",
    ),
    (
        "gdp_pc",
        "GDP_pc_Country",
        "numeric",
        "National GDP per capita, USD (GDP_pc_Country)",
        "GDP per capita (USD)",
    ),
    (
        "population",
        "Pop_Country",
        "numeric",
        "National population (Pop_Country)",
        "Population",
    ),
    (
        "resource_rents",
        "Resources_rents_share_of_GDP",
        "numeric",
        "Natural-resource rents, % of GDP (Resources_rents_share_of_GDP)",
        "Resource rents (% of GDP)",
    ),
    (
        "arable_land",
        "Arable_land",
        "numeric",
        "Arable land share (Arable_land)",
        "Arable land (share)",
    ),
    (
        "trade_share",
        "Trade_GDP_share",
        "numeric",
        "Trade openness, trade/GDP (Trade_GDP_share)",
        "Trade openness (trade/GDP)",
    ),
    (
        "fdi_share",
        "FDI_share_of_GDP",
        "numeric",
        "FDI inflows, share of GDP (FDI_share_of_GDP)",
        "FDI inflows (% of GDP)",
    ),
    ("area", "area", "numeric", "Country area, km^2 (area)", "Area (km²)"),
    (
        "gasoline_price",
        "price_gasoline",
        "numeric",
        "Gasoline price, USD/litre (price_gasoline)",
        "Gasoline price (USD/litre)",
    ),
    (
        "aid",
        "Aid",
        "numeric",
        "Net official development aid received, USD (Aid)",
        "Official development aid (USD)",
    ),
    (
        "school_enrollment",
        "School_enrollment_secondary",
        "numeric",
        "Secondary-school enrollment, % gross (School_enrollment_secondary)",
        "Secondary enrollment (% gross)",
    ),
    (
        "gini_lights",
        "GINIW_Eth_light",
        "numeric",
        "Light-based inequality measure, control (GINIW_Eth_light)",
        "Light-based inequality (Gini)",
    ),
    (
        "polity2",
        "Polity2",
        "numeric",
        "Democracy score, -10..10 (Polity2)",
        "Democracy score (Polity2)",
    ),
    (
        "federal",
        "fedelupd2",
        "factor",
        "Federal-state dummy, 0/1 (fedelupd2)",
        "Federal state",
    ),
    (
        "log_gdp_pc",
        "(derived)",
        "numeric",
        "Natural log of gdp_pc (derived)",
        "Log GDP per capita",
    ),
    (
        "log_gdp_pc_sq",
        "(derived)",
        "numeric",
        "log_gdp_pc squared (derived)",
        "Log GDP per capita²",
    ),
    (
        "log_gdp_pc_cu",
        "(derived)",
        "numeric",
        "log_gdp_pc cubed (derived)",
        "Log GDP per capita³",
    ),
]

# Per-column missingness rate (averaged over years; early years are missed more often).
MISSING_RATES = {
    "gasoline_price": 0.25,
    "school_enrollment": 0.15,
    "gini_lights": 0.15,
    "polity2": 0.12,
    "aid": 0.05,
    "fdi_share": 0.05,
}


def _structural_gini(x: np.ndarray) -> np.ndarray:
    """Map log-income ``x`` to the structural-mean Gini via the N-shaped cubic.

    The cubic has positive leading coefficient with derivative roots at ``X1`` (local max)
    and ``X2`` (local min), so the curve rises, falls, then rises again. It is rescaled
    linearly (shape-preserving) into ``[GINI_LO, GINI_HI]`` using the income cross-section.
    """

    # f'(x) = k (x - X1)(x - X2), k > 0  =>  f(x) = x^3/3 - (X1+X2)/2 x^2 + X1 X2 x
    def cubic(z: np.ndarray) -> np.ndarray:
        return z**3 / 3 - (X1 + X2) / 2 * z**2 + (X1 * X2) * z

    grid = np.linspace(np.log(GDP_LO), np.log(GDP_HI), 1000)
    g = cubic(grid)
    g_lo, g_hi = g.min(), g.max()
    return GINI_LO + (cubic(x) - g_lo) / (g_hi - g_lo) * (GINI_HI - GINI_LO)


def _inject_missing(
    col: np.ndarray, rate: float, rng: np.random.Generator
) -> np.ndarray:
    """Set NaNs in a (n_countries, n_years) array, more often in early years."""
    factor = np.linspace(1.7, 0.3, N_YEARS)  # averages 1.0 -> overall rate ~= ``rate``
    draw = rng.random(col.shape)
    out = col.astype(float).copy()
    out[draw < rate * factor[None, :]] = np.nan
    return out


def build_frame() -> pd.DataFrame:
    """Generate the 880-row kuznets panel as a tidy DataFrame."""
    rng = np.random.default_rng(SEED)
    nc, ny = N_COUNTRIES, N_YEARS
    tt = (YEARS - YEARS[0]).astype(float)  # 0..10

    # --- country-level attributes ------------------------------------------------------
    base_log_gdp = np.linspace(np.log(GDP_LO), np.log(GDP_HI), nc) + rng.normal(
        0, 0.05, nc
    )
    pct = base_log_gdp.argsort().argsort() / (nc - 1)  # income percentile in [0, 1]
    # Heterogeneous income dynamics: a country-specific growth slope *and* curvature so
    # within-country income moves enough (and differently across countries) to identify the
    # cubic Kuznets curve *within* country, not only cross-sectionally.
    growth = np.clip(rng.normal(0.03, 0.05, nc), -0.06, 0.16)
    accel = rng.normal(0.0, 0.003, nc)
    country_re = rng.normal(0.0, 0.02, nc)
    continent_idx = np.clip(np.round(pct * 4 + rng.normal(0, 0.9, nc)), 0, 4).astype(
        int
    )

    pop_base = np.exp(rng.normal(15.5, 2.0, nc))
    pop_growth = rng.normal(0.012, 0.006, nc)
    area = np.clip(np.exp(rng.normal(11.5, 2.0, nc)), 50, 8_400_000)
    res_base = rng.gamma(1.5, 6.0, nc)
    arable_base = np.clip(rng.beta(2, 3, nc) * 0.66, 0.005, 0.66)
    trade_base = np.clip(
        np.exp(rng.normal(np.log(0.6), 0.4, nc)) + 0.05 * (pct - 0.5), 0.147, 1.734
    )
    fdi_base = rng.normal(0.03, 0.05, nc)
    aid_scale = np.exp(rng.normal(19.8, 1.2, nc))

    # federal: more likely for large-area countries; ~20% are federal.
    fed_logit = (np.log(area) - np.log(area).mean()) + rng.normal(0, 0.8, nc)
    federal = (fed_logit > np.quantile(fed_logit, 0.80)).astype(int)

    # --- panel (country x year) arrays -------------------------------------------------
    def col(*, scale: float, base: np.ndarray) -> np.ndarray:
        return base[:, None] + rng.normal(0, scale, (nc, ny))

    # Build the log-income path directly: base level + country slope*t + country curvature*t^2
    # + idiosyncratic year shock. Clipped to a sane band so the structural cubic is not
    # extrapolated far past its design range [GDP_LO, GDP_HI].
    log_gdp = np.clip(
        base_log_gdp[:, None]
        + growth[:, None] * tt[None, :]
        + accel[:, None] * (tt**2)[None, :]
        + rng.normal(0, 0.04, (nc, ny)),
        np.log(500.0),
        np.log(150_000.0),
    )
    gdp_pc = np.exp(log_gdp)
    x = log_gdp

    population = (pop_base[:, None] * (1 + pop_growth)[:, None] ** tt[None, :]).round()
    area_panel = np.repeat(area[:, None], ny, axis=1)
    resource_rents = np.clip(
        res_base[:, None] + (1 - pct)[:, None] * 12 + rng.normal(0, 2, (nc, ny)), 0, 71
    )
    arable_land = np.clip(col(scale=0.01, base=arable_base), 0.005, 0.66)
    trade_share = np.clip(col(scale=0.05, base=trade_base), 0.147, 1.734)
    fdi_share = np.clip(col(scale=0.03, base=fdi_base), -0.165, 0.403)
    gasoline_price = np.clip(
        0.2 + 0.13 * (x - np.log(GDP_LO)) + rng.normal(0, 0.1, (nc, ny)), 0.2, 1.83
    )
    aid = np.clip(
        (1.05 - pct)[:, None] * aid_scale[:, None] * (1 + rng.normal(0, 0.12, (nc, ny)))
        - 1e7,
        -4e7,
        2.2e9,
    )
    school_enrollment = np.clip(
        10 + 17 * (x - np.log(GDP_LO)) + rng.normal(0, 8, (nc, ny)), 6, 160
    )
    polity2 = np.clip(
        np.round(-6 + 2.4 * (x - np.log(GDP_LO)) + rng.normal(0, 3, (nc, ny))), -10, 10
    )

    # --- the N-shaped outcome ----------------------------------------------------------
    def z(a: np.ndarray) -> np.ndarray:
        return (a - a.mean()) / a.std()

    # A small common year effect gives the year fixed effect real work to do (and is absorbed
    # by it); the country intercept is absorbed by the country fixed effect.
    year_effect = rng.normal(0, 0.01, ny)
    structural = _structural_gini(x)
    gini_regional = np.clip(
        structural
        + country_re[:, None]
        + year_effect[None, :]
        + 0.015 * z(resource_rents)
        - 0.015 * z(trade_share)
        - 0.020 * z(school_enrollment)
        + CONTINENT_OFFSET[continent_idx][:, None]
        + rng.normal(0, 0.035, (nc, ny)),
        0.02,
        0.6,
    )
    gini_lights = np.clip(
        0.2 + 0.3 * gini_regional + rng.normal(0, 0.13, (nc, ny)), 0.01, 0.81
    )

    # --- light, early-year-tilted missingness ------------------------------------------
    gasoline_price = _inject_missing(
        gasoline_price, MISSING_RATES["gasoline_price"], rng
    )
    school_enrollment = _inject_missing(
        school_enrollment, MISSING_RATES["school_enrollment"], rng
    )
    gini_lights = _inject_missing(gini_lights, MISSING_RATES["gini_lights"], rng)
    polity2 = _inject_missing(polity2, MISSING_RATES["polity2"], rng)
    aid = _inject_missing(aid, MISSING_RATES["aid"], rng)
    fdi_share = _inject_missing(fdi_share, MISSING_RATES["fdi_share"], rng)

    log_gdp_pc = np.log(gdp_pc)

    # --- flatten (country-major: 11-year blocks per country) ---------------------------
    def flat(a: np.ndarray) -> np.ndarray:
        return a.ravel()

    df = pd.DataFrame(
        {
            "country": np.repeat([f"country {i + 1}" for i in range(nc)], ny),
            "iso": np.repeat([f"C{i + 1:02d}" for i in range(nc)], ny),
            "year": np.tile(YEARS, nc),
            "continent": np.repeat(CONTINENTS[continent_idx], ny),
            "gini_regional": flat(gini_regional),
            "gdp_pc": flat(gdp_pc),
            "population": flat(population).astype("int64"),
            "resource_rents": flat(resource_rents),
            "arable_land": flat(arable_land),
            "trade_share": flat(trade_share),
            "fdi_share": flat(fdi_share),
            "area": flat(area_panel),
            "gasoline_price": flat(gasoline_price),
            "aid": flat(aid),
            "school_enrollment": flat(school_enrollment),
            "gini_lights": flat(gini_lights),
            "polity2": flat(polity2),
            "federal": np.repeat(federal, ny).astype("int64"),
            "log_gdp_pc": flat(log_gdp_pc),
            "log_gdp_pc_sq": flat(log_gdp_pc**2),
            "log_gdp_pc_cu": flat(log_gdp_pc**3),
        }
    )
    # Guard: column order matches the declared schema exactly.
    assert list(df.columns) == [name for name, *_ in SCHEMA]
    return df


def build_data_def() -> pd.DataFrame:
    """Build the variable-definition table (matches the gapminder pattern)."""
    return pd.DataFrame(
        {
            "var_name": [name for name, *_ in SCHEMA],
            "var_def": [desc for _, _, _, desc, _ in SCHEMA],
            "label": [label for *_, label in SCHEMA],
            "type": [typ for _, _, typ, *_ in SCHEMA],
            "can_be_na": [typ not in ("entity", "time") for _, _, typ, *_ in SCHEMA],
        }
    )


def build_config() -> dict:
    """Build a startup config that opens the app on the N-shaped curve."""
    return {
        "sample": "kuznets",
        "subset_factor": "continent",
        "subset_value": "All",
        "outlier_treatment": "1",
        "outlier_factor": "None",
        "balanced_panel": False,
        "udvars": None,
        "delvars": None,
        # bar chart
        "bar_chart_var1": "continent",
        "bar_chart_var2": "federal",
        # missing values
        "missing_values_group_by": "All",
        # descriptive
        "desc_group_by": "All",
        # by-group bar / violin / trend
        "bgbg_var": "gini_regional",
        "bgbg_byvar": "continent",
        "bgbg_stat": "mean",
        "bgbg_sort_by_stat": True,
        "bgvg_var": "gini_regional",
        "bgvg_byvar": "continent",
        "bgvg_sort_by_stat": True,
        "bgtg_var": "gini_regional",
        "bgtg_byvar": "continent",
        # histogram / extreme obs
        "hist_var": "gdp_pc",
        "hist_nr_of_breaks": 30,
        "ext_obs_var": "gini_regional",
        "ext_obs_period_by": "2025",
        # trend / quantile trend
        "trend_graph_var1": "gini_regional",
        "trend_graph_var2": "gdp_pc",
        "quantile_trend_graph_var": "gini_regional",
        "quantile_trend_graph_quantiles": ["0.05", "0.25", "0.50", "0.75", "0.95"],
        # scatter — the headline N (log income on x, loess reveals rise-fall-rise)
        "scatter_x": "log_gdp_pc",
        "scatter_y": "gini_regional",
        "scatter_color": "continent",
        "scatter_size": "population",
        "scatter_loess": True,
        "scatter_sample": False,
        # regression — statistical evidence of the cubic N, controlling for the panel's
        # two-way (country + year) fixed effects, with SEs clustered by country (FE 1).
        "reg_y": "gini_regional",
        "reg_x": ["log_gdp_pc", "log_gdp_pc_sq", "log_gdp_pc_cu"],
        "reg_fe1": "country",
        "reg_fe2": "year",
        "reg_by": "None",
        "cluster": "2",
        "model": "ols",
    }


def build() -> None:
    """Generate and write the three kuznets artifacts into ``src/expdpy/data/``."""
    OUT.mkdir(parents=True, exist_ok=True)

    df = build_frame()
    df.to_parquet(OUT / "kuznets.parquet", index=False)
    print(f"wrote kuznets.parquet  ({df.shape[0]} rows x {df.shape[1]} cols)")

    dd = build_data_def()
    dd.to_parquet(OUT / "kuznets_data_def.parquet", index=False)
    print(f"wrote kuznets_data_def.parquet  ({dd.shape[0]} rows)")

    cfg = build_config()
    (OUT / "expdpy_config_kuznets.json").write_text(json.dumps(cfg, indent=2) + "\n")
    print("wrote expdpy_config_kuznets.json")


if __name__ == "__main__":
    build()
