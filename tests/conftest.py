"""Shared pytest fixtures for the expdpy test-suite."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(scope="session")
def sample_df() -> pd.DataFrame:
    """The deterministic panel shared with the R golden generator."""
    return pd.read_csv(FIXTURES / "sample.csv")


@pytest.fixture(scope="session")
def goldens() -> dict:
    """R reference values produced by ``tests/fixtures/make_goldens.R``."""
    return json.loads((FIXTURES / "goldens.json").read_text())


@pytest.fixture(scope="session")
def kuznets() -> pd.DataFrame:
    from expdpy.data import load_kuznets

    return load_kuznets()


@pytest.fixture
def messy_series() -> pd.Series:
    """A numeric series with injected inf/-inf/NaN values."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=100)
    x[3] = np.inf
    x[7] = -np.inf
    x[11] = np.nan
    return pd.Series(x, name="v")
