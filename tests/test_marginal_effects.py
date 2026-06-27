"""Tests for the marginal-effects / interaction plot."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import expdpy as ex
from expdpy._types import MarginalEffectsResult


@pytest.fixture(scope="module")
def interaction_df() -> pd.DataFrame:
    """A frame with a known interaction: y = 0.5 x + 0.3 z + 0.4 (x*z) + noise."""
    rng = np.random.default_rng(0)
    n = 1500
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    g = rng.integers(0, 20, size=n)
    y = 0.5 * x + 0.3 * z + 0.4 * x * z + rng.normal(scale=0.5, size=n)
    return pd.DataFrame({"y": y, "x": x, "z": z, "g": g})


def test_marginal_effect_recovers_interaction(interaction_df):
    res = ex.analyze_marginal_effects_plot(
        interaction_df, dv="y", focal="x", moderator="z"
    )
    assert isinstance(res, MarginalEffectsResult)
    coef = {str(k): float(v) for k, v in res.model.coef().items()}
    # The interaction is recovered (~0.4) and the focal main effect (~0.5).
    assert coef["x"] == pytest.approx(0.5, abs=0.05)
    assert coef["x:z"] == pytest.approx(0.4, abs=0.05)
    # The plotted marginal effect equals b_x + b_xz * moderator at every grid point.
    grid = res.df["z"].to_numpy()
    np.testing.assert_allclose(res.df["me"], coef["x"] + coef["x:z"] * grid, atol=1e-9)
    # The band brackets the line.
    assert (res.df["ci_lower"] <= res.df["me"]).all()
    assert (res.df["me"] <= res.df["ci_upper"]).all()
    # The average marginal effect sits near the main effect (mean z ~ 0).
    assert res.ame == pytest.approx(0.5, abs=0.06)


def test_sign_flip_is_reported(interaction_df):
    # The marginal effect 0.5 + 0.4 z crosses zero near z = -1.25, inside the data range.
    res = ex.analyze_marginal_effects_plot(
        interaction_df, dv="y", focal="x", moderator="z"
    )
    text = res.interpret()
    assert "switches sign" in text
    for bad in ("causes", "caused by", "effect of", "causal effect of"):
        assert bad not in text
    assert res.explain().topic == "marginal_effects"


def test_fixed_effects_and_clusters_run(interaction_df):
    res = ex.analyze_marginal_effects_plot(
        interaction_df, dv="y", focal="x", moderator="z", feffects=["g"], clusters=["g"]
    )
    assert res.fig is not None
    assert "x:z" in {str(k) for k in res.model.coef().index}


def test_explicit_at_grid(interaction_df):
    res = ex.analyze_marginal_effects_plot(
        interaction_df, dv="y", focal="x", moderator="z", at=[-1.0, 0.0, 1.0]
    )
    assert list(res.df["z"]) == [-1.0, 0.0, 1.0]


def test_non_numeric_dv_raises(interaction_df):
    df = interaction_df.assign(label="a")
    with pytest.raises(NotImplementedError):
        ex.analyze_marginal_effects_plot(df, dv="label", focal="x", moderator="z")
