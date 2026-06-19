"""Best-practice hardening tests: executable doctests and the dark theme variant."""

from __future__ import annotations

import doctest

import plotly.graph_objects as go
import plotly.io as pio
import pytest

from expdpy._estimation import _formula, _vcov
from expdpy._theme import TEMPLATE_NAME_DARK, apply_default_layout
from expdpy.pedagogy import _registry


@pytest.mark.parametrize(
    "module",
    [_formula, _vcov, _registry],
    ids=["formula", "vcov", "registry"],
)
def test_doctests(module):
    """The ``>>>`` examples in the pure helper modules execute and match their output."""
    results = doctest.testmod(module, verbose=False)
    assert results.failed == 0, (
        f"{results.failed} doctest(s) failed in {module.__name__}"
    )


def test_dark_template_registered():
    assert TEMPLATE_NAME_DARK in pio.templates


def test_apply_default_layout_dark_vs_light():
    light = apply_default_layout(go.Figure())
    dark = apply_default_layout(go.Figure(), dark=True)
    # both figures are themed, and the dark variant differs from the light one
    assert light.layout.template is not None
    assert dark.layout.template is not None
    assert light.layout.template != dark.layout.template
