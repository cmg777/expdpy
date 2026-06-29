"""End-to-end smoke tests for the MCP server (require the optional ``mcp`` extra).

The whole module is skipped if the SDK is unavailable. The estimation tests are marked
``mcp`` so they can be deselected explicitly; they run real expdpy functions over bundled
datasets (so they exercise pyfixest/linearmodels JIT codepaths — run under the pixi env,
not the numba-free uv ``.venv``).
"""

from __future__ import annotations

import pytest

pytest.importorskip("mcp")

import mcp.types as types

from expdpy._meta import param_schema, tool_description, tool_specs
from expdpy.mcp._adapter import (
    learn_concept_description,
    learn_concept_schema,
)
from expdpy.mcp.server import _dispatch, build_server


def test_build_server_smoke() -> None:
    """The server constructs and is named expdpy."""
    server = build_server()
    assert server.name == "expdpy"


def test_advertised_tools_accept_our_schemas() -> None:
    """Every curated schema (+ learn_concept) is a valid MCP Tool; total is the registry + 1."""
    tools = [
        types.Tool(
            name=spec.name,
            description=tool_description(spec),
            inputSchema=param_schema(spec),
        )
        for spec in tool_specs()
    ]
    tools.append(
        types.Tool(
            name="learn_concept",
            description=learn_concept_description(),
            inputSchema=learn_concept_schema(),
        )
    )
    names = {t.name for t in tools}
    assert "analyze_iv_regression" in names
    assert "learn_concept" in names
    assert len(tools) == len(tool_specs()) + 1


def test_list_topics_dispatch() -> None:
    """The discovery tool lists the concept registry."""
    text = _dispatch("list_topics", {})
    assert "fixed_effects" in text
    assert "correlation_vs_causation" in text


def test_explain_dispatch() -> None:
    """``explain`` returns the rendered explainer for a topic."""
    text = _dispatch("explain", {"topic": "instrumental_variables"})
    assert "instrumental" in text.lower()


@pytest.mark.mcp
def test_iv_dispatch_over_colonial_origins() -> None:
    """The flagship IV tool runs end-to-end on the AJR dataset and reports the diagnostics."""
    text = _dispatch(
        "analyze_iv_regression",
        {
            "data": {"dataset": "colonial_origins"},
            "dv": "log_gdp_pc_1995",
            "endog": "expropriation_risk",
            "instruments": "log_settler_mortality",
        },
    )
    assert "## analyze_iv_regression" in text
    assert "tidy()" in text and "glance()" in text
    assert "first_stage_f" in text
    # Guardrail: the see-also pointer and no causal phrasing in the rendered output.
    assert "instrumental_variables" in text
    assert "causes" not in text.lower()


@pytest.mark.mcp
def test_scatter_dispatch_writes_a_figure(tmp_path, monkeypatch) -> None:
    """An Explore plot tool writes a figure file and returns its path."""
    monkeypatch.setenv("EXPDPY_MCP_FIGDIR", str(tmp_path))
    monkeypatch.setenv("EXPDPY_MCP_FIGFORMAT", "html")
    text = _dispatch(
        "explore_scatter_plot",
        {"data": {"dataset": "gapminder"}, "x": "gdpPercap", "y": "lifeExp"},
    )
    assert "Figure(s)" in text
    figures = list(tmp_path.glob("*.html"))
    assert figures, "expected a figure file to be written"
    assert str(figures[0]) in text


@pytest.mark.mcp
def test_beta_convergence_dispatch_over_kuznets() -> None:
    """A panel estimator resolves the panel from the bundled data dictionary and runs."""
    text = _dispatch(
        "analyze_beta_convergence",
        {"data": {"dataset": "kuznets"}, "var": "log_gdp_pc"},
    )
    assert "## analyze_beta_convergence" in text
    assert "glance()" in text


@pytest.mark.mcp
def test_learn_concept_dispatch(tmp_path, monkeypatch) -> None:
    """The learn_concept dispatcher runs a sandbox and reports its reading."""
    monkeypatch.setenv("EXPDPY_MCP_FIGDIR", str(tmp_path))
    text = _dispatch("learn_concept", {"topic": "omitted_variable_bias"})
    assert "## learn_omitted_variable_bias" in text
