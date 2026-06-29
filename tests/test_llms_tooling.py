"""Tests for the llms.txt / llms-full.txt / per-page-md generator (``tools/build_llms_txt``).

These exercise the pure render helpers with hand-built sections (no Quarto, no PyYAML, no
``_site``), so they run in the default env. They lock the structure of the agent-facing
text and the guardrail framing, and assert every public function renders into the full dump.
"""

from __future__ import annotations

import sys
from pathlib import Path

import expdpy

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import build_llms_txt as blt  # noqa: E402

BANNED = ("causes", "effect of")

SECTIONS = [
    blt.Section(
        title="Analyze",
        desc="Panel estimators and causal-inference-adjacent tools.",
        package=None,
        names=("analyze_iv_regression", "analyze_beta_convergence"),
    ),
    blt.Section(
        title="Datasets",
        desc="Bundled example data.",
        package="expdpy.data",
        names=("load_kuznets",),
    ),
]
ARTICLES = [blt.Article(title="Instrumental variables", slug="analyze_iv_regression")]
TOPICS = list(expdpy.list_topics())


def test_llms_txt_header_carries_guardrails() -> None:
    """The index opens with the project title and a guardrail blockquote."""
    text = blt.render_llms_txt(SECTIONS, ARTICLES, TOPICS)
    assert text.startswith("# expdpy\n")
    head = text.split("## ", 1)[0].lower()
    assert head.startswith("# expdpy")
    assert "> " in head  # blockquote
    for token in ("entity", "time", "associat", "interpret("):
        assert token in head


def test_llms_txt_sections_links_and_tool_pointers() -> None:
    """Sections render live reference links; the tools section points at the schemas + MCP."""
    text = blt.render_llms_txt(SECTIONS, ARTICLES, TOPICS)
    assert "## Analyze" in text
    assert (
        "https://cmg777.github.io/expdpy/reference/analyze_iv_regression.html" in text
    )
    assert "## Articles" in text
    assert "https://cmg777.github.io/expdpy/analyze_iv_regression.html" in text
    assert "## Concepts" in text
    assert "## Tools for AI agents" in text
    assert "tools/anthropic_tools.json" in text
    assert "tools/openai_tools.json" in text
    assert "use-with-llms.html" in text
    assert "llms-full.txt" in text
    # The curated tool names are advertised.
    assert "analyze_iv_regression" in text.split("## Tools for AI agents", 1)[1]


def test_llms_txt_lists_every_topic() -> None:
    """Every explainer topic appears in the Concepts section."""
    text = blt.render_llms_txt(SECTIONS, ARTICLES, TOPICS)
    concepts = text.split("## Concepts", 1)[1]
    for topic in TOPICS:
        assert topic in concepts


def test_function_block_has_signature_and_reference() -> None:
    """A function block shows the brief signature, a reference link and the docstring."""
    obj = expdpy.analyze_iv_regression
    block = blt._function_block("analyze_iv_regression", obj)
    assert block.startswith("### analyze_iv_regression(")
    assert "reference/analyze_iv_regression.html" in block
    assert "Parameters" in block  # full docstring is included


def test_llms_full_contains_functions_explainers_and_article() -> None:
    """The full dump concatenates function blocks, explainers and article prose."""
    text = blt.render_llms_full(SECTIONS, ARTICLES)
    assert "### analyze_iv_regression(" in text
    assert "### load_kuznets(" in text
    # An explainer rendered from the registry.
    assert expdpy.explain("instrumental_variables").title in text
    # The article body (front matter stripped).
    assert "## Instrumental variables" in text or "### Instrumental variables" in text


def test_every_public_function_renders_in_full() -> None:
    """Every explore_/analyze_/learn_ function appears as a block in llms-full (anti-drift)."""
    names = tuple(
        n for n in expdpy.__all__ if n.startswith(("explore_", "analyze_", "learn_"))
    )
    section = [blt.Section("All", "", None, names)]
    text = blt.render_llms_full(section, [])
    for name in names:
        assert f"### {name}(" in text, f"{name} missing from llms-full.txt"


def test_reference_and_article_page_md() -> None:
    """Per-page companions have an H1, the guard line, and the content."""
    ref = blt.reference_page_md("analyze_iv_regression", expdpy.analyze_iv_regression)
    assert ref.startswith("# analyze_iv_regression\n")
    assert blt.GUARD_LINE in ref
    assert "### analyze_iv_regression(" in ref

    art = blt.article_page_md(ARTICLES[0])
    assert art.startswith("# Instrumental variables\n")
    assert blt.GUARD_LINE in art


def test_article_body_strips_front_matter() -> None:
    """``_article_body`` removes the YAML front matter and returns prose."""
    body = blt._article_body("analyze_iv_regression")
    assert body
    assert not body.lstrip().startswith("---")


def test_authored_framing_avoids_causal_phrasings() -> None:
    """The strings expdpy authors (summary, guard line) never use causal phrasings."""
    for text in (blt.HEADER_SUMMARY.lower(), blt.GUARD_LINE.lower()):
        for phrase in BANNED:
            assert phrase not in text
