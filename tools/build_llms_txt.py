#!/usr/bin/env python
"""Generate the LLM-friendly text artifacts: ``llms.txt``, ``llms-full.txt`` and per-page ``.md``.

Following the llmstxt.org convention, this writes:

* ``docs/llms.txt`` (committed) — a curated index: a project summary carrying expdpy's
  guardrails, then H2 sections with links to the live docs for every public function,
  article, dataset, concept and the agent tool schemas.
* ``docs/_site/llms-full.txt`` (build artifact) — the full text in one file: every public
  function's signature + docstring, all concept explainers, and the article prose.
* ``docs/_site/<page>.html.md`` (build artifacts) — a clean Markdown companion beside each
  rendered HTML page, so agents fetch Markdown instead of parsing HTML.

It also copies the committed ``schemas/*.json`` into ``docs/_site/tools/`` so they serve at
``/tools/...``. Everything derives from ``expdpy.__all__``, the live signatures/docstrings,
the ``explain()`` registry and ``docs/_quarto.yml`` — never hand-maintained.

The ``_site`` outputs require ``quarto render`` to have run first (the reference HTML pages
only exist then), so the default run is the LAST docs-build step. ``--canonical-only`` writes
just the committed ``docs/llms.txt`` (for the CI freshness check, no ``_site`` needed)::

    pixi run -e docs python tools/build_llms_txt.py                 # full (after quarto render)
    pixi run -e docs python tools/build_llms_txt.py --canonical-only  # committed index only
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

from build_quickstart_notebook import _strip_front_matter, _strip_raw_html  # noqa: E402
from build_reference_enrichment import _brief_signature  # noqa: E402
from build_source_pages import DocObj, load_doc_objects  # noqa: E402

import expdpy  # noqa: E402
from expdpy._meta import GUARDRAIL_PREAMBLE, tool_specs  # noqa: E402

DOCS = REPO / "docs"
SITE = DOCS / "_site"
SCHEMA_DIR = REPO / "schemas"
BASE_URL = "https://cmg777.github.io/expdpy"

HEADER_SUMMARY = (
    f"expdpy (v{expdpy.__version__}) is a Python library for interactive panel and "
    "cross-sectional data analysis - a port of the R package ExPanDaR - organized as "
    "Explore / Analyze / Learn modules. " + GUARDRAIL_PREAMBLE
)

GUARD_LINE = 'expdpy reports associations, not causation; see explain("correlation_vs_causation").'


@dataclass(frozen=True)
class Section:
    """One quartodoc section (a group of public objects)."""

    title: str
    desc: str
    package: str | None
    names: tuple[str, ...]


@dataclass(frozen=True)
class Article:
    """A navbar documentation page (module walkthrough, article or explanation)."""

    title: str
    slug: str


# ------------------------------------------------------------------------- pure helpers


def reference_url(name: str) -> str:
    """URL of the rendered reference page for a public object."""
    return f"{BASE_URL}/reference/{name}.html"


def article_url(slug: str) -> str:
    """URL of a rendered article/explanation page."""
    return f"{BASE_URL}/{slug}.html"


def _resolve(name: str, package: str | None) -> object | None:
    try:
        module = importlib.import_module(package or "expdpy")
    except ImportError:
        return None
    return getattr(module, name, None)


def doc_summary(name: str, package: str | None = None) -> str:
    """First non-blank line of an object's docstring (its one-line summary)."""
    obj = _resolve(name, package)
    for line in (inspect.getdoc(obj) or "").splitlines():
        if line.strip():
            return line.strip()
    return ""


def _blockquote(text: str) -> str:
    return "\n".join(f"> {line}" if line else ">" for line in text.splitlines())


def _function_block(name: str, obj: object | None) -> str:
    """A full Markdown block for one function: heading + signature, link, docstring."""
    sig = _brief_signature(obj) if obj is not None else ""
    doc = inspect.getdoc(obj) or ""
    return (
        f"### {name}{sig}\n\n[Reference]({reference_url(name)})\n\n{doc}".rstrip()
        + "\n"
    )


def _article_body(slug: str) -> str:
    """Clean Markdown body of an article ``.qmd`` (front matter / raw HTML stripped)."""
    src = (DOCS / f"{slug}.qmd").read_text()
    body = _strip_raw_html(_strip_front_matter(src))
    return body.replace("```{python}", "```python").strip()


def render_llms_txt(
    sections: list[Section],
    articles: list[Article],
    topics: list[str],
) -> str:
    """Render the curated ``llms.txt`` index."""
    out: list[str] = ["# expdpy", "", _blockquote(HEADER_SUMMARY), ""]

    for sec in sections:
        out.append(f"## {sec.title}")
        if sec.desc:
            out += [sec.desc, ""]
        else:
            out.append("")
        for name in sec.names:
            summary = doc_summary(name, sec.package)
            line = f"- [{name}]({reference_url(name)})"
            if summary:
                line += f": {summary}"
            out.append(line)
        out.append("")

    out += ["## Articles", ""]
    for art in articles:
        out.append(f"- [{art.title}]({article_url(art.slug)})")
    out.append("")

    out += [
        "## Concepts",
        'Plain-language method explainers via explain("<topic>") - always associational '
        '(see "correlation_vs_causation"):',
        "",
        ", ".join(topics) + ".",
        "",
    ]

    out += [
        "## Tools for AI agents",
        "",
        f"- [Anthropic tool schemas]({BASE_URL}/tools/anthropic_tools.json): curated "
        "expdpy tools in Anthropic function-calling format",
        f"- [OpenAI tool schemas]({BASE_URL}/tools/openai_tools.json): the same tools in "
        "OpenAI function-calling format",
        f"- [Agent skill and MCP setup]({BASE_URL}/use-with-llms.html): the use-expdpy "
        "skill and the `pip install expdpy[mcp]` server",
        f"- [Full text for LLMs]({BASE_URL}/llms-full.txt): every function, docstring and "
        "concept concatenated into one file",
        "",
        "Curated agent tools: " + ", ".join(spec.name for spec in tool_specs()) + ".",
        "",
    ]
    return "\n".join(out).rstrip() + "\n"


def render_llms_full(sections: list[Section], articles: list[Article]) -> str:
    """Render the full ``llms-full.txt`` concatenation."""
    out: list[str] = [
        "# expdpy - full reference for LLMs",
        "",
        _blockquote(HEADER_SUMMARY),
        "",
        (expdpy.__doc__ or "").strip(),
        "",
    ]
    for sec in sections:
        out += [f"## {sec.title}", sec.desc, ""]
        for name in sec.names:
            out += [_function_block(name, _resolve(name, sec.package)), ""]

    out += ["## Concepts", ""]
    for topic in expdpy.list_topics():
        out += [expdpy.explain(topic).to_markdown().rstrip(), ""]

    out += ["## Articles", ""]
    for art in articles:
        out += [
            f"### {art.title}",
            "",
            f"[Article]({article_url(art.slug)})",
            "",
            _article_body(art.slug),
            "",
        ]
    return "\n".join(out).rstrip() + "\n"


def reference_page_md(name: str, obj: object | None) -> str:
    """Markdown companion for a reference page."""
    return (
        f"# {name}\n\n"
        f"> Markdown companion of {reference_url(name)} - {GUARD_LINE}\n\n"
        f"{_function_block(name, obj)}"
    )


def article_page_md(art: Article) -> str:
    """Markdown companion for an article page."""
    return (
        f"# {art.title}\n\n"
        f"> Markdown companion of {article_url(art.slug)} - {GUARD_LINE}\n\n"
        f"{_article_body(art.slug)}\n"
    )


# ----------------------------------------------------------------------------- loaders


def load_sections() -> list[Section]:
    """Read the quartodoc sections from ``docs/_quarto.yml`` (title, desc, package, names)."""
    import yaml  # lazy: keeps the pure render helpers importable without PyYAML

    config = yaml.safe_load((DOCS / "_quarto.yml").read_text())
    sections: list[Section] = []
    for section in config["quartodoc"]["sections"]:
        names = tuple(c for c in section.get("contents", []) if isinstance(c, str))
        if not names:
            continue
        desc = " ".join((section.get("desc") or "").split())
        sections.append(
            Section(
                title=section.get("title", ""),
                desc=desc,
                package=section.get("package"),
                names=names,
            )
        )
    return sections


def load_articles() -> list[Article]:
    """Read the navbar pages from ``docs/_quarto.yml`` (skipping the reference index)."""
    import yaml

    config = yaml.safe_load((DOCS / "_quarto.yml").read_text())
    nav = config["website"]["navbar"].get("left", [])
    articles: list[Article] = []

    def _add(item: dict) -> None:
        file = item.get("file", "")
        if file.endswith(".qmd") and not file.startswith("reference/"):
            articles.append(Article(title=item.get("text", file), slug=file[:-4]))

    for item in nav:
        _add(item)
        for sub in item.get("menu", []):
            _add(sub)
    return articles


# -------------------------------------------------------------------------------- main


def _write_site_artifacts(sections: list[Section], articles: list[Article]) -> None:
    (SITE / "llms-full.txt").write_text(render_llms_full(sections, articles))

    for obj in load_doc_objects():
        page = SITE / "reference" / f"{obj.name}.html.md"
        if page.parent.exists():
            page.write_text(reference_page_md(obj.name, _resolve_doc_object(obj)))

    for art in articles:
        page = SITE / f"{art.slug}.html.md"
        page.parent.mkdir(parents=True, exist_ok=True)
        page.write_text(article_page_md(art))

    tools_dir = SITE / "tools"
    tools_dir.mkdir(parents=True, exist_ok=True)
    for name in ("anthropic_tools.json", "openai_tools.json"):
        src = SCHEMA_DIR / name
        if src.exists():
            shutil.copyfile(src, tools_dir / name)


def _resolve_doc_object(obj: DocObj) -> object | None:
    module = importlib.import_module(obj.module)
    return getattr(module, obj.name, None)


def main(argv: list[str] | None = None) -> int:
    """Write ``docs/llms.txt`` and (unless ``--canonical-only``) the ``_site`` artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--canonical-only",
        action="store_true",
        help="Write only the committed docs/llms.txt (skip the _site build artifacts).",
    )
    args = parser.parse_args(argv)

    sections = load_sections()
    articles = load_articles()
    topics = list(expdpy.list_topics())

    llms_txt = render_llms_txt(sections, articles, topics)
    (DOCS / "llms.txt").write_text(llms_txt)
    print("build_llms_txt: wrote docs/llms.txt")

    if args.canonical_only:
        return 0
    if not SITE.exists():
        print(
            "build_llms_txt: docs/_site not found; run after `quarto render docs`.",
            file=sys.stderr,
        )
        return 0
    (SITE / "llms.txt").write_text(llms_txt)
    _write_site_artifacts(sections, articles)
    print("build_llms_txt: wrote _site llms-full.txt, per-page .md and /tools schemas")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
