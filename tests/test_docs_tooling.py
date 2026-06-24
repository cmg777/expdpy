"""Unit tests for the docs build helpers in ``tools/``.

These cover the two pure functions that power the splot-style reference pages:
``split_module_sections`` (the source ``_modules`` pages) and ``enrich_reference_text`` (the
``[source]`` link plus executable examples). They exercise pure string/AST logic only, so they
run in the default test environment without Quarto.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import build_reference_enrichment as bre  # noqa: E402
import build_source_pages as bsp  # noqa: E402

SRC_FILES = sorted((REPO / "src" / "expdpy").rglob("*.py"))

# A page matching quartodoc's generated layout (verified against the real output): an H1 with a
# fully-qualified anchor, a signature code block, a Parameters table, then Examples.
SAMPLE_PAGE = """\
# explore_descriptive_table { #expdpy.explore_descriptive_table }

```python
explore_descriptive_table(df, digits=(0, 3))
```

Report descriptive statistics.

## Parameters {.doc-section .doc-section-parameters}

| Name | Type | Description | Default |
|------|------|-------------|---------|
| df   | DataFrame | the data | _required_ |

## Examples {.doc-section .doc-section-examples}

Basic:

```python
import expdpy as ex
ex.explore_descriptive_table(df).gt
```

Advanced — reuse ``df`` from the previous block:

```python
ex.explore_descriptive_table(df, digits=(0, 2)).df.head()
```
"""

NO_EXAMPLE_PAGE = """\
# set_panel { #expdpy.set_panel }

```python
set_panel(df, entity, time)
```

Declare the panel keys.
"""


# --------------------------------------------------------------- split_module_sections


@pytest.mark.parametrize("path", SRC_FILES, ids=lambda p: str(p.relative_to(REPO)))
def test_split_reproduces_source_exactly(path: Path) -> None:
    """Concatenating every section must reproduce the file byte-for-byte."""
    source = path.read_text()
    sections = bsp.split_module_sections(source)
    assert "".join(s.code for s in sections) == source


def test_split_anchors_decorated_definitions() -> None:
    source = (
        "import functools\n"
        "\n"
        "X = 1\n"
        "\n"
        "\n"
        "@functools.cache\n"
        "def foo():\n"
        "    return X\n"
        "\n"
        "\n"
        "class Bar:\n"
        "    pass\n"
    )
    sections = bsp.split_module_sections(source)
    names = [s.name for s in sections]
    assert names == [None, "foo", "Bar"]
    foo = next(s for s in sections if s.name == "foo")
    # the decorator line travels with the function it decorates, not the preamble
    assert "@functools.cache" in foo.code
    # module-level code before the first def stays in the preamble
    assert "X = 1" in sections[0].code


def test_split_module_without_definitions_is_one_preamble() -> None:
    source = "import os\n\nVALUE = os.getcwd()\n"
    sections = bsp.split_module_sections(source)
    assert sections == [bsp.Section(name=None, code=source)]


# --------------------------------------------------------------- enrich_reference_text


def test_enrich_inserts_source_link_and_makes_examples_executable() -> None:
    url = "_modules/expdpy.tables.html#explore_descriptive_table"
    out, n = bre.enrich_reference_text(SAMPLE_PAGE, url, executable=True)

    # two example blocks became executable; the signature block did not
    assert n == 2
    assert out.count("```{python}") == 2
    assert "```python\nexplore_descriptive_table(df, digits=(0, 3))" in out

    # the source link sits below the title, before the signature
    assert f"[source]({url}){{.viewcode-link" in out
    assert out.index(".viewcode-link") < out.index(
        "explore_descriptive_table(df, digits"
    )

    # executable cells require a Jupyter engine
    assert out.startswith("---\njupyter: python3\n---\n")


def test_enrich_executable_false_links_but_does_not_execute() -> None:
    url = "_modules/expdpy.streamlit_app.html#ExploreApp"
    out, n = bre.enrich_reference_text(SAMPLE_PAGE, url, executable=False)
    assert n == 0
    assert "```{python}" not in out
    assert f"[source]({url})" in out
    assert not out.startswith("---")  # no front matter without executable cells


def test_enrich_no_examples_adds_link_only() -> None:
    url = "_modules/expdpy._panel.html#set_panel"
    out, n = bre.enrich_reference_text(NO_EXAMPLE_PAGE, url, executable=True)
    assert n == 0
    assert f"[source]({url})" in out
    assert not out.startswith("---")


def test_enrich_without_url_skips_link() -> None:
    out, n = bre.enrich_reference_text(SAMPLE_PAGE, None, executable=True)
    assert ".viewcode-link" not in out
    assert n == 2  # examples still made executable


def test_enrich_is_idempotent() -> None:
    url = "_modules/expdpy.tables.html#explore_descriptive_table"
    once, _ = bre.enrich_reference_text(SAMPLE_PAGE, url, executable=True)
    twice, n = bre.enrich_reference_text(once, url, executable=True)
    assert once == twice
    assert n == 0  # nothing left to convert on a second pass
    assert once.count(".viewcode-link") == 1


def test_enrich_leaves_doctest_blocks_static() -> None:
    # A doctest block (``>>>`` lines) is not valid Python and must stay static, while a plain
    # block in the same Examples section is still made executable.
    page = (
        "# explain { #expdpy.explain }\n"
        "\n"
        "## Examples {.doc-section .doc-section-examples}\n"
        "\n"
        "```python\n"
        ">>> import expdpy as ex\n"
        '>>> ex.explain("fixed_effects").title\n'
        "'Fixed effects'\n"
        "```\n"
        "\n"
        "```python\n"
        "import expdpy as ex\n"
        '_ = ex.explain("fixed_effects")\n'
        "```\n"
    )
    out, n = bre.enrich_reference_text(page, "_modules/x.html#explain", executable=True)
    assert n == 1  # only the plain block converts
    assert ">>> import expdpy" in out  # doctest body preserved verbatim
    # the doctest block keeps its static fence; the plain one becomes executable
    assert out.count("```{python}") == 1
    assert "```python\n>>> import expdpy" in out


def test_render_module_page_keeps_anchor_at_heading_end(tmp_path: Path) -> None:
    # The Quarto ``{ #id }`` attribute only takes effect at the very end of a heading line; if
    # anything (like the back-link) follows it, Quarto auto-generates a different id and the
    # reference page's ``[source]`` anchor breaks. Lock the heading shape against that.
    mod = tmp_path / "mod.py"
    mod.write_text('"""A module."""\n\n\ndef foo():\n    return 1\n')
    page = bsp.render_module_page("expdpy.mod", mod, {"foo"})

    assert "## `foo` { #foo }\n" in page  # attribute ends the heading line
    assert "[↩ docs](../foo.html){.viewcode-backlink}" in page  # back-link present...
    assert "{ #foo } [" not in page  # ...but on its own line, not after the attribute
    assert "def foo():\n    return 1" in page  # the source is reproduced


def test_streamlit_apps_are_marked_no_execute() -> None:
    assert {"ExploreApp", "AnalyzeApp", "LearnApp"} <= bsp.NO_EXECUTE


# --------------------------------------------------------------- index signatures


def _sig_sample(
    a: int, b: int, c: int = 1, *, entity: object = None, time: object = None
) -> int:
    """Sample callable: two required args, then optionals (one defaulted, two keyword-only)."""
    return a + b + c + bool(entity) + bool(time)


def _sig_none() -> None:
    """Sample callable with no parameters."""


def _sig_all_optional(a: int = 1, b: int = 2) -> int:
    """Sample callable whose every parameter is optional."""
    return a + b


def test_brief_signature_splot_style() -> None:
    # required args bare, then the first two optionals bracketed with a trailing ``...``
    assert bre._brief_signature(_sig_sample) == "(a, b[, c, entity, ...])"
    assert bre._brief_signature(_sig_none) == "()"
    assert bre._brief_signature(_sig_all_optional) == "([a, b])"


def test_enrich_index_text_injects_signatures() -> None:
    sample = (
        "# API Reference {.doc .doc-index}\n"
        "\n"
        "## Explore\n"
        "\n"
        "| | |\n"
        "| --- | --- |\n"
        "| [explore_histogram](explore_histogram.qmd#expdpy.explore_histogram) | "
        "Histogram of a numeric variable. |\n"
        "| [bogus_fn](bogus_fn.qmd#expdpy.bogus_fn) | Not a real function. |\n"
    )
    out, n = bre.enrich_index_text(sample)

    assert n == 1  # only the resolvable row is signed
    assert '<span class="doc-sig">(df' in out  # real signature injected after the link
    assert "Histogram of a numeric variable. |" in out  # description untouched
    # header, separator, and the unresolvable row are left as-is
    assert "| | |\n" in out
    assert "| --- | --- |\n" in out
    assert "[bogus_fn](bogus_fn.qmd#expdpy.bogus_fn) | Not a real function. |" in out
    assert "doc-sig" not in out.split("bogus_fn")[1].split("\n")[0]

    again, n2 = bre.enrich_index_text(out)  # idempotent
    assert again == out
    assert n2 == 0
