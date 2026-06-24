#!/usr/bin/env python
"""Generate splot-style source-listing pages for the API reference.

quartodoc has no built-in "view source" link (only an unmerged draft PR), so this script
exposes the implementation of every documented function on the docs site itself — mirroring
Sphinx's ``viewcode`` / splot's ``_modules`` pages. For each Python module that *defines* a
documented public object it writes ``docs/reference/modules/<dotted.module>.qmd`` containing
the **full module source**, reproduced verbatim and split into sections anchored at each
top-level definition. The companion :mod:`build_reference_enrichment` adds a ``[source]`` link
on each reference page that jumps to the matching anchor here. (The directory is ``modules/``,
not ``_modules/``: Quarto silently skips any path beginning with an underscore.)

The reference directory (``docs/reference/``) is gitignored and regenerated on every build, so
these pages are build artifacts — run this *after* ``quartodoc build`` and *before*
``quarto render``::

    pixi run -e docs python tools/build_source_pages.py

The defining module of each object is discovered automatically from the ``quartodoc.sections``
of ``docs/_quarto.yml`` (no hand-maintained list), so new functions are picked up for free.
"""

from __future__ import annotations

import ast
import importlib
import inspect
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
QUARTO_YML = REPO / "docs" / "_quarto.yml"
MODULES_DIR = REPO / "docs" / "reference" / "modules"

# Public objects whose example cannot be executed headless (they launch a Streamlit UI). They
# still get a ``[source]`` link and a source page; only example execution is skipped. Shared
# with :mod:`build_reference_enrichment`.
NO_EXECUTE: frozenset[str] = frozenset({"ExploreApp", "AnalyzeApp", "LearnApp"})


@dataclass(frozen=True)
class DocObj:
    """A documented public object, resolved to its defining source."""

    fqn: str  # fully-qualified name as it appears in the reference H1 anchor
    name: str  # the defined name (anchor on the source page; reference-page slug)
    module: str  # defining dotted module, e.g. ``expdpy.convergence``
    sourcefile: Path  # absolute path to the file that defines it


@dataclass(frozen=True)
class Section:
    """One slice of a module's source: an anchored definition, or the leading preamble."""

    name: (
        str | None
    )  # ``None`` for the module preamble (imports / docstring / module code)
    code: str  # exact source lines for this slice (newline-terminated)


# --------------------------------------------------------------------------- resolution


def _iter_section_specs() -> list[tuple[str | None, list[str]]]:
    """Read ``(package, [names])`` for every ``quartodoc`` section in ``_quarto.yml``."""
    import yaml  # lazy — keeps the pure helpers importable without PyYAML

    config = yaml.safe_load(QUARTO_YML.read_text())
    specs: list[tuple[str | None, list[str]]] = []
    for section in config["quartodoc"]["sections"]:
        package = section.get("package")
        names = [c for c in section.get("contents", []) if isinstance(c, str)]
        if names:
            specs.append((package, names))
    return specs


def load_doc_objects() -> list[DocObj]:
    """Resolve every documented object to its defining module and source file."""
    objects: list[DocObj] = []
    seen: set[str] = set()
    for package, names in _iter_section_specs():
        base = importlib.import_module(package or "expdpy")
        for name in names:
            obj = getattr(base, name)
            target = inspect.unwrap(obj)  # see through functools.lru_cache etc.
            try:
                sourcefile = Path(inspect.getsourcefile(target) or "")
            except TypeError:
                continue  # not introspectable (e.g. a C builtin) — no source page
            fqn = f"{package or 'expdpy'}.{name}"
            if fqn in seen:
                continue
            seen.add(fqn)
            objects.append(
                DocObj(
                    fqn=fqn,
                    name=getattr(target, "__name__", name),
                    module=target.__module__,
                    sourcefile=sourcefile.resolve(),
                )
            )
    return objects


def source_link(obj: DocObj) -> str:
    """Relative URL (from a ``reference/<slug>.html`` page) to ``obj``'s anchored source."""
    return f"modules/{obj.module}.html#{obj.name}"


# --------------------------------------------------------------------------- rendering


def _node_start(node: ast.stmt) -> int:
    """0-based line index where a definition begins, including any decorators."""
    decorators = getattr(node, "decorator_list", None)
    if decorators:
        return min(d.lineno for d in decorators) - 1
    return node.lineno - 1  # type: ignore[attr-defined]


def split_module_sections(source: str) -> list[Section]:
    """Split ``source`` into a leading preamble plus one section per top-level definition.

    Every line of ``source`` lands in exactly one section, in order, so that
    ``"".join(s.code for s in split_module_sections(source)) == source``. Each definition's
    section spans from its first decorator (or ``def``/``class`` line) up to the start of the
    next definition, capturing any trailing module-level code with it.
    """
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    defs = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]
    if not defs:
        return [Section(name=None, code=source)]

    starts = [_node_start(node) for node in defs]
    sections = [Section(name=None, code="".join(lines[: starts[0]]))]
    for i, node in enumerate(defs):
        end = starts[i + 1] if i + 1 < len(defs) else len(lines)
        sections.append(Section(name=node.name, code="".join(lines[starts[i] : end])))
    return sections


def render_module_page(module: str, sourcefile: Path, public_names: set[str]) -> str:
    """Render the ``.qmd`` source page for one module."""
    source = sourcefile.read_text()
    try:
        rel = sourcefile.relative_to(REPO).as_posix()
        location = (
            f"[`{rel}`](https://github.com/cmg777/expdpy/blob/main/{rel}) on GitHub"
        )
    except ValueError:  # a file outside the repo (e.g. a unit-test fixture)
        location = f"`{sourcefile.name}`"

    out: list[str] = [
        "---",
        f'title: "{module}"',
        "search: false",
        "code-tools: false",
        "---",
        "",
        f"Full source of `{module}` — {location}. "
        "Each top-level definition is anchored; reach it from the **source** link on a "
        "reference page.",
        "",
    ]
    for section in split_module_sections(source):
        if section.name is None:
            if section.code.strip():
                out += ["```python", section.code.rstrip("\n"), "```", ""]
            continue
        # The ``{ #id }`` attribute must end the heading line, or Quarto falls back to an
        # auto-generated id and the reference page's ``[source]`` anchor would not resolve. The
        # "back to docs" link therefore goes on its own line below the heading.
        out += [f"## `{section.name}` {{ #{section.name} }}", ""]
        if section.name in public_names:
            out += [
                f"[↩ docs](../{section.name}.html){{.viewcode-backlink}}",
                "",
            ]
        out += ["```python", section.code.rstrip("\n"), "```", ""]
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------- main


def main() -> None:
    """Generate one ``_modules/<module>.qmd`` page per defining module."""
    objects = load_doc_objects()
    by_module: dict[str, dict[str, object]] = {}
    for obj in objects:
        entry = by_module.setdefault(
            obj.module, {"sourcefile": obj.sourcefile, "names": set()}
        )
        entry["names"].add(obj.name)  # type: ignore[union-attr]

    MODULES_DIR.mkdir(parents=True, exist_ok=True)
    for module, entry in sorted(by_module.items()):
        sourcefile: Path = entry["sourcefile"]  # type: ignore[assignment]
        public_names: set[str] = entry["names"]  # type: ignore[assignment]
        page = render_module_page(module, sourcefile, public_names)
        (MODULES_DIR / f"{module}.qmd").write_text(page)

    print(
        f"build_source_pages: wrote {len(by_module)} module source page(s) to "
        f"{MODULES_DIR.relative_to(REPO)}"
    )


if __name__ == "__main__":
    main()
