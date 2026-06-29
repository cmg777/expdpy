#!/usr/bin/env python
"""Emit static Anthropic + OpenAI function-calling tool schemas from ``expdpy._meta``.

Single source of truth: ``expdpy._meta.tool_specs()``. The same ``param_schema()``
compiler backs the live MCP server, so the committed JSON here and the MCP server's
advertised input schemas can never drift.

Usage::

    pixi run -e docs python tools/build_tool_schemas.py          # (re)write schemas/*.json
    pixi run -e docs python tools/build_tool_schemas.py --check  # CI: fail on drift

Output (committed, small, reviewable, and published to the docs site under /tools/):

* ``schemas/anthropic_tools.json`` — ``{"tools": [{name, description, input_schema}, ...]}``
* ``schemas/openai_tools.json``    — ``{"tools": [{type, function: {...}}, ...]}``

``ensure_ascii=True`` keeps the files pure ASCII, so a stray Unicode minus or smart quote
in a docstring cannot land in a committed artifact and stable byte-diffing is preserved.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from expdpy._meta import anthropic_schema, openai_schema, tool_specs

REPO = Path(__file__).resolve().parents[1]
SCHEMA_DIR = REPO / "schemas"
ANTHROPIC_PATH = SCHEMA_DIR / "anthropic_tools.json"
OPENAI_PATH = SCHEMA_DIR / "openai_tools.json"


def build_anthropic() -> list[dict]:
    """Return the curated tools in Anthropic tool-use format."""
    return [anthropic_schema(spec) for spec in tool_specs()]


def build_openai() -> list[dict]:
    """Return the curated tools in OpenAI function-calling format."""
    return [openai_schema(spec) for spec in tool_specs()]


def _dump(payload: dict) -> str:
    """Serialise deterministically (ASCII, 2-space indent, trailing newline)."""
    return json.dumps(payload, indent=2, ensure_ascii=True, sort_keys=False) + "\n"


def render() -> dict[Path, str]:
    """Return ``{path: file-text}`` for every emitted schema file."""
    return {
        ANTHROPIC_PATH: _dump({"tools": build_anthropic()}),
        OPENAI_PATH: _dump({"tools": build_openai()}),
    }


def main(argv: list[str] | None = None) -> int:
    """Write the schema files, or check them for drift with ``--check``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if any committed schema differs from its regeneration.",
    )
    args = parser.parse_args(argv)
    files = render()

    if args.check:
        stale = [
            path
            for path, text in files.items()
            if not path.exists() or path.read_text() != text
        ]
        if stale:
            names = "\n".join(f"  - {p.relative_to(REPO)}" for p in stale)
            print(
                "Stale tool schemas (run `python tools/build_tool_schemas.py`):\n"
                + names,
                file=sys.stderr,
            )
            return 1
        print("build_tool_schemas: schemas are up to date.")
        return 0

    SCHEMA_DIR.mkdir(parents=True, exist_ok=True)
    for path, text in files.items():
        path.write_text(text)
    print(
        f"build_tool_schemas: wrote {len(files)} schema file(s) to "
        f"{SCHEMA_DIR.relative_to(REPO)} ({len(tool_specs())} tools each)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
