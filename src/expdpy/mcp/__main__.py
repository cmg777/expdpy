"""Console entry point for the expdpy MCP server (``expdpy-mcp`` / ``python -m expdpy.mcp``).

Imports the server lazily so that a missing optional ``mcp`` extra produces a friendly
install hint instead of a bare ``ModuleNotFoundError``.
"""

from __future__ import annotations


def main() -> None:
    """Launch the stdio MCP server, guiding the user to install the extra if missing."""
    try:
        from .server import run
    except ModuleNotFoundError as exc:  # the optional mcp SDK is not installed
        if (exc.name or "").split(".")[0] == "mcp":
            raise SystemExit(
                "The expdpy MCP server requires the optional 'mcp' extra. "
                "Install it with:\n\n    pip install 'expdpy[mcp]'\n"
            ) from exc
        raise
    run()


if __name__ == "__main__":
    main()
