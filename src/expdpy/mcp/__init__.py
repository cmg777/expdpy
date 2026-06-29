"""The expdpy MCP server (optional ``expdpy[mcp]`` extra).

Exposes a curated subset of expdpy as Model Context Protocol tools over stdio, so an
agent host (Claude Desktop, Claude Code, ...) can call expdpy directly. Each tool returns
the result's ``.interpret()`` prose, its tables, and a saved figure path.

The MCP SDK is imported lazily, so ``import expdpy.mcp`` succeeds even when the optional
extra is not installed; building or running the server is what requires it. Launch with
the ``expdpy-mcp`` console script (or ``python -m expdpy.mcp``).
"""

from __future__ import annotations

from typing import Any

__all__ = ["build_server", "run", "main"]


def __getattr__(name: str) -> Any:
    """Lazily import server entry points so the SDK is only required on use."""
    if name in ("build_server", "run"):
        from . import server

        return getattr(server, name)
    if name == "main":
        from .__main__ import main

        return main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
