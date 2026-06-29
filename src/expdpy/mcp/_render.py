"""Figure persistence for the MCP server.

Plotly figures cannot render as live interactive objects over MCP, so each figure is
written to a file and the absolute path is returned to the agent. HTML is the
dependency-free default (it always works headless); PNG is opt-in and best-effort.

Configuration (environment variables):

* ``EXPDPY_MCP_FIGDIR`` — directory to write figures into (kept; never auto-deleted).
  If unset, a per-session temp directory is created and removed at process exit.
* ``EXPDPY_MCP_FIGFORMAT`` — ``html`` (default), ``png`` or ``both``.
"""

from __future__ import annotations

import atexit
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any

_SESSION_DIR: Path | None = None


def _cleanup(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)


def output_dir() -> Path:
    """Return the directory figures are written to, creating it on first use."""
    global _SESSION_DIR
    env = os.environ.get("EXPDPY_MCP_FIGDIR")
    if env:
        path = Path(env).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path
    if _SESSION_DIR is None:
        _SESSION_DIR = Path(tempfile.mkdtemp(prefix="expdpy-mcp-"))
        atexit.register(_cleanup, _SESSION_DIR)
    return _SESSION_DIR


def figure_format() -> str:
    """Return the configured figure format (``html`` / ``png`` / ``both``)."""
    return os.environ.get("EXPDPY_MCP_FIGFORMAT", "html").lower()


def _write_html(fig: Any, path: Path) -> None:
    fig.write_html(str(path), include_plotlyjs="cdn", full_html=True)


def save_figure(fig: Any, *, stem: str) -> list[str]:
    """Write ``fig`` to the output dir; return the absolute path(s).

    HTML is always written when requested; PNG is attempted only when configured and
    falls back to HTML if the static-image backend (kaleido) is unavailable.
    """
    out = output_dir()
    fmt = figure_format()
    suffix = uuid.uuid4().hex[:8]
    paths: list[str] = []

    if fmt in ("html", "both"):
        html_path = out / f"{stem}-{suffix}.html"
        _write_html(fig, html_path)
        paths.append(str(html_path))

    if fmt in ("png", "both"):
        png_path = out / f"{stem}-{suffix}.png"
        try:
            fig.write_image(str(png_path))
            paths.append(str(png_path))
        except Exception:
            if not paths:  # png-only request: still hand back something viewable
                html_path = out / f"{stem}-{suffix}.html"
                _write_html(fig, html_path)
                paths.append(str(html_path))

    return paths
