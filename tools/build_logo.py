#!/usr/bin/env python
"""Rasterise the expdpy logo/icon assets from the canonical SVGs.

The two hand-authored SVGs under ``src/expdpy/_assets`` are the single source of truth:

- ``logo.svg``    — the N-shaped Kuznets-curve mark in expdpy blue (``#1f77b4``) on a
                    transparent background (Quarto navbar / README).
- ``favicon.svg`` — the same mark in white on a solid rounded blue tile, which stays legible
                    at browser-tab sizes (favicon / Streamlit page icon).

This script renders the PNG copies with ``rsvg-convert`` and copies the web-facing files into
``docs/images/`` so the docs site, the README, and the apps all share one design. Re-run it
whenever a source SVG changes and commit the generated outputs — there is no build-time
dependency on ``rsvg-convert``::

    python tools/build_logo.py
    # or: pixi run build-logo
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ASSETS = REPO / "src" / "expdpy" / "_assets"
DOCS_IMAGES = REPO / "docs" / "images"

LOGO_SVG = ASSETS / "logo.svg"
FAVICON_SVG = ASSETS / "favicon.svg"


def _rsvg() -> str:
    """Locate ``rsvg-convert`` or exit with an install hint."""
    exe = shutil.which("rsvg-convert")
    if exe is None:
        sys.exit(
            "rsvg-convert not found on PATH — install librsvg "
            "(e.g. `brew install librsvg` or `apt install librsvg2-bin`)."
        )
    return exe


def render(svg: Path, out: Path, size: int) -> None:
    """Rasterise ``svg`` to a square ``size``x``size`` PNG at ``out``."""
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [_rsvg(), "-w", str(size), "-h", str(size), str(svg), "-o", str(out)],
        check=True,
    )
    print(f"  rendered {out.relative_to(REPO)}  ({size}px)")


def copy(src: Path, dest: Path) -> None:
    """Copy ``src`` to ``dest``, creating parent directories as needed."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dest)
    print(f"  copied   {dest.relative_to(REPO)}")


def build() -> None:
    """Generate every derived logo asset from the canonical SVGs."""
    print("Packaged rasters (src/expdpy/_assets):")
    render(LOGO_SVG, ASSETS / "logo.png", 512)  # README / general use
    render(FAVICON_SVG, ASSETS / "favicon.png", 256)  # Streamlit page icon (tab tile)

    print("Docs site copies (docs/images):")
    copy(LOGO_SVG, DOCS_IMAGES / "logo.svg")  # Quarto navbar logo
    copy(
        ASSETS / "logo.png", DOCS_IMAGES / "logo.png"
    )  # README raw.githubusercontent URL
    render(FAVICON_SVG, DOCS_IMAGES / "favicon.png", 64)  # Quarto favicon


if __name__ == "__main__":
    build()
