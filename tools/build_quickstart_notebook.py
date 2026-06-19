#!/usr/bin/env python
"""Build the Google Colab notebook from the Quickstart page.

The Quickstart docs page (``docs/quickstart.qmd``) is the single source of truth for the
code-along walkthrough. This script regenerates ``notebooks/quickstart.ipynb`` from it so the
two never drift: the "Run the Quick Start in Google Colab" button on the docs site opens that
notebook straight from GitHub (https://colab.research.google.com/github/cmg777/expdpy/blob/main/notebooks/quickstart.ipynb).

The conversion is ``quarto convert`` (the canonical ``.qmd`` -> ``.ipynb`` mapping: prose ->
markdown cells, ``{python}`` blocks -> code cells) followed by light post-processing with
``nbformat``: the YAML front-matter cell is dropped and three cells are prepended — a title
cell, a GitHub ``pip install`` (with the ``panel`` extra) so the notebook is runnable from a
cold Colab runtime, and a setup cell that forces Plotly's ``colab`` renderer so every figure
draws.

``quarto`` lives in the pixi ``docs`` environment, so run this from there::

    pixi run -e docs python tools/build_quickstart_notebook.py

or simply ``pixi run build-quickstart-notebook``.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell

REPO = Path(__file__).resolve().parents[1]
SRC_QMD = REPO / "docs" / "quickstart.qmd"
OUT_IPYNB = REPO / "notebooks" / "quickstart.ipynb"

# Install with the optional ``panel`` extra (linearmodels) so the classic panel-model and
# Hausman cells run live in Colab; the rest of the toolkit comes with the core install.
INSTALL_CELL = (
    '!pip install -q "expdpy[panel] @ git+https://github.com/cmg777/expdpy.git"'
)

# Colab does not always pick a Plotly renderer that draws figures returned as the last cell
# expression, so force the dedicated "colab" renderer there. This is a no-op in Jupyter /
# nbclient (where the default renderer already produces a rich mimebundle) and is injected
# into the notebook only — the docs site keeps Quarto's own Plotly handling.
SETUP_CELL = (
    "# Ensure Plotly figures render in Google Colab (a no-op in other notebook frontends).\n"
    "import plotly.io as pio\n"
    "\n"
    "try:\n"
    "    import google.colab  # noqa: F401  (present only on Colab)\n"
    "\n"
    '    pio.renderers.default = "colab"\n'
    "except ImportError:\n"
    "    pass"
)

TITLE_CELL = (
    "# expdpy — Quickstart\n"
    "\n"
    "A cloud-runnable walkthrough of [expdpy](https://github.com/cmg777/expdpy) on the bundled "
    "`kuznets` panel. Run the install cell below first, then run the rest top to bottom.\n"
    "\n"
    "> If Colab prompts you to **restart the runtime** after the install, do so, then "
    "continue from the setup cell.\n"
    "\n"
    "This notebook mirrors the [Quickstart page](https://cmg777.github.io/expdpy/quickstart.html) "
    "of the docs."
)

KERNELSPEC = {"display_name": "Python 3", "language": "python", "name": "python3"}


def _strip_front_matter(source: str) -> str:
    """Remove a leading ``---\\n...\\n---`` YAML block, keeping any prose after it.

    quarto convert keeps the ``.qmd`` front matter *and* the intro paragraph that follows it
    in the same first cell, so dropping the whole cell would lose the intro. This strips only
    the YAML block.
    """
    match = re.match(r"\s*---\n.*?\n---\n?", source, flags=re.DOTALL)
    return source[match.end() :].lstrip("\n") if match else source


def _strip_raw_html(source: str) -> str:
    """Remove ```` ```{=html} ... ``` ```` raw blocks (site-only, e.g. the Colab banner)."""
    return re.sub(r"```\{=html\}\n.*?\n```\n?", "", source, flags=re.DOTALL).lstrip(
        "\n"
    )


def convert_with_quarto(qmd: Path, dest: Path) -> None:
    """Run ``quarto convert`` to turn ``qmd`` into the notebook ``dest``."""
    quarto = shutil.which("quarto")
    if quarto is None:
        sys.exit(
            "quarto not found on PATH — run inside the docs env:\n"
            "    pixi run -e docs python tools/build_quickstart_notebook.py\n"
            "(or: pixi run build-quickstart-notebook)"
        )
    subprocess.run(
        [quarto, "convert", str(qmd), "--output", str(dest)],
        check=True,
    )


def build() -> None:
    """Generate ``notebooks/quickstart.ipynb`` from ``docs/quickstart.qmd``."""
    with tempfile.TemporaryDirectory() as tmp:
        converted = Path(tmp) / "quickstart.ipynb"
        convert_with_quarto(SRC_QMD, converted)
        nb = nbformat.read(converted, as_version=4)

    cells = list(nb.cells)
    # The first cell shares the YAML front matter with the intro prose — strip just the YAML.
    if cells and cells[0].source.lstrip().startswith("---"):
        cells[0]["cell_type"] = "markdown"
        cells[0]["source"] = _strip_front_matter(cells[0].source)
    # Drop site-only raw-HTML blocks (the Colab banner) and any markdown cell left empty.
    for cell in cells:
        if cell.cell_type == "markdown":
            cell["source"] = _strip_raw_html(cell.source)
    cells = [c for c in cells if c.cell_type != "markdown" or c.source.strip()]

    title = new_markdown_cell(TITLE_CELL)
    install = new_code_cell(INSTALL_CELL)
    setup = new_code_cell(SETUP_CELL)
    nb.cells = [title, install, setup, *cells]

    # Reconcile cell ids with the nbformat minor version quarto emitted: ids are required from
    # 4.5+ and forbidden before it. Use fixed ids for our cells so re-runs stay byte-stable.
    if nb.nbformat_minor >= 5:
        title["id"], install["id"], setup["id"] = "title", "install", "setup"
    else:
        for cell in nb.cells:
            cell.pop("id", None)

    # A clean, Colab-friendly kernel; strip any execution counts/outputs from the conversion.
    nb.metadata["kernelspec"] = KERNELSPEC
    for cell in nb.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None

    nbformat.validate(nb)
    OUT_IPYNB.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, OUT_IPYNB)
    print(f"wrote {OUT_IPYNB.relative_to(REPO)}  ({len(nb.cells)} cells)")


if __name__ == "__main__":
    build()
