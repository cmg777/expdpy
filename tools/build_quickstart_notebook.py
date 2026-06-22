#!/usr/bin/env python
"""Build the Google Colab notebooks from the three module pages.

The module docs pages (``docs/explore.qmd``, ``docs/analyze.qmd``, ``docs/learn.qmd``) are the
single source of truth for the code-along walkthroughs. This script regenerates one notebook
per module under ``notebooks/`` so the two never drift — each can be opened straight from
GitHub in Google Colab.

The conversion is ``quarto convert`` (the canonical ``.qmd`` -> ``.ipynb`` mapping: prose ->
markdown cells, ``{python}`` blocks -> code cells) followed by light post-processing with
``nbformat``: the YAML front-matter is stripped and three cells are prepended — a title cell,
a GitHub ``pip install`` so the notebook is runnable from a cold Colab runtime, and a setup
cell that forces Plotly's ``colab`` renderer so every figure draws.

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
from datetime import datetime, timezone
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell

REPO = Path(__file__).resolve().parents[1]

# The docs pages that each produce a self-contained Colab notebook. The three module pages use
# the shared TITLE_TEMPLATE; feature notebooks (named after their function) supply a custom
# ``title_md`` so their first cell reads as a function guide rather than a module walkthrough.
MODULES = [
    {"slug": "explore", "title": "Explore"},
    {"slug": "analyze", "title": "Analyze"},
    {"slug": "learn", "title": "Learn"},
    {
        "slug": "analyze_beta_convergence",
        "title": "analyze_beta_convergence",
        "title_md": (
            "# analyze_beta_convergence — β-convergence for panel data\n"
            "\n"
            "_Notebook version: built {BUILD_STAMP} — re-open this notebook from GitHub if "
            "yours is older, to get the latest version._\n"
            "\n"
            "An extended **user guide** *and* a **verification harness** for "
            "`analyze_beta_convergence` from [expdpy](https://github.com/cmg777/expdpy): what "
            "it does, every argument, everything it returns, and synthetic-data checks that it "
            "recovers known parameters. Run the install cell below first, then run the rest top "
            "to bottom.\n"
            "\n"
            "> If Colab prompts you to **restart the runtime** after the install, do so, then "
            "continue from the setup cell.\n"
            "\n"
            "This notebook mirrors the [analyze_beta_convergence page]"
            "(https://cmg777.github.io/expdpy/analyze_beta_convergence.html) of the docs."
        ),
    },
    {
        "slug": "analyze_sigma_convergence",
        "title": "analyze_sigma_convergence",
        "title_md": (
            "# analyze_sigma_convergence — σ-convergence for panel data\n"
            "\n"
            "_Notebook version: built {BUILD_STAMP} — re-open this notebook from GitHub if "
            "yours is older, to get the latest version._\n"
            "\n"
            "An extended **user guide** *and* a **verification harness** for "
            "`analyze_sigma_convergence` from [expdpy](https://github.com/cmg777/expdpy): what "
            "it does, every argument, everything it returns, and synthetic-data checks that it "
            "recovers a known dispersion rate. Run the install cell below first, then run the "
            "rest top to bottom.\n"
            "\n"
            "> If Colab prompts you to **restart the runtime** after the install, do so, then "
            "continue from the setup cell.\n"
            "\n"
            "This notebook mirrors the [analyze_sigma_convergence page]"
            "(https://cmg777.github.io/expdpy/analyze_sigma_convergence.html) of the docs."
        ),
    },
]

# linearmodels (random/correlated random effects, the Hausman test) ships with the core
# install now, so a plain ``expdpy`` install runs every cell. ``numba>=0.61`` is co-resolved in
# the same pip pass because Colab pre-installs an older numba (e.g. 0.60) that caps numpy below
# 2.1, while expdpy needs numpy>=2.1 — resolving them together lands on a compatible pair
# (numpy 2.4.x + numba 0.65.x) instead of leaving Colab's stale numba to conflict. The second
# line force-refreshes only the expdpy code to the latest ``main`` commit (pip skips a git
# reinstall when the version string is unchanged, so a warm runtime would otherwise keep stale
# code).
INSTALL_CELL = (
    '!pip install -q "expdpy @ git+https://github.com/cmg777/expdpy.git" "numba>=0.61"\n'
    "!pip install -q --force-reinstall --no-deps "
    '"expdpy @ git+https://github.com/cmg777/expdpy.git"'
)

# Colab does not always pick a Plotly renderer that draws figures returned as the last cell
# expression, so force the dedicated "colab" renderer there. This is a no-op in Jupyter.
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

TITLE_TEMPLATE = (
    "# expdpy — {title} panel data\n"
    "\n"
    "_Notebook version: built {{BUILD_STAMP}} — re-open this notebook from GitHub if yours is "
    "older, to get the latest version._\n"
    "\n"
    "A cloud-runnable walkthrough of the **{title}** module of "
    "[expdpy](https://github.com/cmg777/expdpy) on the bundled `kuznets` panel. Run the install "
    "cell below first, then run the rest top to bottom.\n"
    "\n"
    "> If Colab prompts you to **restart the runtime** after the install, do so, then continue "
    "from the setup cell.\n"
    "\n"
    "This notebook mirrors the [{title} page](https://cmg777.github.io/expdpy/{slug}.html) of "
    "the docs."
)

KERNELSPEC = {"display_name": "Python 3", "language": "python", "name": "python3"}


def _strip_front_matter(source: str) -> str:
    """Remove a leading ``---\\n...\\n---`` YAML block, keeping any prose after it."""
    match = re.match(r"\s*---\n.*?\n---\n?", source, flags=re.DOTALL)
    return source[match.end() :].lstrip("\n") if match else source


def _strip_raw_html(source: str) -> str:
    """Remove ```` ```{=html} ... ``` ```` raw blocks (site-only)."""
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
    subprocess.run([quarto, "convert", str(qmd), "--output", str(dest)], check=True)


def build_one(
    slug: str, title: str, build_stamp: str, *, title_md: str | None = None
) -> Path:
    """Generate ``notebooks/<slug>.ipynb`` from ``docs/<slug>.qmd``.

    ``title_md`` overrides the shared :data:`TITLE_TEMPLATE` for the first (title) cell; it may
    contain a ``{BUILD_STAMP}`` placeholder.
    """
    src_qmd = REPO / "docs" / f"{slug}.qmd"
    out_ipynb = REPO / "notebooks" / f"{slug}.ipynb"
    with tempfile.TemporaryDirectory() as tmp:
        converted = Path(tmp) / f"{slug}.ipynb"
        convert_with_quarto(src_qmd, converted)
        nb = nbformat.read(converted, as_version=4)

    cells = list(nb.cells)
    if cells and cells[0].source.lstrip().startswith("---"):
        cells[0]["cell_type"] = "markdown"
        cells[0]["source"] = _strip_front_matter(cells[0].source)
    for cell in cells:
        if cell.cell_type == "markdown":
            cell["source"] = _strip_raw_html(cell.source)
    cells = [c for c in cells if c.cell_type != "markdown" or c.source.strip()]

    # ``.format`` above collapses the template's ``{{BUILD_STAMP}}`` to a single-braced
    # ``{BUILD_STAMP}``, which is what we substitute the build time into here.
    rendered_title = (
        title_md
        if title_md is not None
        else TITLE_TEMPLATE.format(title=title, slug=slug)
    ).replace("{BUILD_STAMP}", build_stamp)
    title_cell = new_markdown_cell(rendered_title)
    install = new_code_cell(INSTALL_CELL)
    setup = new_code_cell(SETUP_CELL)
    nb.cells = [title_cell, install, setup, *cells]

    if nb.nbformat_minor >= 5:
        title_cell["id"], install["id"], setup["id"] = "title", "install", "setup"
    else:
        for cell in nb.cells:
            cell.pop("id", None)

    nb.metadata["kernelspec"] = KERNELSPEC
    for cell in nb.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None

    nbformat.validate(nb)
    out_ipynb.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, out_ipynb)
    print(f"wrote {out_ipynb.relative_to(REPO)}  ({len(nb.cells)} cells)")
    return out_ipynb


def build() -> None:
    """Generate one Colab notebook per module from its docs page."""
    build_stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    for module in MODULES:
        build_one(
            module["slug"],
            module["title"],
            build_stamp,
            title_md=module.get("title_md"),
        )


if __name__ == "__main__":
    build()
