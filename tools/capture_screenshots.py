#!/usr/bin/env python
"""Regenerate the Streamlit app screenshots used in the docs, on the kuznets dataset.

Launches the Streamlit ``ExPdPy`` app headless on the bundled ``kuznets`` panel with its
preset config (so it opens on the N-shaped Kuznets curve), then drives a headless Chromium via
Playwright to capture the three pages embedded in ``docs/tutorials/using-streamlit.qmd``:

* ``docs/images/streamlit-overview.png``      — Overview & Data
* ``docs/images/streamlit-correlations.png``  — Correlations & Scatter (the N-curve)
* ``docs/images/streamlit-regression.png``    — Regression (the cubic)

Prerequisites: the ``streamlit`` extra and Playwright's Chromium::

    python -m playwright install chromium

Run from the repo root::

    python tools/capture_screenshots.py
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
IMAGES = REPO / "docs" / "images"
PORT = 8791
# (url path — "" is the default page, file name)
PAGES = [
    ("", "streamlit-overview.png"),
    ("correlations", "streamlit-correlations.png"),
    ("regression", "streamlit-regression.png"),
]


def _wait_for_server(url: str, timeout: float = 90.0) -> None:
    deadline = time.time() + timeout
    last_err: Exception | None = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    return
        except Exception as exc:
            last_err = exc
        time.sleep(1.0)
    raise TimeoutError(f"streamlit server not ready at {url}: {last_err}")


def _launch_app() -> subprocess.Popen:
    """Write the kuznets bundle, set the handover env, and start streamlit headless."""
    from expdpy.data import get_config, load_kuznets, load_kuznets_data_def
    from expdpy.streamlit_app import ExPdPy

    cmd = ExPdPy(
        load_kuznets(),
        df_def=load_kuznets_data_def(),
        config_list=get_config("kuznets"),
        df_name="Kuznets",
        run=False,  # writes the bundle + sets the env var, returns the command
        headless=True,
        port=PORT,
    )
    env = dict(os.environ)
    env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    env["STREAMLIT_SERVER_HEADLESS"] = "true"
    return subprocess.Popen(cmd, env=env)


def _capture(out_dir: Path) -> None:
    from playwright.sync_api import sync_playwright

    out_dir.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(
            viewport={"width": 1600, "height": 1200}, device_scale_factor=2
        )
        for url_path, fname in PAGES:
            page.set_viewport_size({"width": 1600, "height": 1200})
            page.goto(
                f"http://localhost:{PORT}/{url_path}",
                wait_until="networkidle",
                timeout=90_000,
            )
            page.wait_for_selector('[data-testid="stAppViewContainer"]', timeout=90_000)
            # Let the analysis run and Plotly/tables paint.
            with contextlib.suppress(Exception):
                page.wait_for_selector(
                    ".stPlotlyChart, [data-testid='stDataFrame'], [data-testid='stTable']",
                    timeout=90_000,
                )
            page.wait_for_timeout(7_000)
            # Streamlit's main area scrolls internally, so `full_page` is cropped. Grow the
            # viewport to the main content's height, then screenshot (keeps the sidebar in shot).
            content_h = page.evaluate(
                "() => { const e = document.querySelector('[data-testid=\"stMainBlockContainer\"]')"
                " || document.querySelector('[data-testid=\"stMain\"]');"
                " return e ? Math.ceil(e.scrollHeight) : document.body.scrollHeight; }"
            )
            height = min(max(int(content_h) + 160, 1000), 6000)
            page.set_viewport_size({"width": 1600, "height": height})
            page.wait_for_timeout(2_000)
            page.screenshot(path=str(out_dir / fname), full_page=False)
            print(f"captured {fname} ({1600}x{height})")
        browser.close()


def main() -> int:
    proc = _launch_app()
    try:
        _wait_for_server(f"http://localhost:{PORT}/_stcore/health")
        _capture(IMAGES)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
    return 0


if __name__ == "__main__":
    sys.exit(main())
