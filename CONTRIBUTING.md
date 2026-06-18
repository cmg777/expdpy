# Contributing to expdpy

Thanks for your interest in improving **expdpy**! This project is a Python port of the
[ExPanDaR](https://github.com/trr266/ExPanDaR) R package.

## Development setup

> Just want to **use** expdpy? Install it from GitHub instead — see
> [Installation](README.md#installation). The steps below are for **developing** the package.

The project is managed with [pixi](https://pixi.sh):

```bash
git clone https://github.com/cmg777/expdpy
cd expdpy
pixi install                 # create the default environment
pixi run test                # run the test-suite
pixi run lint                # ruff + mypy via pre-commit
pixi run docs-build          # build the Quarto docs locally
```

If you prefer `uv`:

```bash
uv venv --python 3.12
uv pip install -e ".[app,streamlit]" pytest pytest-cov hypothesis ruff mypy pandas-stubs
uv run pytest -m "not against_r"
```

## Guidelines

- **Style**: code is formatted and linted with [ruff](https://docs.astral.sh/ruff/) and
  type-checked with [mypy](https://mypy-lang.org/). Run `pixi run lint` before pushing.
- **Docstrings**: NumPy style (rendered into the API reference by quartodoc).
- **Tests**: add tests under `tests/`. Numerical functions should ideally include a
  golden-value or `against_r` parity check (see `tests/test_vs_expandar.py`).
- **Datasets**: the bundled parquet/JSON files are generated from the R sources by
  `tools/build_datasets.py`; do not edit them by hand.

## Numerical parity with ExPanDaR

A core goal is faithful numerical parity with the original R package. The `against_r`
tests use `rpy2` to compare against ExPanDaR directly (run in the pixi `r` environment).
Frozen golden values in `tests/fixtures/goldens.json` cover the base-R-equivalent
computations so the fast test run needs no R install.

## Pull requests

1. Fork and create a feature branch.
2. Add tests and update docs.
3. Ensure `pixi run lint` and `pixi run test` pass.
4. Open a PR describing the change and referencing any related issue.
