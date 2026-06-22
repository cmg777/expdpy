#!/usr/bin/env bash
# Deterministic quality gate for a newly written expdpy function.
#
# Runs the same checks CI does, in order: ruff format (check), ruff lint, mypy, pytest.
# Every stage runs even if an earlier one fails (so all problems surface in one pass); the
# script exits non-zero if any stage failed, printing a FAILED line per failing stage.
#
# Usage (from the repo root):
#   bash .claude/skills/write-function/scripts/verify.sh            # full fast suite
#   bash .claude/skills/write-function/scripts/verify.sh tests/test_<fn>.py   # one test file
#
# The optional argument narrows pytest to a path/expression; ruff and mypy always run on the
# whole tree because new code routinely touches shared modules (_types, __init__, pedagogy).
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$REPO_ROOT" || { echo "could not cd to repo root: $REPO_ROOT"; exit 2; }

PYTEST_TARGET="${1:-}"
status=0

run() {
  echo ""
  echo "=== $1 ==="
  shift
  if ! "$@"; then
    echo ">>> FAILED: $*"
    status=1
    return 1
  fi
}

# 1. Formatting (CI uses --check; fix locally with `ruff format` if this fails).
run "ruff format --check" pixi run -e lint ruff format --check src tests || true

# 2. Lint.
run "ruff check" pixi run -e lint ruff check src tests || true

# 3. Types.
run "mypy" pixi run -e lint mypy src || true

# 4. Tests (narrowed if a target was given).
if [ -n "$PYTEST_TARGET" ]; then
  run "pytest ($PYTEST_TARGET)" pixi run pytest -q "$PYTEST_TARGET" || true
else
  run "pytest (fast, excludes against_r)" pixi run pytest -q || true
fi

echo ""
if [ "$status" -eq 0 ]; then
  echo "ALL GATES PASSED"
else
  echo "ONE OR MORE GATES FAILED — see the FAILED lines above."
fi
exit "$status"
