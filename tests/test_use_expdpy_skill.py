"""Tests for the use-expdpy agent skill and its generator (``tools/build_use_skill``).

The skill prose is hand-authored, so these guard it against drift: every function and topic
name it cites must be real, its frontmatter must be valid, the generated catalog and hub page
must be in sync, and the helper script must be read-only.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

import expdpy

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import build_use_skill as bus  # noqa: E402

SKILL_DIR = REPO / ".claude" / "skills" / "use-expdpy"
SKILL_MD = SKILL_DIR / "SKILL.md"
SCRIPT = SKILL_DIR / "scripts" / "check_panel.py"
PROSE_FILES = [SKILL_MD, *sorted((SKILL_DIR / "references").glob("*.md"))]
ALL_PROSE = "\n".join(p.read_text() for p in PROSE_FILES)


def _parse_frontmatter(text: str) -> tuple[str, str]:
    assert text.startswith("---\n"), "SKILL.md must start with YAML frontmatter"
    end = text.index("\n---\n", 4)
    fm = text[4:end]
    name = ""
    desc_lines: list[str] = []
    capture = False
    for line in fm.splitlines():
        if line.startswith("name:"):
            name = line.split(":", 1)[1].strip()
            capture = False
        elif line.startswith("description:"):
            capture = True
        elif capture:
            desc_lines.append(line.strip())
    return name, " ".join(d for d in desc_lines if d)


def test_frontmatter_is_valid() -> None:
    """The skill name and description obey the Agent Skills constraints."""
    name, description = _parse_frontmatter(SKILL_MD.read_text())
    assert name == "use-expdpy"
    assert re.fullmatch(r"[a-z0-9-]{1,64}", name)
    assert "anthropic" not in name and "claude" not in name
    assert 0 < len(description) <= 1024
    # States both WHAT it does and WHEN to use it, and carries the guardrail framing.
    assert "should be used" in description.lower()
    assert "associat" in description.lower()


def test_every_referenced_function_exists() -> None:
    """Every explore_/analyze_/learn_ name the prose calls is exported (anti-drift)."""
    idents = set(re.findall(r"`?([a-z_]+)\(", ALL_PROSE))
    idents |= set(re.findall(r"xp\.([a-z_]+)\(", ALL_PROSE))
    analysis = {i for i in idents if i.startswith(("explore_", "analyze_", "learn_"))}
    assert analysis, "expected the prose to reference some functions"
    missing = sorted(n for n in analysis if n not in expdpy.__all__)
    assert not missing, f"prose references non-exported functions: {missing}"


def test_every_referenced_topic_exists() -> None:
    """Every explain("<topic>") the prose cites is a registered concept key."""
    topics_used = set(re.findall(r'explain\("([a-z_]+)"\)', ALL_PROSE))
    registered = set(expdpy.list_topics())
    missing = sorted(t for t in topics_used if t not in registered)
    assert not missing, f"prose references unregistered topics: {missing}"


def test_guardrails_present_in_skill() -> None:
    """The skill teaches the association/entity-time guardrails and the result contract."""
    body = SKILL_MD.read_text().lower()
    assert "associations, not causation" in body
    assert "entity" in body and "time" in body
    assert ".interpret()" in body


def test_catalog_is_fresh() -> None:
    """The committed function catalog equals its regeneration."""
    assert bus.CATALOG.read_text() == bus.render_catalog()


def test_hub_page_is_fresh() -> None:
    """The committed hub page equals its regeneration from SKILL.md."""
    assert bus.HUB.read_text() == bus.render_hub_page()


def test_hub_page_has_no_executable_cells() -> None:
    """The hub page uses display-only fences so nothing executes at quarto render."""
    assert "```{python}" not in bus.HUB.read_text()


def test_check_panel_script_is_read_only(tmp_path: Path) -> None:
    """The helper reports panel candidates and writes nothing."""
    csv = tmp_path / "d.csv"
    pd.DataFrame(
        {"country": ["a", "a", "b", "b"], "year": [1, 2, 1, 2], "x": [1.0, 2, 3, 4]}
    ).to_csv(csv, index=False)
    before = sorted(p.name for p in tmp_path.iterdir())

    proc = subprocess.run(
        [sys.executable, str(SCRIPT), str(csv)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr
    assert "Inferred panel dimensions" in proc.stdout
    after = sorted(p.name for p in tmp_path.iterdir())
    assert before == after, "check_panel.py must not write any files"
