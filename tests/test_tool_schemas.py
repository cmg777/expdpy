"""Tests for the static Anthropic/OpenAI tool-schema emitter (``tools/build_tool_schemas``).

They validate the emitted schemas and assert the committed ``schemas/*.json`` are in sync
with their regeneration (an in-test mirror of the ``artifacts-fresh`` CI drift gate), so a
docstring or signature change that is not regenerated fails fast.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import expdpy
from expdpy import _meta

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import build_tool_schemas as bts  # noqa: E402

BANNED = ("causes", "effect of")
EXPECTED_NAMES = list(_meta.CURATED_TOOL_NAMES)


def test_committed_schemas_match_regeneration() -> None:
    """The committed JSON equals a fresh render (the drift guarantee)."""
    for path, text in bts.render().items():
        assert path.exists(), f"missing committed schema {path}"
        assert path.read_text() == text, f"{path.name} is stale; rerun the emitter"


def test_check_mode_reports_up_to_date() -> None:
    """``--check`` returns 0 when the committed files are current."""
    assert bts.main(["--check"]) == 0


def test_files_are_ascii_and_valid_json() -> None:
    """Emitted files parse as JSON and contain only ASCII bytes."""
    for path in (bts.ANTHROPIC_PATH, bts.OPENAI_PATH):
        raw = path.read_bytes()
        raw.decode("ascii")  # raises if any non-ASCII byte slipped in
        json.loads(raw)


def test_anthropic_shape_and_parity() -> None:
    """Anthropic file: one well-formed tool per curated spec, in order."""
    tools = json.loads(bts.ANTHROPIC_PATH.read_text())["tools"]
    assert [t["name"] for t in tools] == EXPECTED_NAMES
    for tool in tools:
        assert set(tool) == {"name", "description", "input_schema"}
        assert tool["input_schema"]["type"] == "object"


def test_openai_shape_and_parity() -> None:
    """OpenAI file: one well-formed function per curated spec, in order."""
    tools = json.loads(bts.OPENAI_PATH.read_text())["tools"]
    assert [t["function"]["name"] for t in tools] == EXPECTED_NAMES
    for tool in tools:
        assert tool["type"] == "function"
        assert set(tool["function"]) == {"name", "description", "parameters"}
        assert tool["function"]["parameters"]["type"] == "object"


def test_required_subset_of_properties() -> None:
    """Every required key is an actual property (in both formats)."""
    for spec in _meta.tool_specs():
        schema = _meta.param_schema(spec)
        assert set(schema["required"]) <= set(schema["properties"])


def test_schemas_validate_against_metaschema() -> None:
    """Each input schema is a valid JSON Schema document."""
    jsonschema = pytest.importorskip("jsonschema")
    tools = json.loads(bts.ANTHROPIC_PATH.read_text())["tools"]
    for tool in tools:
        jsonschema.Draft202012Validator.check_schema(tool["input_schema"])


def test_descriptions_carry_guardrails() -> None:
    """Every tool description teaches associations and avoids causal phrasings."""
    tools = json.loads(bts.ANTHROPIC_PATH.read_text())["tools"]
    for tool in tools:
        desc = tool["description"].lower()
        assert "associat" in desc and "interpret(" in desc
        for phrase in BANNED:
            assert phrase not in desc, f"{tool['name']}: banned phrase {phrase!r}"


def test_every_tool_name_is_exported() -> None:
    """The schema files never reference a non-exported function (anti-drift)."""
    tools = json.loads(bts.ANTHROPIC_PATH.read_text())["tools"]
    for tool in tools:
        assert tool["name"] in expdpy.__all__
