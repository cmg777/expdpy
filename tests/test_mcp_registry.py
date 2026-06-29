"""SDK-free tests for the MCP package: import-guarding and the learn_concept dispatcher.

These do NOT require the optional ``mcp`` extra. Two subprocess tests prove that the core
package and the adapter import cleanly even when the MCP SDK is absent (a meta-path finder
blocks ``mcp``), and that the console entry point degrades to a friendly install hint.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

import expdpy
from expdpy.mcp import _adapter

# A snippet that blocks the optional ``mcp`` SDK before importing anything expdpy.
_BLOCK_MCP = """
import sys
class _Block:
    def find_spec(self, name, path=None, target=None):
        if name == "mcp" or name.startswith("mcp."):
            raise ModuleNotFoundError("No module named %r" % name, name=name)
        return None
sys.meta_path.insert(0, _Block())
for _m in [m for m in sys.modules if m == "mcp" or m.startswith("mcp.")]:
    del sys.modules[_m]
"""


def _run(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_core_and_adapter_import_without_mcp_sdk() -> None:
    """``import expdpy`` and the SDK-free MCP modules work without the extra installed."""
    proc = _run(
        _BLOCK_MCP
        + "import expdpy\n"
        + "import expdpy.mcp\n"
        + "import expdpy.mcp._adapter\n"
        + "import expdpy.mcp._data\n"
        + "import expdpy.mcp._render\n"
        + "print('SDK_FREE_OK')\n"
    )
    assert proc.returncode == 0, proc.stderr
    assert "SDK_FREE_OK" in proc.stdout


def test_console_entry_point_hints_to_install_extra() -> None:
    """``expdpy-mcp`` exits with an install hint when the SDK is missing."""
    proc = _run(
        _BLOCK_MCP
        + "from expdpy.mcp.__main__ import main\n"
        + "try:\n"
        + "    main()\n"
        + "except SystemExit as e:\n"
        + "    print('GUARD:', e)\n"
    )
    assert "expdpy[mcp]" in (proc.stdout + proc.stderr)


def test_learn_topics_cover_all_sandboxes() -> None:
    """The learn_concept enum lists every ``learn_*`` sandbox (prefix-stripped)."""
    expected = sorted(
        n[len("learn_") :] for n in expdpy.__all__ if n.startswith("learn_")
    )
    assert _adapter.learn_topics() == expected
    assert len(expected) >= 13


def test_learn_concept_schema_is_valid() -> None:
    """The learn_concept tool advertises a valid, topic-required JSON Schema."""
    jsonschema = pytest.importorskip("jsonschema")
    schema = _adapter.learn_concept_schema()
    jsonschema.Draft202012Validator.check_schema(schema)
    assert schema["required"] == ["topic"]
    assert schema["properties"]["topic"]["enum"] == _adapter.learn_topics()


def test_learn_concept_description_is_guardrailed() -> None:
    """The dispatcher description carries the association-not-causation framing."""
    desc = _adapter.learn_concept_description().lower()
    assert "associat" in desc
    assert "causes" not in desc and "effect of" not in desc
