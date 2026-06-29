"""Tests for ``expdpy._meta`` — the single source of truth for agent tool schemas.

These lock the anti-drift contract: the curated tool registry must reference only real
exported functions, its hand-authored output-surface metadata must match the live result
dataclasses, and every compiled JSON Schema must be valid and faithful to the signature.
They import only ``expdpy`` (no MCP SDK, no Quarto), so they run in the default env.
"""

from __future__ import annotations

import inspect

import pytest

import expdpy
from expdpy import _meta

BANNED = ("causes", "effect of")
SPECS = _meta.tool_specs()
SPEC_IDS = [s.name for s in SPECS]


def test_validate_passes() -> None:
    """The registry is internally consistent."""
    _meta.validate()


def test_curated_set_size() -> None:
    """The curated surface stays small enough for an agent to reason over."""
    assert 8 <= len(SPECS) <= 16
    assert len(SPEC_IDS) == len(set(SPEC_IDS))  # unique names


@pytest.mark.parametrize("spec", SPECS, ids=SPEC_IDS)
def test_curated_name_is_exported_callable(spec: _meta.ToolSpec) -> None:
    """Every curated tool maps to an exported, callable function."""
    assert spec.name in expdpy.__all__
    assert callable(spec.func)


@pytest.mark.parametrize("spec", SPECS, ids=SPEC_IDS)
def test_forced_and_omit_params_are_real(spec: _meta.ToolSpec) -> None:
    """Hand-authored param references resolve against the live signature."""
    valid = set(inspect.signature(spec.func).parameters)
    for pname in (*spec.omit_params, *spec.forced):
        assert pname in valid, f"{spec.name}: {pname} not a parameter"


@pytest.mark.parametrize("spec", SPECS, ids=SPEC_IDS)
def test_see_also_topics_registered(spec: _meta.ToolSpec) -> None:
    """Every see-also concept key is in the explainer registry."""
    topics = set(expdpy.list_topics())
    for topic in spec.see_also_topics:
        assert topic in topics, f"{spec.name}: topic {topic} not registered"


@pytest.mark.parametrize("spec", SPECS, ids=SPEC_IDS)
def test_output_surface_matches_result_dataclass(spec: _meta.ToolSpec) -> None:
    """figure_attrs / tables / table_methods refer to real result members.

    This is the drift guard for the non-introspectable output metadata: it must track
    the live result dataclass returned by the function.
    """
    if not (spec.figure_attrs or spec.tables or spec.table_methods):
        return
    ret_name = inspect.signature(spec.func).return_annotation
    result_cls = getattr(expdpy, ret_name)
    fields = set(getattr(result_cls, "__dataclass_fields__", {}))
    for attr in (*spec.figure_attrs, *spec.tables):
        assert attr in fields, f"{spec.name}: {attr} not a field of {ret_name}"
    for meth in spec.table_methods:
        assert hasattr(result_cls, meth), f"{spec.name}: {ret_name} has no {meth}()"


@pytest.mark.parametrize("spec", SPECS, ids=SPEC_IDS)
def test_required_equals_params_without_defaults(spec: _meta.ToolSpec) -> None:
    """`required` is exactly the visible parameters that have no default."""
    schema = _meta.param_schema(spec)
    sig = inspect.signature(spec.func)
    hidden = _meta.HIDDEN_PARAMS | set(spec.omit_params) | set(spec.forced)
    expected = {
        name
        for name, p in sig.parameters.items()
        if name != "df"
        and name not in hidden
        and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
        and p.default is inspect.Parameter.empty
    }
    if spec.takes_data:
        expected.add("data")
    assert set(schema["required"]) == expected


@pytest.mark.parametrize("spec", SPECS, ids=SPEC_IDS)
def test_param_schema_is_valid_jsonschema(spec: _meta.ToolSpec) -> None:
    """Each compiled input schema is a valid JSON Schema document."""
    jsonschema = pytest.importorskip("jsonschema")
    jsonschema.Draft202012Validator.check_schema(_meta.param_schema(spec))


@pytest.mark.parametrize("spec", SPECS, ids=SPEC_IDS)
def test_data_handle_is_the_shared_constant(spec: _meta.ToolSpec) -> None:
    """Data-bearing tools advertise exactly the shared data handle (no second source)."""
    schema = _meta.param_schema(spec)
    if spec.takes_data:
        assert schema["properties"]["data"] == _meta.DATA_HANDLE_SCHEMA
        assert "data" in schema["required"]
    else:
        assert "data" not in schema["properties"]


@pytest.mark.parametrize("spec", SPECS, ids=SPEC_IDS)
def test_descriptions_carry_guardrails_without_banned_phrasings(
    spec: _meta.ToolSpec,
) -> None:
    """Every tool description teaches associations and never uses causal phrasings."""
    desc = _meta.tool_description(spec).lower()
    assert "associat" in desc
    assert "interpret(" in desc
    for phrase in BANNED:
        assert phrase not in desc, f"{spec.name}: banned phrase {phrase!r}"


def test_both_schema_envelopes() -> None:
    """Anthropic and OpenAI wrappers differ only in the envelope, share the body."""
    spec = next(s for s in SPECS if s.name == "analyze_iv_regression")
    a = _meta.anthropic_schema(spec)
    o = _meta.openai_schema(spec)
    assert set(a) == {"name", "description", "input_schema"}
    assert o["type"] == "function"
    assert set(o["function"]) == {"name", "description", "parameters"}
    assert a["input_schema"] == o["function"]["parameters"]
    assert a["description"] == o["function"]["description"]
    # The flagship IV example: required is exactly the no-default arguments.
    assert a["input_schema"]["required"] == ["data", "dv", "endog", "instruments"]


def test_discovery_tools_have_no_data_handle() -> None:
    """Pure knowledge tools take no data; ``explain`` requires a topic."""
    explain = _meta.param_schema(next(s for s in SPECS if s.name == "explain"))
    assert "data" not in explain["properties"]
    assert explain["required"] == ["topic"]
    topics = _meta.param_schema(next(s for s in SPECS if s.name == "list_topics"))
    assert topics["properties"] == {}
    assert topics["required"] == []


def test_guardrail_preamble_is_clean() -> None:
    """The shared preamble avoids the banned phrasings so derived-content tests can keep it."""
    low = _meta.GUARDRAIL_PREAMBLE.lower()
    assert "associat" in low
    assert "interpret(" in low
    for phrase in BANNED:
        assert phrase not in low


def test_parse_doc_params_extracts_descriptions() -> None:
    """The NumPy ``Parameters`` parser keys descriptions by name and stops at the next header."""
    doc = (
        "Summary line.\n\n"
        "Parameters\n----------\n"
        "alpha : int\n    The first one.\n"
        "beta : str | None\n    The second,\n    wrapped across lines.\n\n"
        "Returns\n-------\nThing\n    Not a parameter.\n"
    )
    params = _meta.parse_doc_params(doc)
    assert params["alpha"] == "The first one."
    assert params["beta"] == "The second, wrapped across lines."
    assert "Returns" not in params and "Thing" not in params


def test_uncurated_functions_is_a_clean_report() -> None:
    """Coverage triage lists uncurated analysis functions (no curated name leaks in)."""
    uncurated = _meta.uncurated_analysis_functions()
    assert set(uncurated).isdisjoint(_meta.CURATED_TOOL_NAMES)
    assert all(n.startswith(("explore_", "analyze_", "learn_")) for n in uncurated)
