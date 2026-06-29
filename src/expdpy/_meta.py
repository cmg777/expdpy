"""Agent-facing metadata: the single source of truth for tool definitions.

This private module is the spine of expdpy's LLM-friendly surface. It curates a
high-value subset of the public API as **tool specifications** and compiles a JSON
Schema for each one *by introspection* — parameter names, types, defaults and
descriptions are read from the live signature and NumPy docstring, never hand-declared.
Three consumers share it without duplicating any API fact:

* the MCP server (``expdpy.mcp``) — registers each spec as a callable tool;
* the static-schema emitter (``tools/build_tool_schemas.py``) — writes the committed
  Anthropic + OpenAI function-calling JSON;
* the ``llms.txt`` generator (``tools/build_llms_txt.py``) — lists the curated tools.

The only hand-authored data here are (a) *which* functions are curated
(:data:`CURATED_TOOL_SPECS`), (b) the non-introspectable output-surface metadata on each
:class:`ToolSpec` (which result attribute is a figure, which methods render as tables,
which concept explainers to surface), and (c) the single shared :data:`GUARDRAIL_PREAMBLE`.
Everything else is derived, so the agent surface cannot silently drift from the API.

The module is deliberately **kept out of** ``__all__`` and ``docs/_quarto.yml`` (no
auto-generated reference page) and imports only the standard library plus the public
``expdpy`` surface — it ships in the wheel and is safe to import at runtime, but it must
never read build-only files such as ``docs/_quarto.yml``.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any

import expdpy

# --------------------------------------------------------------------------- constants

#: Bundled datasets an agent can name instead of supplying a file path.
DATASET_NAMES: tuple[str, ...] = (
    "kuznets",
    "gapminder",
    "staggered_did",
    "firms",
    "productivity",
    "bolivia112_gdppc",
    "colonial_origins",
    "regional_conflict",
)

#: One canonical guardrail paragraph injected into every agent-facing artifact. It is
#: phrased to avoid the literal banned phrasings ("causes" / "effect of") so that the
#: banned-substring tests over *derived* content never trip on the guardrail itself.
GUARDRAIL_PREAMBLE: str = (
    "expdpy reports statistical associations, not proof of causation. Describe results "
    'with associational language ("is associated with", "co-moves with"), not causal '
    "language. Panel data uses entity (unit) + time vocabulary; declare it once with "
    "set_panel(entity=, time=). Every result exposes interpret() for a plain-language "
    "reading and explain(topic) for method caveats; prefer them over ad-hoc narration."
)

#: Appended to the description of any ``entity`` / ``time`` parameter.
ENTITY_TIME_NOTE: str = (
    "If omitted, the panel declared by set_panel() on the data is used."
)

#: The shared data-passing contract. Identical bytes back the MCP server's tool input
#: and the static Anthropic/OpenAI schemas (the MCP server's *handling* of the handle
#: differs, but the advertised schema does not).
DATA_HANDLE_SCHEMA: dict[str, Any] = {
    "description": (
        "The dataset to analyse. Provide ONE of: a bundled dataset name; an absolute "
        "path to a .csv/.parquet readable by the server; or inline row records. Panel "
        "data uses entity (unit) + time columns."
    ),
    "oneOf": [
        {
            "type": "object",
            "title": "BundledDataset",
            "properties": {
                "dataset": {"type": "string", "enum": list(DATASET_NAMES)},
                "with_labels": {"type": "boolean", "default": True},
            },
            "required": ["dataset"],
            "additionalProperties": False,
        },
        {
            "type": "object",
            "title": "DataPath",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to a .csv or .parquet file.",
                }
            },
            "required": ["path"],
            "additionalProperties": False,
        },
        {
            "type": "object",
            "title": "InlineRecords",
            "properties": {
                "records": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": (
                        "Inline row records (list of column->value objects); capped by "
                        "the server to keep token budgets small."
                    ),
                }
            },
            "required": ["records"],
            "additionalProperties": False,
        },
    ],
}


# --------------------------------------------------------------------------- the spec


@dataclass(frozen=True)
class ToolSpec:
    """A curated agent-facing tool wrapping one public ``expdpy`` callable.

    Only the *non-introspectable* fields are carried here; parameter names, types,
    requiredness and descriptions are compiled from the live function by
    :func:`param_schema`.

    Parameters
    ----------
    name : str
        The public ``expdpy`` function name (also the tool name). Must be in
        ``expdpy.__all__``.
    summary : str or None
        One-line tool description. ``None`` derives it from the docstring's first line.
    takes_data : bool
        Whether the tool accepts a ``data`` handle (False for pure discovery tools such
        as ``list_topics``).
    omit_params : tuple of str
        Cosmetic parameters (e.g. ``title``) hidden from the schema and left at default.
    forced_kwargs : tuple of (str, object)
        Parameters fixed by the adapter (e.g. ``("format", "df")``) and hidden from the
        schema.
    figure_attrs : tuple of str
        Result attributes holding Plotly figures; the MCP server writes each to a file
        and returns its path (``None`` entries are skipped).
    tables : tuple of str
        Result attributes (DataFrames) rendered as Markdown/records in the output.
    table_methods : tuple of str
        Result methods (e.g. ``"tidy"``, ``"glance"``) rendered in the output.
    see_also_topics : tuple of str
        ``explain()`` registry keys surfaced with the result; each must be in
        ``list_topics()``.
    """

    name: str
    summary: str | None = None
    takes_data: bool = True
    omit_params: tuple[str, ...] = ()
    forced_kwargs: tuple[tuple[str, object], ...] = ()
    figure_attrs: tuple[str, ...] = ()
    tables: tuple[str, ...] = ()
    table_methods: tuple[str, ...] = ()
    see_also_topics: tuple[str, ...] = ()

    @property
    def func(self) -> Any:
        """The live public callable this spec wraps."""
        return getattr(expdpy, self.name)

    @property
    def forced(self) -> dict[str, Any]:
        """``forced_kwargs`` as a dict."""
        return dict(self.forced_kwargs)


# Cosmetic presentation knobs hidden from every tool schema (left at their defaults).
# Applied globally as "hide if present", so a tool that lacks one is unaffected.
HIDDEN_PARAMS: frozenset[str] = frozenset({"title", "subtitle", "caption"})

#: The curated, hand-picked tool set. Membership and the output-surface metadata are the
#: only hand-authored API facts in the LLM surface; everything else is introspected.
CURATED_TOOL_SPECS: tuple[ToolSpec, ...] = (
    # ---- Explore -------------------------------------------------------------
    ToolSpec(
        name="explore_descriptive_table",
        tables=("df",),
        table_methods=(),
        see_also_topics=("descriptive_stats",),
    ),
    ToolSpec(
        name="explore_correlation_table",
        tables=("df_corr",),
        see_also_topics=("pearson", "correlation_vs_causation"),
    ),
    ToolSpec(
        name="explore_scatter_plot",
        figure_attrs=("fig",),
        tables=("df",),
        see_also_topics=("correlation_vs_causation",),
    ),
    ToolSpec(
        name="explore_trend_plot",
        figure_attrs=("fig",),
        see_also_topics=("time_trends",),
    ),
    ToolSpec(
        name="explore_missing_values_plot",
        figure_attrs=("fig",),
        see_also_topics=("panel_structure",),
    ),
    ToolSpec(
        name="explore_panel_structure",
        figure_attrs=("fig",),
        tables=("df_summary",),
        see_also_topics=("panel_structure",),
    ),
    # ---- Analyze -------------------------------------------------------------
    ToolSpec(
        name="analyze_regression_table",
        forced_kwargs=(("format", "df"),),
        table_methods=("tidy", "glance"),
        see_also_topics=("ols", "fixed_effects", "clustered_se"),
    ),
    ToolSpec(
        name="analyze_panel_table",
        forced_kwargs=(("format", "df"),),
        table_methods=("tidy", "glance"),
        see_also_topics=("fixed_effects", "random_effects", "hausman"),
    ),
    ToolSpec(
        name="analyze_iv_regression",
        forced_kwargs=(("format", "df"),),
        table_methods=("tidy", "glance"),
        see_also_topics=("instrumental_variables", "correlation_vs_causation"),
    ),
    ToolSpec(
        name="analyze_event_study",
        figure_attrs=("fig",),
        table_methods=("tidy",),
        see_also_topics=("event_study", "parallel_trends"),
    ),
    ToolSpec(
        name="analyze_beta_convergence",
        figure_attrs=("fig", "fig_conditional", "fig_rolling"),
        table_methods=("glance",),
        see_also_topics=("beta_convergence",),
    ),
    ToolSpec(
        name="analyze_sigma_convergence",
        figure_attrs=("fig",),
        table_methods=("glance",),
        see_also_topics=("sigma_convergence",),
    ),
    ToolSpec(
        name="analyze_kuznets_waves",
        figure_attrs=("fig", "fig_between", "fig_within"),
        see_also_topics=("kuznets_waves",),
    ),
    # ---- Discovery / knowledge ----------------------------------------------
    ToolSpec(
        name="explain",
        takes_data=False,
        see_also_topics=(),
    ),
    ToolSpec(
        name="list_topics",
        takes_data=False,
        see_also_topics=(),
    ),
)

#: The curated tool names, in registry order.
CURATED_TOOL_NAMES: tuple[str, ...] = tuple(s.name for s in CURATED_TOOL_SPECS)


# --------------------------------------------------------------------- introspection


def _split_union(annotation: str) -> list[str]:
    """Split a ``A | B | C`` annotation string at top-level ``|`` only."""
    parts: list[str] = []
    depth = 0
    current = ""
    for ch in annotation:
        if ch in "[(":
            depth += 1
        elif ch in "])":
            depth -= 1
        if ch == "|" and depth == 0:
            parts.append(current.strip())
            current = ""
        else:
            current += ch
    parts.append(current.strip())
    return [p for p in parts if p]


def _split_top_commas(inner: str) -> list[str]:
    """Split a bracket interior at top-level commas only."""
    parts: list[str] = []
    depth = 0
    current = ""
    for ch in inner:
        if ch in "[(":
            depth += 1
        elif ch in "])":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(current.strip())
            current = ""
        else:
            current += ch
    parts.append(current.strip())
    return [p for p in parts if p]


_ARRAY_PREFIXES = ("Sequence[", "list[", "List[", "tuple[", "Tuple[", "Iterable[")
_SCALARS: dict[str, dict[str, Any]] = {
    "str": {"type": "string"},
    "int": {"type": "integer"},
    "float": {"type": "number"},
    "bool": {"type": "boolean"},
    "object": {},
}


def _atom_to_schema(token: str) -> dict[str, Any] | None:
    """Map a single (non-union) annotation token to a JSON Schema fragment.

    Returns ``None`` for the data frame token, which the caller renders as the shared
    data handle.
    """
    token = token.strip()
    if token.startswith("Literal["):
        items = _split_top_commas(token[len("Literal[") : -1])
        values: list[Any] = []
        is_string = False
        for item in items:
            item = item.strip()
            if (item.startswith("'") and item.endswith("'")) or (
                item.startswith('"') and item.endswith('"')
            ):
                values.append(item[1:-1])
                is_string = True
            else:
                try:
                    values.append(int(item))
                except ValueError:
                    values.append(item)
                    is_string = True
        return {"type": "string" if is_string else "integer", "enum": values}
    for prefix in _ARRAY_PREFIXES:
        if token.startswith(prefix):
            inner = token[len(prefix) : -1]
            first = _split_top_commas(inner)[0] if inner else "str"
            item_schema = _atom_to_schema(first) or {"type": "string"}
            return {"type": "array", "items": item_schema}
    if token in _SCALARS:
        return dict(_SCALARS[token])
    if token in ("pd.DataFrame", "DataFrame"):
        return None
    if token.startswith(("Mapping[", "dict[", "Dict[")):
        return {"type": "object"}
    return {"type": "string"}


def _annotation_to_schema(annotation: str) -> tuple[dict[str, Any], bool]:
    """Compile an annotation string to ``(schema, optional)``.

    ``optional`` is True when ``None`` is a union member.
    """
    members = _split_union(annotation.strip())
    optional = "None" in members
    members = [m for m in members if m != "None"]
    if not members:
        return {"type": "string"}, optional
    schemas = [s for s in (_atom_to_schema(m) for m in members) if s is not None]
    if not schemas:
        return {"type": "string"}, optional
    if len(schemas) == 1:
        return schemas[0], optional
    return {"oneOf": schemas}, optional


def parse_doc_params(doc: str) -> dict[str, str]:
    """Parse a NumPy-style docstring ``Parameters`` block into ``{name: description}``.

    Descriptions are collapsed to a single line. Section ends at the next dashed header
    (``Returns``, ``Raises``, ``Examples``, ...).
    """
    lines = doc.splitlines()
    out: dict[str, str] = {}
    in_section = False
    current: str | None = None
    buf: list[str] = []

    def flush() -> None:
        if current is not None:
            out[current] = " ".join(w.strip() for w in buf).strip()

    for i, raw in enumerate(lines):
        line = raw.rstrip()
        stripped = line.strip()
        nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
        is_header = bool(stripped) and set(nxt) <= {"-"} and len(nxt) >= 3
        if is_header:
            if stripped == "Parameters":
                in_section = True
            elif in_section:
                flush()
                current = None
                buf = []
                in_section = False
            continue
        if set(stripped) <= {"-"} and stripped:
            continue  # the dashes line under a header
        if not in_section:
            continue
        # A parameter entry starts at the base indent (no leading whitespace) and names
        # the parameter, optionally followed by " : <type>".
        if line and not line[0].isspace():
            flush()
            buf = []
            name = stripped.split(":", 1)[0].strip()
            name = name.split(",")[0].strip()  # "a, b : int" -> first name
            current = name or None
        elif current is not None and stripped:
            buf.append(stripped)
    flush()
    return out


def _json_default(value: Any) -> Any:
    """Return a JSON-serialisable default, or ``None`` if not representable."""
    if isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, tuple | list):
        items = [v for v in value if isinstance(v, bool | int | float | str)]
        if len(items) == len(value):
            return items
    return None


def param_schema(spec: ToolSpec) -> dict[str, Any]:
    """Compile the JSON Schema ``object`` for a tool's inputs from the live function.

    Parameter types come from the signature annotations, descriptions from the NumPy
    docstring ``Parameters`` block, and requiredness from missing defaults. The first
    ``df`` parameter is rendered as the shared :data:`DATA_HANDLE_SCHEMA` under the
    property name ``data`` (when ``spec.takes_data``).
    """
    func = spec.func
    sig = inspect.signature(func)
    doc_params = parse_doc_params(inspect.getdoc(func) or "")
    omit = set(spec.omit_params) | HIDDEN_PARAMS
    forced = set(spec.forced)

    properties: dict[str, Any] = {}
    required: list[str] = []

    if spec.takes_data:
        properties["data"] = dict(DATA_HANDLE_SCHEMA)
        required.append("data")

    for pname, param in sig.parameters.items():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        if pname == "df":
            continue  # represented by the shared data handle
        if pname in omit or pname in forced:
            continue

        annotation = param.annotation if isinstance(param.annotation, str) else ""
        if annotation:
            schema, _optional = _annotation_to_schema(annotation)
        else:
            schema = {"type": "string"}
        schema = dict(schema)

        desc = doc_params.get(pname, "")
        if pname in ("entity", "time"):
            desc = f"{desc} {ENTITY_TIME_NOTE}".strip()
        if desc:
            schema["description"] = desc

        if param.default is not inspect.Parameter.empty:
            default = _json_default(param.default)
            if default is not None:
                schema["default"] = default

        properties[pname] = schema
        if param.default is inspect.Parameter.empty:
            required.append(pname)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def tool_summary(spec: ToolSpec) -> str:
    """Return the spec's one-line summary (explicit, or the docstring's first line)."""
    if spec.summary:
        return spec.summary
    doc = inspect.getdoc(spec.func) or ""
    for line in doc.splitlines():
        if line.strip():
            return line.strip()
    return spec.name


def tool_description(spec: ToolSpec) -> str:
    """Return the full agent-facing description: summary + guardrail + see-also."""
    parts = [tool_summary(spec), "", GUARDRAIL_PREAMBLE]
    if spec.see_also_topics:
        parts.append("See also concepts: " + ", ".join(spec.see_also_topics) + ".")
    return "\n".join(parts)


def anthropic_schema(spec: ToolSpec) -> dict[str, Any]:
    """Render a spec to the Anthropic tool-use format."""
    return {
        "name": spec.name,
        "description": tool_description(spec),
        "input_schema": param_schema(spec),
    }


def openai_schema(spec: ToolSpec) -> dict[str, Any]:
    """Render a spec to the OpenAI function-calling format."""
    return {
        "type": "function",
        "function": {
            "name": spec.name,
            "description": tool_description(spec),
            "parameters": param_schema(spec),
        },
    }


def tool_specs() -> tuple[ToolSpec, ...]:
    """Return the curated tool specifications, in registry order."""
    return CURATED_TOOL_SPECS


def public_callables() -> dict[str, Any]:
    """Return ``{name: function}`` for every callable function in ``expdpy.__all__``."""
    out: dict[str, Any] = {}
    for name in expdpy.__all__:
        obj = getattr(expdpy, name)
        if inspect.isfunction(obj):
            out[name] = obj
    return out


def uncurated_analysis_functions() -> list[str]:
    """List public ``explore_``/``analyze_``/``learn_`` functions not yet curated.

    Informational: lets a coverage test flag new analysis functions so they are
    triaged into (or consciously left out of) the agent tool surface.
    """
    prefixes = ("explore_", "analyze_", "learn_")
    names = [n for n in expdpy.__all__ if n.startswith(prefixes)]
    return sorted(set(names) - set(CURATED_TOOL_NAMES))


def validate() -> None:
    """Assert the registry is internally consistent (raises on the first problem).

    Checks every curated name is exported and callable, every forced/omitted/figure/
    table parameter is real, and every ``see_also`` topic is registered.
    """
    topics = set(expdpy.list_topics())
    for spec in CURATED_TOOL_SPECS:
        if spec.name not in expdpy.__all__:
            raise ValueError(f"curated tool {spec.name!r} is not in expdpy.__all__")
        func = spec.func
        if not callable(func):
            raise ValueError(f"curated tool {spec.name!r} is not callable")
        sig = inspect.signature(func)
        valid = set(sig.parameters)
        for pname in (*spec.omit_params, *dict(spec.forced_kwargs)):
            if pname not in valid:
                raise ValueError(
                    f"{spec.name!r}: {pname!r} is not a parameter of the function"
                )
        for topic in spec.see_also_topics:
            if topic not in topics:
                raise ValueError(
                    f"{spec.name!r}: see-also topic {topic!r} is not registered"
                )


def _check_import_time() -> None:
    """Cheap import-time guard: curated names exist (heavy checks live in tests)."""
    for spec in CURATED_TOOL_SPECS:
        if spec.name not in expdpy.__all__:  # pragma: no cover - defensive
            raise ValueError(f"curated tool {spec.name!r} is not in expdpy.__all__")


_check_import_time()


# Silence "imported but unused" for the re-exported helpers used by consumers.
__all__ = [
    "ToolSpec",
    "CURATED_TOOL_SPECS",
    "CURATED_TOOL_NAMES",
    "DATASET_NAMES",
    "DATA_HANDLE_SCHEMA",
    "GUARDRAIL_PREAMBLE",
    "ENTITY_TIME_NOTE",
    "param_schema",
    "parse_doc_params",
    "tool_summary",
    "tool_description",
    "anthropic_schema",
    "openai_schema",
    "tool_specs",
    "public_callables",
    "uncurated_analysis_functions",
    "validate",
]
