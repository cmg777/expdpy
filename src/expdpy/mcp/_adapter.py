"""Generic dispatch from an MCP tool call to an expdpy result, rendered as text.

One adapter serves every curated tool: resolve the data handle, call the public function
with the validated arguments, then assemble a text payload from the frozen result —
``.interpret()`` prose, the configured tables (DataFrame attributes / ``tidy()``/``glance()``
methods), and a saved figure path per Plotly figure. Discovery tools (``explain`` /
``list_topics``) and the ``learn_concept`` sandbox dispatcher live here too.

The output always carries expdpy's association-not-causation framing, mirroring the
``.interpret()`` contract, so an agent reading a tool result inherits the same discipline.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

import expdpy
from expdpy._meta import GUARDRAIL_PREAMBLE, ToolSpec

from ._data import resolve_data
from ._render import save_figure

_ASSOC_FOOTER = (
    "_These are statistical associations, not proof of causation. "
    "Call explain('correlation_vs_causation') for more._"
)
_MAX_ROWS = 40


def _format_frame(obj: Any) -> str:
    """Render a DataFrame (or any value) as a fenced, row-capped text block."""
    if not isinstance(obj, pd.DataFrame):
        return f"```\n{obj}\n```"
    n = len(obj)
    body = obj if n <= _MAX_ROWS else obj.head(_MAX_ROWS)
    text = body.to_string()
    if n > _MAX_ROWS:
        text += f"\n... ({n - _MAX_ROWS} more rows; {n} total)"
    return f"```\n{text}\n```"


def _safe(call: Any) -> Any:
    """Call ``call()`` returning its result, or ``None`` if it raises."""
    try:
        return call()
    except Exception:
        return None


def render_result(spec: ToolSpec, result: Any) -> str:
    """Assemble the text payload for a tool result per its output-surface metadata."""
    parts: list[str] = [f"## {spec.name}"]

    interpretation = _safe(result.interpret) if hasattr(result, "interpret") else None
    if isinstance(interpretation, str) and interpretation.strip():
        parts.append(interpretation.strip())

    for attr in spec.tables:
        value = getattr(result, attr, None)
        if value is not None:
            parts.append(f"**{attr}**\n{_format_frame(value)}")

    for method in spec.table_methods:
        frame = _safe(getattr(result, method)) if hasattr(result, method) else None
        if frame is not None:
            parts.append(f"**{method}()**\n{_format_frame(frame)}")

    figure_paths: list[str] = []
    for attr in spec.figure_attrs:
        fig = getattr(result, attr, None)
        if fig is not None:
            figure_paths.extend(save_figure(fig, stem=f"{spec.name}-{attr}"))
    if figure_paths:
        parts.append(
            "**Figure(s)** (open in a browser):\n"
            + "\n".join(f"- {p}" for p in figure_paths)
        )

    if spec.see_also_topics:
        pointers = ", ".join(f"explain('{t}')" for t in spec.see_also_topics)
        parts.append(f"_See also: {pointers}._")
    if not (isinstance(interpretation, str) and interpretation.strip()):
        parts.append(_ASSOC_FOOTER)

    return "\n\n".join(parts)


def run_tool(spec: ToolSpec, arguments: dict[str, Any]) -> str:
    """Resolve data, call the function, and render the result for an estimation tool."""
    args = dict(arguments)
    handle = args.pop("data", None)
    forced = spec.forced
    if spec.takes_data:
        if handle is None:
            raise ValueError("missing required 'data' argument")
        df = resolve_data(handle)
        result = spec.func(df, **{**args, **forced})
    else:
        result = spec.func(**{**args, **forced})
    return render_result(spec, result)


def run_discovery(name: str, arguments: dict[str, Any]) -> str:
    """Handle the read-only knowledge tools ``explain`` and ``list_topics``."""
    if name == "list_topics":
        topics = expdpy.list_topics()
        return "Concept topics (call explain('<topic>')):\n" + "\n".join(
            f"- {t}" for t in topics
        )
    if name == "explain":
        topic = arguments["topic"]
        lang = arguments.get("lang", "en")
        return expdpy.explain(topic, lang=lang).to_markdown()
    raise ValueError(f"unknown discovery tool: {name}")


# --------------------------------------------------------------- learn_concept (MCP-only)


def learn_topics() -> list[str]:
    """Return the concept-sandbox topics (the ``learn_*`` functions, prefix-stripped)."""
    return sorted(
        name[len("learn_") :] for name in expdpy.__all__ if name.startswith("learn_")
    )


def learn_concept_schema() -> dict[str, Any]:
    """JSON Schema for the ``learn_concept`` dispatcher tool."""
    return {
        "type": "object",
        "properties": {
            "topic": {
                "type": "string",
                "enum": learn_topics(),
                "description": "Which concept sandbox to run (a simulated demonstration).",
            }
        },
        "required": ["topic"],
        "additionalProperties": False,
    }


def learn_concept_description() -> str:
    """Return the description for the ``learn_concept`` dispatcher tool."""
    return (
        "Run a Learn sandbox: a self-contained, simulated demonstration of an "
        "econometric concept (e.g. omitted_variable_bias, fixed_effects, nickell_bias). "
        "Returns a plain-language reading, the simulated summary, and a figure path. "
        "Use it to teach a concept, not to estimate on the user's data.\n\n"
        + GUARDRAIL_PREAMBLE
    )


def run_learn_concept(arguments: dict[str, Any]) -> str:
    """Dispatch ``learn_concept`` to the matching ``learn_*`` sandbox."""
    topic = arguments["topic"]
    func = getattr(expdpy, f"learn_{topic}", None)
    if func is None:
        raise ValueError(f"unknown learn topic: {topic!r}")
    result = func()
    parts: list[str] = [f"## learn_{topic}"]
    interpretation = _safe(result.interpret) if hasattr(result, "interpret") else None
    if isinstance(interpretation, str) and interpretation.strip():
        parts.append(interpretation.strip())
    summary = getattr(result, "summary", None)
    if isinstance(summary, dict) and summary:
        rows = "\n".join(f"{k}: {v}" for k, v in summary.items())
        parts.append(f"**Summary**\n```\n{rows}\n```")
    fig = getattr(result, "fig", None)
    if fig is not None:
        paths = save_figure(fig, stem=f"learn_{topic}")
        if paths:
            parts.append("**Figure(s):**\n" + "\n".join(f"- {p}" for p in paths))
    return "\n\n".join(parts)
