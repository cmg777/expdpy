"""The expdpy MCP (Model Context Protocol) server.

Builds a stdio server that advertises the curated tool set from :mod:`expdpy._meta` —
using the *same* ``param_schema()`` the static Anthropic/OpenAI schemas use, so the live
input schemas can never drift from the committed ones — plus the ``learn_concept``
sandbox dispatcher. The MCP SDK is imported here (at module top), so importing this module
requires the optional ``mcp`` extra; :mod:`expdpy.mcp.__main__` guards that with a
friendly install hint.
"""

from __future__ import annotations

import asyncio
from typing import Any

import mcp.types as types
from mcp.server.lowlevel.server import Server
from mcp.server.stdio import stdio_server

from expdpy._meta import param_schema, tool_description, tool_specs

from ._adapter import (
    learn_concept_description,
    learn_concept_schema,
    run_discovery,
    run_learn_concept,
    run_tool,
)

#: Tools that surface the explainer registry rather than analysing data.
_DISCOVERY = frozenset({"explain", "list_topics"})

INSTRUCTIONS = (
    "expdpy exposes a curated set of panel and cross-sectional data-analysis tools "
    "(Explore / Analyze) plus concept tools (explain, list_topics, learn_concept). Pass "
    "data via the 'data' handle: a bundled dataset name, an absolute file path, or inline "
    "records. Panel data uses entity (unit) + time vocabulary. Results are statistical "
    "associations, not proof of causation: prefer each tool's interpret() reading and "
    "consult explain(topic) for method caveats."
)

_SPECS = {spec.name: spec for spec in tool_specs()}


def _dispatch(name: str, arguments: dict[str, Any]) -> str:
    """Route a tool call to the right handler and return the rendered text payload."""
    if name == "learn_concept":
        return run_learn_concept(arguments)
    spec = _SPECS.get(name)
    if spec is None:
        raise ValueError(f"unknown tool: {name!r}")
    if name in _DISCOVERY:
        return run_discovery(name, arguments)
    return run_tool(spec, arguments)


def build_server() -> Server:
    """Construct the stdio MCP server with the curated expdpy tool set."""
    server: Server = Server("expdpy", instructions=INSTRUCTIONS)

    @server.list_tools()
    async def _list_tools() -> list[types.Tool]:
        tools = [
            types.Tool(
                name=spec.name,
                description=tool_description(spec),
                inputSchema=param_schema(spec),
            )
            for spec in tool_specs()
        ]
        tools.append(
            types.Tool(
                name="learn_concept",
                description=learn_concept_description(),
                inputSchema=learn_concept_schema(),
            )
        )
        return tools

    @server.call_tool()
    async def _call_tool(
        name: str, arguments: dict[str, Any]
    ) -> list[types.TextContent]:
        # Estimation can be CPU-bound (pyfixest/linearmodels); run it off the event loop.
        text = await asyncio.to_thread(_dispatch, name, arguments)
        return [types.TextContent(type="text", text=text)]

    return server


async def _serve() -> None:
    server = build_server()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream, write_stream, server.create_initialization_options()
        )


def run() -> None:
    """Run the expdpy MCP server over stdio (blocks until the client disconnects)."""
    asyncio.run(_serve())
