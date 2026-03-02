# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""OTel span-attribute TypedDicts for the Copilot adapter.

Mirrors the GenAI agent span semconv requirement levels so that pyright
catches missing / misspelled keys at lint time.
"""
from typing import NotRequired, TypedDict

# ---------------------------------------------------------------------------
# invoke_agent span
# ---------------------------------------------------------------------------
# https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-agent-spans/#invoke-agent-span

InvokeAgentSpanAttrs = TypedDict("InvokeAgentSpanAttrs", {
    # Required — MUST always be present.
    "gen_ai.operation.name": str,
    "gen_ai.provider.name": str,
    # Conditionally Required — always populated by this adapter.
    "gen_ai.agent.id": str,
    "gen_ai.agent.name": str,
    "gen_ai.request.model": str,
    # Recommended — always populated by this adapter.
    "gen_ai.response.id": str,
    # Conditionally Required — when available.
    "gen_ai.agent.version": NotRequired[str],
    "gen_ai.conversation.id": NotRequired[str],
    # Recommended for CLIENT spans.
    "server.address": NotRequired[str],
    # Conditionally Required — when server.address is set.
    "server.port": NotRequired[int],
})

# ---------------------------------------------------------------------------
# MCP tool-call span
# ---------------------------------------------------------------------------
# https://opentelemetry.io/docs/specs/semconv/gen-ai/mcp/

ToolCallSpanAttrs = TypedDict("ToolCallSpanAttrs", {
    # Required by MCP semconv.
    "mcp.method.name": str,
    # Conditionally Required — when tool is present.
    "gen_ai.tool.name": str,
    # Recommended.
    "gen_ai.operation.name": str,
    "network.transport": str,
    # Optional / Recommended when available.
    "mcp.session.id": NotRequired[str],
    "gen_ai.tool.call.id": NotRequired[str],
    "mcp.server.name": NotRequired[str],
    "gen_ai.tool.call.arguments": NotRequired[str],
})
