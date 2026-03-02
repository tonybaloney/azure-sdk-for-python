# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""Unit tests for invoke_agent span attribute compliance.

Validates that ``_build_invoke_span_attrs`` produces attributes conforming to
the OTel GenAI agent span semantic conventions:
https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-agent-spans/#invoke-agent-span

Attribute requirement levels tested:

    Required
        gen_ai.operation.name
        gen_ai.provider.name

    Conditionally Required (when available)
        gen_ai.agent.id
        gen_ai.agent.name
        gen_ai.agent.version
        gen_ai.request.model
        gen_ai.conversation.id
        server.port           (if server.address is set)

    Recommended
        gen_ai.response.id
        server.address        (when span kind is CLIENT)

See also the ``test_otel_spans.py`` integration tests that verify the full
span lifecycle with a running server.
"""
from __future__ import annotations

import os
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from azure.ai.agentserver.copilot._types import InvokeAgentSpanAttrs
from azure.ai.agentserver.copilot.copilot_adapter import CopilotAdapter
from azure.ai.agentserver.core.server.common.agent_run_context import AgentRunContext


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_context(
    *,
    agent_name: str = "test-agent",
    agent_version: str = "2.0",
    conversation_id: str = "conv_abc123",
    response_id: str | None = None,
) -> AgentRunContext:
    """Build a minimal AgentRunContext with agent & conversation info."""
    payload: Dict[str, Any] = {"input": "hello", "stream": True}
    if agent_name:
        agent: Dict[str, Any] = {"type": "agent_id", "name": agent_name}
        if agent_version:
            agent["version"] = agent_version
        payload["agent"] = agent
    if conversation_id:
        payload["conversation"] = {"id": conversation_id}
    ctx = AgentRunContext(payload)
    if response_id is not None:
        # Override the auto-generated response ID for deterministic tests.
        ctx._response_id = response_id
    return ctx


def _make_config(
    *,
    model: str = "gpt-4.1",
    base_url: str = "",
    bearer_token: str = "tok",
) -> Dict[str, Any]:
    """Build a minimal SessionConfig-like dict."""
    config: Dict[str, Any] = {"model": model}
    if base_url:
        config["provider"] = {
            "type": "openai",
            "base_url": base_url,
            "bearer_token": bearer_token,
        }
    return config


def _build_attrs(
    config: Dict[str, Any] | None = None,
    context: AgentRunContext | None = None,
    conversation_id: str | None = "conv_abc123",
) -> InvokeAgentSpanAttrs:
    """Invoke ``_build_invoke_span_attrs`` on a fresh adapter.

    Uses ``patch`` to skip CopilotClient/base-class side effects.
    """
    if config is None:
        config = _make_config()
    if context is None:
        context = _make_context(conversation_id=conversation_id or "")
    with patch("azure.ai.agentserver.copilot.copilot_adapter._build_session_config", return_value=config):
        adapter = CopilotAdapter.__new__(CopilotAdapter)
        # Minimal init — only what _build_invoke_span_attrs needs
        adapter._session_config = config
    return adapter._build_invoke_span_attrs(config, context, conversation_id)


# ===========================================================================
# Required attributes (MUST always be present)
# ===========================================================================


@pytest.mark.unit
class TestRequiredAttributes:
    """gen_ai.operation.name and gen_ai.provider.name are always Required."""

    def test_operation_name_is_invoke_agent(self) -> None:
        attrs = _build_attrs()
        assert attrs["gen_ai.operation.name"] == "invoke_agent"

    def test_provider_name_is_github_copilot(self) -> None:
        attrs = _build_attrs()
        assert attrs["gen_ai.provider.name"] == "github.copilot"


# ===========================================================================
# Conditionally Required — when available
# ===========================================================================


@pytest.mark.unit
class TestConditionallyRequiredAttributes:
    """Attributes that MUST be present when the information is available."""

    def test_agent_id_from_context(self) -> None:
        attrs = _build_attrs()
        assert "gen_ai.agent.id" in attrs
        # When version is present, id is "name:version"
        assert attrs["gen_ai.agent.id"] == "test-agent:2.0"

    def test_agent_name_from_context(self) -> None:
        attrs = _build_attrs()
        assert attrs["gen_ai.agent.name"] == "test-agent"

    def test_agent_version_from_context(self) -> None:
        """gen_ai.agent.version MUST be set when the version is available."""
        attrs = _build_attrs()
        assert attrs["gen_ai.agent.version"] == "2.0"

    def test_agent_version_omitted_when_not_available(self) -> None:
        """gen_ai.agent.version MUST NOT be set when version is empty."""
        ctx = _make_context(agent_version="")
        attrs = _build_attrs(context=ctx)
        assert "gen_ai.agent.version" not in attrs

    def test_request_model_from_config(self) -> None:
        config = _make_config(model="gpt-5")
        attrs = _build_attrs(config=config)
        assert attrs["gen_ai.request.model"] == "gpt-5"

    def test_conversation_id_present_when_provided(self) -> None:
        attrs = _build_attrs(conversation_id="conv_abc123")
        assert attrs["gen_ai.conversation.id"] == "conv_abc123"

    def test_conversation_id_absent_when_not_provided(self) -> None:
        attrs = _build_attrs(conversation_id=None)
        assert "gen_ai.conversation.id" not in attrs

    def test_server_port_present_when_server_address_set(self) -> None:
        """server.port is Conditionally Required if server.address is set."""
        config = _make_config(base_url="https://myresource.openai.azure.com:8443/openai/v1/")
        attrs = _build_attrs(config=config)
        assert "server.address" in attrs
        assert attrs["server.port"] == 8443

    def test_server_port_absent_when_no_address(self) -> None:
        """Neither server.address nor server.port when no provider URL."""
        config = _make_config(base_url="")
        attrs = _build_attrs(config=config)
        assert "server.address" not in attrs
        assert "server.port" not in attrs


# ===========================================================================
# Recommended attributes
# ===========================================================================


@pytest.mark.unit
class TestRecommendedAttributes:
    """Attributes that SHOULD be present."""

    def test_response_id_present(self) -> None:
        """gen_ai.response.id SHOULD be set from context.response_id."""
        ctx = _make_context(response_id="resp_xyz789")
        attrs = _build_attrs(context=ctx)
        assert attrs["gen_ai.response.id"] == "resp_xyz789"

    def test_response_id_is_string(self) -> None:
        attrs = _build_attrs()
        assert isinstance(attrs["gen_ai.response.id"], str)
        assert len(attrs["gen_ai.response.id"]) > 0

    def test_server_address_from_provider_url(self) -> None:
        """server.address SHOULD be set when span kind is CLIENT."""
        config = _make_config(base_url="https://myresource.openai.azure.com/openai/v1/")
        attrs = _build_attrs(config=config)
        assert attrs["server.address"] == "myresource.openai.azure.com"

    def test_server_address_not_set_without_provider(self) -> None:
        """server.address omitted when there is no provider config (GitHub default)."""
        config = _make_config(base_url="")
        attrs = _build_attrs(config=config)
        assert "server.address" not in attrs


# ===========================================================================
# Span name format
# ===========================================================================


@pytest.mark.unit
class TestSpanNameFormat:
    """Span name SHOULD be 'invoke_agent {gen_ai.agent.name}'."""

    def test_span_name_includes_agent_name(self) -> None:
        attrs = _build_attrs()
        agent_name = attrs["gen_ai.agent.name"]
        expected_span_name = "invoke_agent %s" % agent_name
        assert expected_span_name == "invoke_agent test-agent"

    def test_span_name_fallback_when_no_agent(self) -> None:
        """When no agent info is available, falls back to 'HostedAgent-Copilot'."""
        ctx = _make_context(agent_name="", agent_version="")
        with patch.dict(os.environ, {}, clear=False):
            # Remove any AGENT_NAME/AGENT_ID env vars
            env = {k: v for k, v in os.environ.items()
                   if k not in ("AGENT_NAME", "AGENT_ID")}
            with patch.dict(os.environ, env, clear=True):
                attrs = _build_attrs(context=ctx)
        assert attrs["gen_ai.agent.name"] == "HostedAgent-Copilot"


# ===========================================================================
# Agent identity fallback chain
# ===========================================================================


@pytest.mark.unit
class TestAgentIdentityFallback:
    """Agent identity resolution: context → env vars → hardcoded default."""

    def test_env_var_fallback(self) -> None:
        """When context has no agent info, falls back to AGENT_NAME env var."""
        ctx = _make_context(agent_name="", agent_version="")
        with patch.dict(os.environ, {"AGENT_NAME": "EnvAgent", "AGENT_ID": "env-id-1"}, clear=False):
            attrs = _build_attrs(context=ctx)
        assert attrs["gen_ai.agent.name"] == "EnvAgent"
        assert attrs["gen_ai.agent.id"] == "env-id-1"

    def test_agent_id_defaults_to_name(self) -> None:
        """gen_ai.agent.id defaults to gen_ai.agent.name when no separate id."""
        ctx = _make_context(agent_name="solo-agent", agent_version="")
        with patch.dict(os.environ, {}, clear=False):
            env = {k: v for k, v in os.environ.items()
                   if k not in ("AGENT_NAME", "AGENT_ID")}
            with patch.dict(os.environ, env, clear=True):
                attrs = _build_attrs(context=ctx)
        assert attrs["gen_ai.agent.id"] == "solo-agent"
        assert attrs["gen_ai.agent.name"] == "solo-agent"


# ===========================================================================
# Server address parsing edge cases
# ===========================================================================


@pytest.mark.unit
class TestServerAddressParsing:
    """Edge cases for extracting server.address and server.port."""

    def test_default_https_port_not_emitted(self) -> None:
        """Standard port 443 is implicit for https and not emitted."""
        config = _make_config(base_url="https://host.example.com/openai/v1/")
        attrs = _build_attrs(config=config)
        assert attrs["server.address"] == "host.example.com"
        # urllib.parse doesn't return port for default scheme ports
        assert "server.port" not in attrs

    def test_custom_port_emitted(self) -> None:
        config = _make_config(base_url="https://host.example.com:9090/openai/v1/")
        attrs = _build_attrs(config=config)
        assert attrs["server.address"] == "host.example.com"
        assert attrs["server.port"] == 9090

    def test_localhost_url(self) -> None:
        config = _make_config(base_url="http://localhost:5000/v1/")
        attrs = _build_attrs(config=config)
        assert attrs["server.address"] == "localhost"
        assert attrs["server.port"] == 5000
