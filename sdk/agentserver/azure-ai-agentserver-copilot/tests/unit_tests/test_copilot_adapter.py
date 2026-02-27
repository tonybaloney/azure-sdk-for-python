# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""Unit tests for CopilotAdapter — _HealthCheckFilter, get_agent_identifier,
get_trace_attributes, permission handling, and session lifecycle helpers."""

import logging
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from azure.ai.agentserver.copilot.copilot_adapter import (
    CopilotAdapter,
    _HealthCheckFilter,
    _build_session_config,
)
from azure.ai.agentserver.copilot.tool_acl import ToolAcl


# ---------------------------------------------------------------------------
# _HealthCheckFilter
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestHealthCheckFilter:
    """Tests for the logging filter that suppresses health-check noise."""

    def _make_record(self, msg: str) -> logging.LogRecord:
        record = logging.LogRecord(
            name="uvicorn.access", level=logging.INFO,
            pathname="", lineno=0, msg=msg, args=(), exc_info=None,
        )
        return record

    def test_filters_liveness(self):
        f = _HealthCheckFilter()
        assert f.filter(self._make_record('GET /liveness HTTP/1.1 200')) is False

    def test_filters_readiness(self):
        f = _HealthCheckFilter()
        assert f.filter(self._make_record('GET /readiness HTTP/1.1 200')) is False

    def test_passes_normal_request(self):
        f = _HealthCheckFilter()
        assert f.filter(self._make_record('POST /responses HTTP/1.1 200')) is True

    def test_passes_unrelated_log(self):
        f = _HealthCheckFilter()
        assert f.filter(self._make_record('Application startup complete.')) is True

    def test_filters_liveness_substring(self):
        f = _HealthCheckFilter()
        assert f.filter(self._make_record('127.0.0.1 - "GET /liveness" 200')) is False


# ---------------------------------------------------------------------------
# get_agent_identifier
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGetAgentIdentifier:
    """Tests for get_agent_identifier() which returns an OTel-friendly name."""

    def _make_adapter(self) -> CopilotAdapter:
        adapter = CopilotAdapter.__new__(CopilotAdapter)
        return adapter

    def test_default_identifier(self):
        adapter = self._make_adapter()
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("AGENT_NAME", None)
            os.environ.pop("AGENT_ID", None)
            result = adapter.get_agent_identifier()
        assert result == "HostedAgent-Copilot"

    def test_agent_name_env(self):
        adapter = self._make_adapter()
        with patch.dict(os.environ, {"AGENT_NAME": "my-copilot"}, clear=False):
            result = adapter.get_agent_identifier()
        assert result == "my-copilot"

    def test_agent_id_env(self):
        adapter = self._make_adapter()
        with patch.dict(os.environ, {"AGENT_ID": "agent-42"}, clear=False):
            os.environ.pop("AGENT_NAME", None)
            result = adapter.get_agent_identifier()
        assert result == "agent-42"

    def test_agent_name_takes_priority(self):
        adapter = self._make_adapter()
        with patch.dict(os.environ, {"AGENT_NAME": "name", "AGENT_ID": "id"}, clear=False):
            result = adapter.get_agent_identifier()
        assert result == "name"


# ---------------------------------------------------------------------------
# get_trace_attributes
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGetTraceAttributes:
    """Tests for get_trace_attributes() which enriches OTel span attributes."""

    def test_includes_service_namespace(self):
        adapter = CopilotAdapter.__new__(CopilotAdapter)
        # Provide minimal attributes that super().get_trace_attributes() needs
        adapter.tracer = None
        with patch.object(type(adapter).__bases__[0], "get_trace_attributes", return_value={}):
            attrs = adapter.get_trace_attributes()
        assert attrs["service.namespace"] == "azure.ai.agentserver.copilot"


# ---------------------------------------------------------------------------
# Permission request handling (_on_permission callback)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPermissionHandling:
    """Tests for the _on_permission callback inside agent_run."""

    def test_no_acl_approves_all(self):
        """When no ACL is configured, all tool requests are auto-approved."""
        from copilot.types import PermissionRequestResult

        adapter = CopilotAdapter.__new__(CopilotAdapter)
        adapter._acl = None

        # Simulate the callback logic from agent_run
        acl = adapter._acl
        req = {"kind": "shell", "fullCommandText": "rm -rf /"}
        if acl is None:
            result = PermissionRequestResult(kind="approved")
        elif acl.is_allowed(req):
            result = PermissionRequestResult(kind="approved")
        else:
            result = PermissionRequestResult(kind="denied-by-rules", rules=[])

        assert result["kind"] == "approved"

    def test_acl_allows_permitted_request(self, tmp_path):
        """ACL approves requests that match an allow rule."""
        from copilot.types import PermissionRequestResult

        acl_file = tmp_path / "acl.yaml"
        acl_file.write_text('version: "1"\ndefault_action: allow\nrules: []\n')
        acl = ToolAcl.from_file(acl_file)

        adapter = CopilotAdapter.__new__(CopilotAdapter)
        adapter._acl = acl

        req = {"kind": "shell", "fullCommandText": "echo hi"}
        if adapter._acl is None:
            result = PermissionRequestResult(kind="approved")
        elif adapter._acl.is_allowed(req):
            result = PermissionRequestResult(kind="approved")
        else:
            result = PermissionRequestResult(kind="denied-by-rules", rules=[])

        assert result["kind"] == "approved"

    def test_acl_denies_blocked_request(self, tmp_path):
        """ACL denies requests that fall through to default deny."""
        from copilot.types import PermissionRequestResult

        acl_file = tmp_path / "acl.yaml"
        acl_file.write_text('version: "1"\ndefault_action: deny\nrules: []\n')
        acl = ToolAcl.from_file(acl_file)

        adapter = CopilotAdapter.__new__(CopilotAdapter)
        adapter._acl = acl

        req = {"kind": "shell", "fullCommandText": "rm -rf /"}
        if adapter._acl is None:
            result = PermissionRequestResult(kind="approved")
        elif adapter._acl.is_allowed(req):
            result = PermissionRequestResult(kind="approved")
        else:
            result = PermissionRequestResult(kind="denied-by-rules", rules=[])

        assert result["kind"] == "denied-by-rules"


# ---------------------------------------------------------------------------
# _build_session_config — API key mode
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBuildSessionConfigApiKey:
    """Tests for _build_session_config with AZURE_AI_FOUNDRY_API_KEY."""

    def test_api_key_mode(self):
        """With both FOUNDRY_URL and API_KEY, uses API key directly as bearer token."""
        env = {
            "AZURE_AI_FOUNDRY_RESOURCE_URL": "https://myresource.openai.azure.com",
            "AZURE_AI_FOUNDRY_API_KEY": "test-api-key-123",
            "COPILOT_MODEL": "gpt-4.1",
        }
        with patch.dict(os.environ, env, clear=False):
            config = _build_session_config()

        assert config["model"] == "gpt-4.1"
        assert config["provider"]["bearer_token"] == "test-api-key-123"
        assert config["provider"]["base_url"] == "https://myresource.openai.azure.com/openai/v1/"

    def test_api_key_mode_default_model(self):
        """API key mode defaults to gpt-4.1."""
        env = {
            "AZURE_AI_FOUNDRY_RESOURCE_URL": "https://myresource.openai.azure.com",
            "AZURE_AI_FOUNDRY_API_KEY": "key",
        }
        with patch.dict(os.environ, env, clear=False):
            os.environ.pop("COPILOT_MODEL", None)
            config = _build_session_config()

        assert config["model"] == "gpt-4.1"


# ---------------------------------------------------------------------------
# _ensure_client
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEnsureClient:
    """Tests for the lazy CopilotClient initialization."""

    @pytest.mark.asyncio(loop_scope="function")
    async def test_creates_client_on_first_call(self):
        adapter = CopilotAdapter.__new__(CopilotAdapter)
        adapter._client = None

        mock_client = AsyncMock()
        with patch("azure.ai.agentserver.copilot.copilot_adapter.CopilotClient", return_value=mock_client):
            result = await adapter._ensure_client()

        assert result is mock_client
        mock_client.start.assert_awaited_once()

    @pytest.mark.asyncio(loop_scope="function")
    async def test_reuses_existing_client(self):
        adapter = CopilotAdapter.__new__(CopilotAdapter)
        existing = AsyncMock()
        adapter._client = existing

        result = await adapter._ensure_client()
        assert result is existing


# ---------------------------------------------------------------------------
# CopilotResponseConverter.to_response
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestNonStreamingResponse:
    """Tests for CopilotResponseConverter.to_response (non-streaming path)."""

    def test_builds_response_with_text(self):
        from azure.ai.agentserver.core.server.common.agent_run_context import AgentRunContext
        from azure.ai.agentserver.copilot.models.copilot_response_converter import CopilotResponseConverter

        ctx = AgentRunContext({"input": "hello", "stream": False})
        resp = CopilotResponseConverter.to_response("Hello world", ctx)
        assert resp["id"] == ctx.response_id
        assert len(resp["output"]) == 1
        assert resp["output"][0]["content"][0]["text"] == "Hello world"

    def test_builds_response_with_empty_text_fallback(self):
        from azure.ai.agentserver.core.server.common.agent_run_context import AgentRunContext
        from azure.ai.agentserver.copilot.models.copilot_response_converter import CopilotResponseConverter

        ctx = AgentRunContext({"input": "hello", "stream": False})
        resp = CopilotResponseConverter.to_response("   ", ctx)
        assert "(No response text was produced by the agent.)" in resp["output"][0]["content"][0]["text"]


# ---------------------------------------------------------------------------
# Session LRU eviction
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSessionLruEviction:
    """Tests for the bounded OrderedDict session cache."""

    def _make_adapter(self, max_sessions: int = 3) -> CopilotAdapter:
        adapter = CopilotAdapter.__new__(CopilotAdapter)
        import collections
        adapter._sessions = collections.OrderedDict()
        adapter._max_sessions = max_sessions
        return adapter

    def test_sessions_evicted_when_over_limit(self):
        adapter = self._make_adapter(max_sessions=2)
        adapter._sessions["conv-1"] = "session-1"
        adapter._sessions["conv-2"] = "session-2"
        adapter._sessions["conv-3"] = "session-3"
        # Simulate eviction that happens in agent_run
        while len(adapter._sessions) > adapter._max_sessions:
            adapter._sessions.popitem(last=False)
        assert "conv-1" not in adapter._sessions
        assert "conv-2" in adapter._sessions
        assert "conv-3" in adapter._sessions

    def test_reused_session_moved_to_end(self):
        adapter = self._make_adapter(max_sessions=3)
        adapter._sessions["conv-1"] = "s1"
        adapter._sessions["conv-2"] = "s2"
        adapter._sessions["conv-3"] = "s3"
        # Reusing conv-1 should move it to end
        adapter._sessions.move_to_end("conv-1")
        keys = list(adapter._sessions.keys())
        assert keys == ["conv-2", "conv-3", "conv-1"]

    def test_eviction_removes_oldest_not_most_recent(self):
        adapter = self._make_adapter(max_sessions=2)
        adapter._sessions["conv-a"] = "a"
        adapter._sessions["conv-b"] = "b"
        adapter._sessions.move_to_end("conv-a")  # touch conv-a
        adapter._sessions["conv-c"] = "c"  # add new
        while len(adapter._sessions) > adapter._max_sessions:
            adapter._sessions.popitem(last=False)
        # conv-b should be evicted (oldest untouched), conv-a and conv-c remain
        assert "conv-b" not in adapter._sessions
        assert "conv-a" in adapter._sessions
        assert "conv-c" in adapter._sessions


# ---------------------------------------------------------------------------
# from_copilot() factory function
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFromCopilot:
    """Tests for the from_copilot() factory function in __init__.py."""

    def test_factory_returns_adapter(self):
        """from_copilot() should return a CopilotAdapter instance."""
        from azure.ai.agentserver.copilot import from_copilot
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("AZURE_AI_FOUNDRY_RESOURCE_URL", None)
            os.environ.pop("AZURE_AI_FOUNDRY_API_KEY", None)
            os.environ.pop("TOOL_ACL_PATH", None)
            adapter = from_copilot()
        assert isinstance(adapter, CopilotAdapter)

    def test_factory_with_acl_path(self, tmp_path):
        """from_copilot(acl_path=...) should load ACL from file."""
        from azure.ai.agentserver.copilot import from_copilot
        acl_file = tmp_path / "acl.yaml"
        acl_file.write_text('version: "1"\ndefault_action: deny\nrules: []\n')
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("AZURE_AI_FOUNDRY_RESOURCE_URL", None)
            os.environ.pop("AZURE_AI_FOUNDRY_API_KEY", None)
            adapter = from_copilot(acl_path=acl_file)
        assert isinstance(adapter, CopilotAdapter)
        assert adapter._acl is not None


# ---------------------------------------------------------------------------
# Constructor coverage — merging session config, credential setup
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAdapterConstructor:
    """Tests for CopilotAdapter.__init__ branches."""

    def test_constructor_with_merged_session_config(self):
        """User-provided session_config should be merged with defaults."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("AZURE_AI_FOUNDRY_RESOURCE_URL", None)
            os.environ.pop("AZURE_AI_FOUNDRY_API_KEY", None)
            os.environ.pop("TOOL_ACL_PATH", None)
            user_config = {"model": "custom-model"}
            adapter = CopilotAdapter(session_config=user_config)
        # The merged config should include user's model
        assert adapter._session_config.get("model") == "custom-model"

    def test_constructor_with_acl_object(self):
        """Passing an ACL object directly should use it."""
        acl = MagicMock(spec=ToolAcl)
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("AZURE_AI_FOUNDRY_RESOURCE_URL", None)
            os.environ.pop("AZURE_AI_FOUNDRY_API_KEY", None)
            adapter = CopilotAdapter(acl=acl)
        assert adapter._acl is acl

    def test_constructor_foundry_with_managed_identity(self):
        """Setting FOUNDRY_RESOURCE_URL without API key should create a credential."""
        with patch.dict(os.environ, {
            "AZURE_AI_FOUNDRY_RESOURCE_URL": "https://example.services.ai.azure.com",
        }, clear=False):
            os.environ.pop("AZURE_AI_FOUNDRY_API_KEY", None)
            os.environ.pop("TOOL_ACL_PATH", None)
            with patch("azure.identity.DefaultAzureCredential") as mock_cred:
                adapter = CopilotAdapter()
            mock_cred.assert_called_once()
            assert adapter._credential is not None

    def test_constructor_max_sessions_env(self):
        """COPILOT_MAX_SESSIONS env var should configure max session count."""
        with patch.dict(os.environ, {"COPILOT_MAX_SESSIONS": "5"}, clear=False):
            os.environ.pop("AZURE_AI_FOUNDRY_RESOURCE_URL", None)
            os.environ.pop("AZURE_AI_FOUNDRY_API_KEY", None)
            os.environ.pop("TOOL_ACL_PATH", None)
            adapter = CopilotAdapter()
        assert adapter._max_sessions == 5


# ---------------------------------------------------------------------------
# agent_run — non-streaming path
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAgentRunNonStreaming:
    """Tests for agent_run non-streaming code path with mocked Copilot SDK."""

    @pytest.mark.asyncio(loop_scope="function")
    async def test_agent_run_non_streaming_returns_response(self):
        """Non-streaming agent_run should return a complete response."""
        from copilot.generated.session_events import SessionEventType

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("AZURE_AI_FOUNDRY_RESOURCE_URL", None)
            os.environ.pop("AZURE_AI_FOUNDRY_API_KEY", None)
            os.environ.pop("TOOL_ACL_PATH", None)
            adapter = CopilotAdapter()

        # Mock client
        mock_client = AsyncMock()
        mock_session = AsyncMock()
        mock_client.create_session = AsyncMock(return_value=mock_session)
        adapter._client = mock_client

        # Create a mock event for ASSISTANT_MESSAGE
        msg_event = MagicMock()
        msg_event.type = SessionEventType.ASSISTANT_MESSAGE
        msg_event.data = MagicMock()
        msg_event.data.content = "Hello from Copilot!"

        # Build a mock context
        context = MagicMock()
        context.request = MagicMock()
        context.request.get = lambda k, d=None: "Hello" if k == "input" else d
        context.request.__getitem__ = lambda self_, k: "Hello" if k == "input" else None
        context.stream = False
        context.conversation_id = None

        # Patch _iter_copilot_events
        async def fake_iter(session, prompt, attachments=None):
            yield msg_event

        with patch("azure.ai.agentserver.copilot.copilot_adapter._iter_copilot_events", side_effect=fake_iter):
            result = await adapter.agent_run(context)

        assert result is not None

    @pytest.mark.asyncio(loop_scope="function")
    async def test_agent_run_caches_session_for_conversation(self):
        """agent_run should cache session when conversation_id is provided."""
        from copilot.generated.session_events import SessionEventType

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("AZURE_AI_FOUNDRY_RESOURCE_URL", None)
            os.environ.pop("AZURE_AI_FOUNDRY_API_KEY", None)
            os.environ.pop("TOOL_ACL_PATH", None)
            adapter = CopilotAdapter()

        mock_client = AsyncMock()
        mock_session = AsyncMock()
        mock_session.session_id = "sess-123"
        mock_client.create_session = AsyncMock(return_value=mock_session)
        adapter._client = mock_client

        msg_event = MagicMock()
        msg_event.type = SessionEventType.ASSISTANT_MESSAGE
        msg_event.data = MagicMock()
        msg_event.data.content = "Reply"

        context = MagicMock()
        context.request = MagicMock()
        context.request.get = lambda k, d=None: "Hi" if k == "input" else d
        context.stream = False
        context.conversation_id = "conv-abc"

        async def fake_iter(session, prompt, attachments=None):
            yield msg_event

        with patch("azure.ai.agentserver.copilot.copilot_adapter._iter_copilot_events", side_effect=fake_iter):
            await adapter.agent_run(context)

        assert "conv-abc" in adapter._sessions
