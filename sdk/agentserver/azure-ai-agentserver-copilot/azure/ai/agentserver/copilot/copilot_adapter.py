# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
# pylint: disable=logging-fstring-interpolation,broad-exception-caught
import asyncio
import collections
import dataclasses
import json
import os
import uuid
from typing import Any, Dict, Optional

from copilot import CopilotClient, MessageOptions, ProviderConfig, SessionConfig
from copilot.generated.session_events import SessionEventType
from copilot.types import PermissionRequest, PermissionRequestResult, ResumeSessionConfig
from opentelemetry import context as otel_context, trace

from azure.ai.agentserver.core.constants import Constants
from azure.ai.agentserver.core.logger import get_logger
from azure.ai.agentserver.core.server.base import FoundryCBAgent
from azure.ai.agentserver.core.server.common.agent_run_context import AgentRunContext

from .models.copilot_request_converter import ConvertedAttachments, CopilotRequestConverter
from .models.copilot_response_converter import CopilotResponseConverter, CopilotStreamingResponseConverter
from .tool_acl import ToolAcl



logger = get_logger()

# The opentelemetry.context.detach() call can emit "Failed to detach context" warnings
# when an async generator's finally block runs in a different contextvars context than
# where the token was created.  This is expected behaviour in async streaming code and
# does not affect tracing correctness (spans are still started and ended properly).
import logging as _logging
_logging.getLogger("opentelemetry.context").setLevel(_logging.CRITICAL)


class _HealthCheckFilter(_logging.Filter):
    """Drop health-check access-log records so they don't pollute App Insights."""

    _PATHS = ("/liveness", "/readiness")

    def filter(self, record: _logging.LogRecord) -> bool:  # noqa: A003
        msg = record.getMessage()
        return not any(p in msg for p in self._PATHS)

_COGNITIVE_SERVICES_SCOPE = "https://cognitiveservices.azure.com/.default"


def _build_session_config() -> SessionConfig:
    """Build a SessionConfig from environment variables.

    When ``AZURE_AI_FOUNDRY_RESOURCE_URL`` is set the adapter runs in
    **BYOK mode** against Azure AI Foundry.

    Authentication priority:
    1. ``AZURE_AI_FOUNDRY_API_KEY`` — static API key (easiest for local dev)
    2. ``DefaultAzureCredential`` — Managed Identity / Azure CLI (for prod)

    When using an API key the token is set once at startup.  When using
    ``DefaultAzureCredential`` the bearer token starts as a placeholder and
    is refreshed lazily by ``_refresh_token_if_needed()`` before each
    session.

    Environment variables
    ---------------------
    AZURE_AI_FOUNDRY_RESOURCE_URL
        The Azure AI Foundry resource URL, e.g.
        ``https://<resource>.cognitiveservices.azure.com``.
    AZURE_AI_FOUNDRY_API_KEY
        Static API key for the Foundry resource.  If set, takes priority
        over ``DefaultAzureCredential``.
    COPILOT_MODEL
        Model deployment name (default ``gpt-4.1``).
    """
    foundry_url = os.getenv("AZURE_AI_FOUNDRY_RESOURCE_URL")
    model = os.getenv("COPILOT_MODEL", "gpt-4.1")

    if foundry_url:
        base_url = foundry_url.rstrip("/") + "/openai/v1/"
        api_key = os.getenv("AZURE_AI_FOUNDRY_API_KEY")

        if api_key:
            logger.info(f"BYOK mode (API key): {base_url}")
            return SessionConfig(
                model=model,
                provider=ProviderConfig(
                    type="openai",
                    base_url=base_url,
                    bearer_token=api_key,
                    wire_api="responses",
                ),
            )

        logger.info(f"BYOK mode (Managed Identity): {base_url}")
        return SessionConfig(
            model=model,
            provider=ProviderConfig(
                type="openai",
                base_url=base_url,
                bearer_token="placeholder",  # refreshed before first use
                wire_api="responses",
            ),
        )

    # Fallback: default GitHub Copilot models
    return SessionConfig(model=os.getenv("COPILOT_MODEL", "gpt-5"))


class CopilotAdapter(FoundryCBAgent):
    """Adapter that bridges a GitHub Copilot SDK session to an Azure AI Agent Server.

    When ``AZURE_AI_FOUNDRY_RESOURCE_URL`` is set the adapter uses Azure AI
    Foundry models via BYOK with Managed Identity authentication.  Otherwise
    it falls back to the default GitHub Copilot models.

    :param session_config: Override for the Copilot session config.  When
        *None* the config is built automatically from environment variables.
    :type session_config: Optional[SessionConfig]
    """

    def __init__(
        self,
        session_config: Optional[SessionConfig] = None,
        acl: Optional[ToolAcl] = None,
    ):
        super().__init__()

        # Register a startup event to suppress health-check noise.
        # This MUST be a startup event (not inline __init__ code) because
        # uvicorn's configure_logging() calls dictConfig() which resets
        # loggers and clears any filters we add before the server starts.
        # Startup events fire AFTER uvicorn's logging is configured.
        @self.app.on_event("startup")
        async def _suppress_health_check_logs():
            _hc_filter = _HealthCheckFilter()
            for _name in ("uvicorn", "uvicorn.access", "uvicorn.error"):
                _logging.getLogger(_name).addFilter(_hc_filter)

        # Build default config (handles provider setup from env vars)
        default_config = _build_session_config()

        if session_config is not None:
            # Merge user-provided config with default config.
            # User config takes priority for most fields, but provider
            # is taken from default if not specified by user.
            merged = dict(default_config)  # Start with default
            user_dict = dict(session_config)

            # Merge user settings — only override if user provided a non-None value
            for key, value in user_dict.items():
                if value is not None:
                    merged[key] = value

            self._session_config = SessionConfig(**merged)  # type: ignore[arg-type]
        else:
            self._session_config = default_config

        self._client: Optional[CopilotClient] = None
        self._credential = None

        # Tool ACL: explicit parameter takes priority over TOOL_ACL_PATH env var.
        # If neither is set, every tool request is auto-approved (approve-all).
        if acl is not None:
            self._acl: Optional[ToolAcl] = acl
        else:
            self._acl = ToolAcl.from_env("TOOL_ACL_PATH")
            if self._acl is None:
                logger.warning(
                    "No tool ACL configured (TOOL_ACL_PATH not set). "
                    "All tool requests will be auto-approved. "
                    "Set TOOL_ACL_PATH to a YAML ACL file for production use."
                )

        # Multi-turn: map conversation_id → live CopilotSession (LRU-bounded)
        _MAX_SESSIONS = int(os.getenv("COPILOT_MAX_SESSIONS", "1000"))
        self._sessions: collections.OrderedDict[str, Any] = collections.OrderedDict()
        self._max_sessions = _MAX_SESSIONS

        # Keep credential for token refresh when using Foundry with Managed Identity
        if os.getenv("AZURE_AI_FOUNDRY_RESOURCE_URL") and not os.getenv("AZURE_AI_FOUNDRY_API_KEY"):
            from azure.identity import DefaultAzureCredential

            self._credential = DefaultAzureCredential()

    async def _refresh_token_if_needed(self) -> SessionConfig:
        """Return the session config, refreshing the bearer token if using Foundry.

        Token acquisition is run in a thread executor to avoid blocking the
        async event loop during the synchronous HTTP round-trip.
        """
        if self._credential is None or "provider" not in self._session_config:
            return self._session_config

        token = await asyncio.to_thread(
            self._credential.get_token, _COGNITIVE_SERVICES_SCOPE
        )
        self._session_config["provider"]["bearer_token"] = token.token
        return self._session_config

    async def _ensure_client(self) -> CopilotClient:
        """Lazily start the CopilotClient."""
        if self._client is None:
            self._client = CopilotClient()
            await self._client.start()
            logger.info("CopilotClient started")
        return self._client


    async def agent_run(self, context: AgentRunContext):

        req_converter = CopilotRequestConverter(context.request)
        prompt = req_converter.convert()
        logger.debug(f"Copilot prompt: {prompt!r}")
        converted_attachments = req_converter.convert_attachments()
        if converted_attachments:
            logger.debug(f"Attachments: {len(converted_attachments.attachments)} item(s)")

        client = await self._ensure_client()
        config = await self._refresh_token_if_needed()

        acl = self._acl

        def _on_permission(req: PermissionRequest, _ctx: dict) -> PermissionRequestResult:
            kind = req.get("kind", "unknown")
            if acl is None:
                # No ACL configured — approve everything (development / local mode).
                logger.info(f"Auto-approving tool request (no ACL): kind={kind}")
                return PermissionRequestResult(kind="approved")

            if acl.is_allowed(req):
                logger.info(f"ACL allowed tool request: kind={kind}")
                return PermissionRequestResult(kind="approved")
            else:
                logger.warning(f"ACL denied tool request: kind={kind}")
                return PermissionRequestResult(
                    kind="denied-by-rules",
                    rules=[],
                )

        conversation_id = context.conversation_id
        session = self._sessions.get(conversation_id) if conversation_id else None

        if session is None:
            logger.info(
                f"Creating new Copilot session"
                + (f" for conversation {conversation_id!r}" if conversation_id else "")
            )
            session_config = SessionConfig(**config, on_permission_request=_on_permission)
            session = await client.create_session(session_config)
            if conversation_id:
                self._sessions[conversation_id] = session
                # Evict oldest session if we've exceeded the cap
                while len(self._sessions) > self._max_sessions:
                    evicted_id, _ = self._sessions.popitem(last=False)
                    logger.debug(f"Evicted oldest session for conversation {evicted_id!r}")
                logger.debug(f"Cached session {session.session_id!r} under conversation {conversation_id!r}")
        else:
            self._sessions.move_to_end(conversation_id)
            logger.info(f"Reusing Copilot session {session.session_id!r} for conversation turn (conversation={conversation_id!r})")

        tracer = self.tracer or trace.get_tracer(__name__)
        agent_name = self.get_agent_identifier()
        request_model = config.get("model", "") if hasattr(config, "get") else ""

        span_attrs: Dict[str, Any] = {
            "gen_ai.operation.name": "invoke_agent",
            "gen_ai.provider.name": "github.copilot",
            "gen_ai.agent.id": "copilot",
            "gen_ai.agent.name": agent_name,
            "gen_ai.request.model": request_model,
        }
        if conversation_id:
            span_attrs["gen_ai.conversation.id"] = conversation_id

        if not context.stream:
            # Non-streaming: collect all events and extract the final text
            # from the authoritative ASSISTANT_MESSAGE event.
            text = ""
            try:
                async for event in _iter_copilot_events(session, prompt, attachments=converted_attachments.attachments):
                    if event.type == SessionEventType.ASSISTANT_MESSAGE and event.data and event.data.content:
                        text = event.data.content
            finally:
                converted_attachments.cleanup()
            return CopilotResponseConverter.to_response(text, context)

        # Streaming: return an async generator so events flow to the client
        # immediately as the Copilot SDK emits them — no wait-until-idle.
        return self._run_streaming(session, prompt, converted_attachments, context, tracer, span_attrs)

    async def _run_streaming(
        self,
        session: Any,
        prompt: str,
        converted_attachments: ConvertedAttachments,
        context: AgentRunContext,
        tracer: Any,
        span_attrs: Dict[str, Any],
    ):
        """Async generator: converts Copilot events → RAPI stream events on-the-fly.

        The OTel ``invoke_agent`` span is started before the first event and
        closed in the ``finally`` block so its duration covers the full stream.
        Usage attributes (token counts, response model) are set as soon as the
        ``ASSISTANT_USAGE`` event arrives, before the span ends.

        Per-tool ``execute_tool`` child spans are opened on
        ``TOOL_EXECUTION_START`` and closed on ``TOOL_EXECUTION_COMPLETE``.
        They inherit the trace context from the enclosing ``invoke_agent`` span.
        """
        agent_name = span_attrs.get("gen_ai.agent.name", "")
        span = tracer.start_span(
            name=f"invoke_agent {agent_name}",
            kind=trace.SpanKind.CLIENT,
            attributes=span_attrs,
        )
        token = otel_context.attach(trace.set_span_in_context(span))
        # tool_call_id → (child_span, otel_token) for in-flight tool executions
        tool_spans: Dict[str, Any] = {}
        try:
            converter = CopilotStreamingResponseConverter(context)
            async for copilot_event in _iter_copilot_events(session, prompt, attachments=converted_attachments.attachments):
                data = copilot_event.data

                # ── Tool execution start: open a child span ────────────────────
                # Span follows OTel MCP semconv (tools/call):
                # https://opentelemetry.io/docs/specs/semconv/gen-ai/mcp/
                if copilot_event.type == SessionEventType.TOOL_EXECUTION_START and data:
                    call_id = data.tool_call_id or str(uuid.uuid4())
                    tool_name = data.mcp_tool_name or data.tool_name or "unknown"
                    tool_attrs: Dict[str, Any] = {
                        # Required by MCP semconv
                        "mcp.method.name": "tools/call",
                        # Conditionally required when tool is present
                        "gen_ai.tool.name": tool_name,
                        # Recommended
                        "gen_ai.operation.name": "execute_tool",
                        "network.transport": "pipe",  # Copilot SDK uses stdio
                    }
                    # mcp.session.id — recommended when part of a session
                    try:
                        tool_attrs["mcp.session.id"] = session.session_id
                    except AttributeError:
                        pass
                    if call_id:
                        tool_attrs["gen_ai.tool.call.id"] = call_id
                    if data.mcp_server_name:
                        tool_attrs["mcp.server.name"] = data.mcp_server_name
                    # gen_ai.tool.call.arguments — opt-in per MCP semconv
                    if data.arguments is not None:
                        try:
                            tool_attrs["gen_ai.tool.call.arguments"] = json.dumps(data.arguments)
                        except Exception:
                            tool_attrs["gen_ai.tool.call.arguments"] = str(data.arguments)
                    tool_span = tracer.start_span(
                        name=f"tools/call {tool_name}",
                        kind=trace.SpanKind.CLIENT,
                        attributes=tool_attrs,
                    )
                    tool_token = otel_context.attach(trace.set_span_in_context(tool_span))
                    tool_spans[call_id] = (tool_span, tool_token)
                    logger.debug(f"Tool span started: {tool_name!r} call_id={call_id!r}")

                # ── Tool execution complete: close the matching child span ──────
                elif copilot_event.type == SessionEventType.TOOL_EXECUTION_COMPLETE and data:
                    call_id = data.tool_call_id
                    entry = tool_spans.pop(call_id, None) if call_id else None
                    if entry:
                        tool_span, tool_token = entry
                        # gen_ai.tool.call.result — opt-in per MCP semconv
                        if data.result is not None:
                            try:
                                result_str = json.dumps(data.result)
                            except Exception:
                                result_str = str(data.result)
                            tool_span.set_attribute("gen_ai.tool.call.result", result_str[:512])
                        # error.type — required when tool execution failed
                        if hasattr(data, "success") and data.success is False:
                            tool_span.set_attribute("error.type", "tool_error")
                        try:
                            otel_context.detach(tool_token)
                        except ValueError:
                            pass  # token created in a different async context — safe to ignore
                        tool_span.end()
                        logger.debug(f"Tool span ended: call_id={call_id!r}")

                # ── Enrich parent span from usage event ───────────────────────
                elif copilot_event.type == SessionEventType.ASSISTANT_USAGE and data:
                    if data.model:
                        span.set_attribute("gen_ai.response.model", data.model)
                    if data.input_tokens is not None:
                        span.set_attribute("gen_ai.usage.input_tokens", int(data.input_tokens))
                    if data.output_tokens is not None:
                        span.set_attribute("gen_ai.usage.output_tokens", int(data.output_tokens))

                # Yield RAPI events immediately — 0..N per Copilot event
                for rapi_event in converter._convert_event(copilot_event, context):
                    yield rapi_event

            # NOTE: The RAPI gateway/Container App ingress currently drops
            # the final SSE events (text_done → completed → [DONE]) regardless
            # of how long the adapter waits before closing.  A sleep here does
            # NOT help — tested up to 2 seconds with no effect.  The issue is
            # in the platform infrastructure, not timing.

            span.set_attribute("gen_ai.response.finish_reasons", ["stop"])
        except Exception as e:
            span.set_attribute("error.type", type(e).__name__)
            logger.error(f"Error during Copilot streaming: {e}")
            raise
        finally:
            # Close any tool spans that were never completed (e.g. stream aborted)
            for call_id, (tool_span, tool_token) in list(tool_spans.items()):
                tool_span.set_attribute("error.type", "stream_aborted")
                try:
                    otel_context.detach(tool_token)
                except ValueError:
                    pass  # token created in a different async context — safe to ignore
                tool_span.end()
            try:
                otel_context.detach(token)
            except ValueError:
                pass  # token created in a different async context — safe to ignore
            span.end()
            converted_attachments.cleanup()

    def get_trace_attributes(self):
        attrs = super().get_trace_attributes()
        attrs["service.namespace"] = "azure.ai.agentserver.copilot"
        return attrs

    def get_agent_identifier(self) -> str:
        agent_name = os.getenv(Constants.AGENT_NAME)
        if agent_name:
            return agent_name
        agent_id = os.getenv(Constants.AGENT_ID)
        if agent_id:
            return agent_id
        return "HostedAgent-Copilot"


async def _iter_copilot_events(session, prompt: str, attachments: Optional[list] = None, timeout: int = 120):
    """Send *prompt* to *session* and yield each ``SessionEvent`` as it arrives.

    This is a true async generator — it yields each event to the caller
    immediately rather than waiting for ``SESSION_IDLE`` to collect them all.
    The caller therefore begins processing (and forwarding to the HTTP client)
    as soon as the first event arrives from the Copilot SDK.

    Consecutive duplicate events (a ``resume_session``/reconnect artefact) are
    silently dropped.  The generator stops after the ``SESSION_IDLE`` sentinel.
    The event listener is unsubscribed in the ``finally`` block so reused
    multi-turn sessions don't accumulate stale listeners.
    """
    queue: asyncio.Queue = asyncio.Queue()
    last_key = None
    event_count = 0

    def on_event(event):
        nonlocal last_key, event_count
        # Build a dedup key from type + content
        text = ""
        if event.data and hasattr(event.data, "content") and event.data.content:
            text = event.data.content
        key = (event.type, text)
        if key == last_key:
            logger.debug(f"Skipping consecutive duplicate: {event.type}")
            return
        last_key = key

        event_count += 1
        # INFO-level trace so deployed images can diagnose event flow without requiring DEBUG.
        event_name = event.type.name if event.type else "UNKNOWN"
        if text:
            logger.info(f"Copilot event #{event_count:03d}: {event_name} content_len={len(text)}")
        else:
            logger.info(f"Copilot event #{event_count:03d}: {event_name}")

        # Always log SESSION_ERROR details at WARNING level so production logs contain
        # enough information to diagnose model/auth/infra failures without DEBUG.
        if event.type == SessionEventType.SESSION_ERROR and event.data:
            error_msg = getattr(event.data, 'message', None) or getattr(event.data, 'content', None) or repr(event.data)
            logger.warning(f"SESSION_ERROR details: {error_msg}")
        if logger.isEnabledFor(10):  # DEBUG
            data_fields: Dict[str, Any] = {}
            if event.data is not None:
                try:
                    raw = dataclasses.asdict(event.data)
                    data_fields = {k: v for k, v in raw.items() if v is not None}
                except Exception:
                    data_fields = {"repr": repr(event.data)}
            logger.debug(
                f"Event #{event_count:03d} {event_name} "
                f"data={data_fields}"
            )

        queue.put_nowait(event)
        if event.type == SessionEventType.SESSION_IDLE:
            queue.put_nowait(None)  # sentinel — closes the async for loop after SESSION_IDLE is processed

    unsubscribe = session.on(on_event)
    try:
        msg_opts: Dict[str, Any] = {"prompt": prompt}
        if attachments:
            msg_opts["attachments"] = attachments
        await session.send(MessageOptions(**msg_opts))
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise asyncio.TimeoutError(f"Copilot session idle timeout after {timeout}s")
            event = await asyncio.wait_for(queue.get(), timeout=remaining)
            if event is None:  # sentinel
                return
            yield event
    finally:
        unsubscribe()

