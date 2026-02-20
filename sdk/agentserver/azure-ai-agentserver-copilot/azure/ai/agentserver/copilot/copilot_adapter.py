# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
# pylint: disable=logging-fstring-interpolation,broad-exception-caught
import asyncio
import dataclasses
import os
import uuid
from typing import Any, Dict, List, Optional

from copilot import CopilotClient, MessageOptions, ProviderConfig, SessionConfig
from copilot.generated.session_events import SessionEventType
from copilot.types import PermissionRequest, PermissionRequestResult, ResumeSessionConfig
from opentelemetry import context as otel_context, trace

from azure.ai.agentserver.core.constants import Constants
from azure.ai.agentserver.core.logger import get_logger
from azure.ai.agentserver.core.server.base import FoundryCBAgent
from azure.ai.agentserver.core.server.common.agent_run_context import AgentRunContext

from .models.copilot_request_converter import CopilotRequestConverter
from .models.copilot_response_converter import CopilotResponseConverter, CopilotStreamingResponseConverter

logger = get_logger()

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

    def __init__(self, session_config: Optional[SessionConfig] = None):
        super().__init__()
        self._session_config = session_config or _build_session_config()
        self._client: Optional[CopilotClient] = None
        self._credential = None

        # Map approval_request_id → {session_id, prompt, denied_requests}
        self._pending_approvals: Dict[str, Dict[str, Any]] = {}

        # Multi-turn: map conversation_id → live CopilotSession
        self._sessions: Dict[str, Any] = {}

        # Keep credential for token refresh when using Foundry with Managed Identity
        if os.getenv("AZURE_AI_FOUNDRY_RESOURCE_URL") and not os.getenv("AZURE_AI_FOUNDRY_API_KEY"):
            from azure.identity import DefaultAzureCredential

            self._credential = DefaultAzureCredential()

    def _refresh_token_if_needed(self) -> SessionConfig:
        """Return the session config, refreshing the bearer token if using Foundry."""
        if self._credential is None or "provider" not in self._session_config:
            return self._session_config

        token = self._credential.get_token(_COGNITIVE_SERVICES_SCOPE).token
        self._session_config["provider"]["bearer_token"] = token
        return self._session_config

    async def _ensure_client(self) -> CopilotClient:
        """Lazily start the CopilotClient."""
        if self._client is None:
            self._client = CopilotClient()
            await self._client.start()
            logger.info("CopilotClient started")
        return self._client


    async def agent_run(self, context: AgentRunContext):

        prompt = CopilotRequestConverter(context.request).convert()
        logger.debug(f"Copilot prompt: {prompt!r}")

        client = await self._ensure_client()
        config = self._refresh_token_if_needed()

        # First run: deny all tool requests and capture them
        denied_requests: List[Dict[str, Any]] = []

        def _on_permission(req: PermissionRequest, _ctx: dict) -> PermissionRequestResult:
            request_id = str(uuid.uuid4())
            kind = req.get("kind", "unknown")
            logger.info(f"Denied tool (needs approval): kind={kind} id={request_id}")
            denied_requests.append({
                "request_id": request_id,
                "permission_request": dict(req),
            })
            return PermissionRequestResult(
                kind="denied-no-approval-rule-and-could-not-request-from-user"
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
                logger.debug(f"Cached session {session.session_id!r} under conversation {conversation_id!r}")
        else:
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
            # Non-streaming: collect all events to extract the final text.
            text = ""
            async for event in _iter_copilot_events(session, prompt):
                if event.type == SessionEventType.ASSISTANT_MESSAGE and event.data and event.data.content:
                    text = event.data.content
            return CopilotResponseConverter.to_response(text, context)

        # Streaming: return an async generator so events flow to the client
        # immediately as the Copilot SDK emits them — no wait-until-idle.
        return self._run_streaming(session, prompt, context, tracer, span_attrs)

    async def _run_streaming(
        self,
        session: Any,
        prompt: str,
        context: AgentRunContext,
        tracer: Any,
        span_attrs: Dict[str, Any],
    ):
        """Async generator: converts Copilot events → RAPI stream events on-the-fly.

        The OTel ``invoke_agent`` span is started before the first event and
        closed in the ``finally`` block so its duration covers the full stream.
        Usage attributes (token counts, response model) are set as soon as the
        ``ASSISTANT_USAGE`` event arrives, before the span ends.
        """
        agent_name = span_attrs.get("gen_ai.agent.name", "")
        span = tracer.start_span(
            name=f"invoke_agent {agent_name}",
            kind=trace.SpanKind.CLIENT,
            attributes=span_attrs,
        )
        token = otel_context.attach(trace.set_span_in_context(span))
        try:
            converter = CopilotStreamingResponseConverter(context)
            item_id = context.id_generator.generate_message_id()
            async for copilot_event in _iter_copilot_events(session, prompt):
                # Enrich span from usage event (arrives after the assistant turn)
                if copilot_event.type == SessionEventType.ASSISTANT_USAGE and copilot_event.data:
                    data = copilot_event.data
                    if data.model:
                        span.set_attribute("gen_ai.response.model", data.model)
                    if data.input_tokens is not None:
                        span.set_attribute("gen_ai.usage.input_tokens", int(data.input_tokens))
                    if data.output_tokens is not None:
                        span.set_attribute("gen_ai.usage.output_tokens", int(data.output_tokens))
                # Yield RAPI events immediately — 0..N per Copilot event
                for rapi_event in converter._convert_event(copilot_event, context, item_id):
                    yield rapi_event
            span.set_attribute("gen_ai.response.finish_reasons", ["stop"])
        except Exception as e:
            span.set_attribute("error.type", type(e).__name__)
            logger.error(f"Error during Copilot streaming: {e}")
            raise
        finally:
            otel_context.detach(token)
            span.end()

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


async def _iter_copilot_events(session, prompt: str, timeout: int = 120):
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
        if logger.isEnabledFor(10):  # DEBUG
            data_fields: Dict[str, Any] = {}
            if event.data is not None:
                try:
                    raw = dataclasses.asdict(event.data)
                    data_fields = {k: v for k, v in raw.items() if v is not None}
                except Exception:
                    data_fields = {"repr": repr(event.data)}
            logger.debug(
                f"Event #{event_count:03d} {event.type.name if event.type else '?'} "
                f"data={data_fields}"
            )

        queue.put_nowait(event)
        if event.type == SessionEventType.SESSION_IDLE:
            queue.put_nowait(None)  # sentinel — closes the async for loop

    unsubscribe = session.on(on_event)
    try:
        await session.send(MessageOptions(prompt=prompt))
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

