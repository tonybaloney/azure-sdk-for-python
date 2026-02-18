# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
# pylint: disable=logging-fstring-interpolation,broad-exception-caught
import asyncio
import os
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Dict, Optional

from copilot import CopilotClient, MessageOptions, ProviderConfig, SessionConfig
from copilot.generated.session_events import SessionEventType
from copilot.types import PermissionRequest, PermissionRequestResult

from azure.ai.agentserver.core.constants import Constants
from azure.ai.agentserver.core.logger import get_logger
from azure.ai.agentserver.core.server.base import FoundryCBAgent
from azure.ai.agentserver.core.server.common.agent_run_context import AgentRunContext

from .models.copilot_request_converter import CopilotRequestConverter
from .models.copilot_response_converter import CopilotResponseConverter

logger = get_logger()

_COGNITIVE_SERVICES_SCOPE = "https://cognitiveservices.azure.com/.default"


@dataclass
class _PendingSession:
    """Holds a Copilot session that is paused waiting for tool approval."""

    session: Any
    approval_future: asyncio.Future
    event_queue: asyncio.Queue
    done_event: asyncio.Event
    approval_request_id: str
    permission_request: Dict[str, Any] = field(default_factory=dict)


def _build_session_config() -> SessionConfig:
    """Build a SessionConfig from environment variables.

    When ``AZURE_AI_FOUNDRY_RESOURCE_URL`` is set the adapter runs in
    **BYOK mode** against Azure AI Foundry.  A short-lived bearer token
    is obtained via ``DefaultAzureCredential`` (supports Managed Identity,
    Azure CLI, etc.) and injected as ``bearer_token`` in the provider
    config.

    Environment variables
    ---------------------
    AZURE_AI_FOUNDRY_RESOURCE_URL
        The Azure AI Foundry resource URL, e.g.
        ``https://<resource>.openai.azure.com``.  When set, BYOK mode is
        activated.
    COPILOT_MODEL
        Model deployment name (default ``gpt-4.1``).

    :return: A ready-to-use SessionConfig.
    :rtype: SessionConfig
    """
    foundry_url = os.getenv("AZURE_AI_FOUNDRY_RESOURCE_URL")
    model = os.getenv("COPILOT_MODEL", "gpt-4.1")

    if foundry_url:
        from azure.identity import DefaultAzureCredential

        credential = DefaultAzureCredential()
        token = credential.get_token(_COGNITIVE_SERVICES_SCOPE).token

        base_url = foundry_url.rstrip("/") + "/openai/v1/"
        logger.info(f"BYOK mode: using Azure AI Foundry at {base_url}")

        return SessionConfig(
            model=model,
            provider=ProviderConfig(
                type="openai",
                base_url=base_url,
                bearer_token=token,
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

    Tool approval
    ~~~~~~~~~~~~~
    When the Copilot CLI requests permission to execute a tool (file write,
    shell command, etc.) the adapter yields an ``mcp_approval_request``
    output item and returns the response as ``incomplete``.  The caller
    sends a follow-up request containing an ``mcp_approval_response`` item
    to approve or deny, and the adapter resumes the same Copilot session.

    :param session_config: Override for the Copilot session config.  When
        *None* the config is built automatically from environment variables.
    :type session_config: Optional[SessionConfig]
    """

    def __init__(self, session_config: Optional[SessionConfig] = None):
        super().__init__()
        self._session_config = session_config or _build_session_config()
        self._client: Optional[CopilotClient] = None
        self._credential = None

        # Map approval_request_id → _PendingSession for sessions paused
        # on a tool approval request.
        self._pending_sessions: Dict[str, _PendingSession] = {}

        # Keep credential for token refresh when using Foundry
        if os.getenv("AZURE_AI_FOUNDRY_RESOURCE_URL"):
            from azure.identity import DefaultAzureCredential

            self._credential = DefaultAzureCredential()

    def _refresh_token_if_needed(self) -> SessionConfig:
        """Return the session config, refreshing the bearer token if using Foundry."""
        if self._credential is None or "provider" not in self._session_config:
            return self._session_config

        token = self._credential.get_token(_COGNITIVE_SERVICES_SCOPE).token
        # Rebuild provider with fresh token
        self._session_config["provider"]["bearer_token"] = token
        return self._session_config

    async def _ensure_client(self) -> CopilotClient:
        """Lazily start the CopilotClient."""
        if self._client is None:
            self._client = CopilotClient()
            await self._client.start()
            logger.info("CopilotClient started")
        return self._client

    # ------------------------------------------------------------------
    # Input helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_approval_response(context: AgentRunContext) -> Optional[Dict[str, Any]]:
        """Return the first ``mcp_approval_response`` item in the request input, if any."""
        input_items = context.request.get("input")
        if not isinstance(input_items, list):
            return None
        for item in input_items:
            if isinstance(item, dict) and item.get("type") == "mcp_approval_response":
                return item
        return None

    # ------------------------------------------------------------------
    # agent_run – entry point
    # ------------------------------------------------------------------

    async def agent_run(self, context: AgentRunContext):
        # Check if this is a follow-up request containing a tool approval
        approval = self._find_approval_response(context)
        if approval:
            return await self._handle_approval_response(approval, context)

        prompt = CopilotRequestConverter(context.request).convert()
        logger.debug(f"Copilot prompt: {prompt!r}")

        client = await self._ensure_client()
        config = self._refresh_token_if_needed()

        # Set up shared state for the permission handler
        loop = asyncio.get_running_loop()
        approval_queue: asyncio.Queue = asyncio.Queue()

        async def _on_permission(
            req: PermissionRequest, _ctx: dict
        ) -> PermissionRequestResult:
            """Pause execution and surface the approval request to the caller."""
            request_id = str(uuid.uuid4())
            future: asyncio.Future = loop.create_future()
            logger.info(
                f"Tool approval requested: kind={req.get('kind')} "
                f"id={request_id}"
            )
            await approval_queue.put(
                {"request_id": request_id, "permission_request": dict(req), "future": future}
            )
            # Block the CLI until the caller approves or denies
            return await future

        session_config = SessionConfig(**config, on_permission_request=_on_permission)
        session = await client.create_session(session_config)

        try:
            if not context.stream:
                return await self._run_non_stream(
                    session, prompt, context, approval_queue
                )
            return self._run_stream(session, prompt, context, approval_queue)
        except Exception as e:
            logger.error(f"Error during Copilot agent run: {e}")
            raise

    # ------------------------------------------------------------------
    # Approval follow-up
    # ------------------------------------------------------------------

    async def _handle_approval_response(
        self, approval: Dict[str, Any], context: AgentRunContext
    ):
        """Resume a pending session after the caller approves/denies a tool."""
        request_id = approval.get("approval_request_id", "")
        pending = self._pending_sessions.pop(request_id, None)
        if pending is None:
            logger.warning(f"No pending session for approval_request_id={request_id}")
            return CopilotResponseConverter.to_response(
                "Error: no pending tool approval found for the given ID.", context
            )

        approved = approval.get("approve", False)
        reason = approval.get("reason", "")
        if approved:
            kind = "approved"
            logger.info(f"Tool approved: {request_id}")
        else:
            kind = "denied-interactively-by-user"
            logger.info(f"Tool denied: {request_id} reason={reason}")

        # Unblock the Copilot CLI's on_permission_request callback
        pending.approval_future.set_result(
            PermissionRequestResult(kind=kind)
        )

        # Wait for the session to finish processing
        await asyncio.wait_for(pending.done_event.wait(), timeout=300)

        # Collect remaining events
        remaining_events = []
        while not pending.event_queue.empty():
            remaining_events.append(pending.event_queue.get_nowait())

        await pending.session.destroy()

        if not context.stream:
            full_text = _collect_text(remaining_events)
            return CopilotResponseConverter.to_response(full_text, context)
        return CopilotResponseConverter.to_stream_events(remaining_events, context)

    # ------------------------------------------------------------------
    # Non-streaming
    # ------------------------------------------------------------------

    async def _run_non_stream(
        self, session, prompt: str, context: AgentRunContext,
        approval_queue: asyncio.Queue,
    ):
        """Execute a non-streaming request.

        If a tool approval is needed the response is returned as incomplete
        with an ``mcp_approval_request`` output item.
        """
        event_queue: asyncio.Queue = asyncio.Queue()
        done = asyncio.Event()

        def on_event(event):
            event_queue.put_nowait(event)
            if event.type == SessionEventType.SESSION_IDLE:
                done.set()

        session.on(on_event)
        await session.send(MessageOptions(prompt=prompt))

        # Race: either the session finishes or a tool approval is requested
        done_task = asyncio.ensure_future(done.wait())
        approval_task = asyncio.ensure_future(approval_queue.get())
        finished, pending_tasks = await asyncio.wait(
            {done_task, approval_task},
            timeout=300,
            return_when=asyncio.FIRST_COMPLETED,
        )
        for t in pending_tasks:
            t.cancel()

        if approval_task in finished:
            # Tool approval requested — park the session
            approval_info = approval_task.result()
            return self._park_session_and_return_approval(
                session, event_queue, done, approval_info, context
            )

        # Normal completion
        collected = _drain_queue(event_queue)
        await session.destroy()
        full_text = _collect_text(collected)
        logger.debug(
            f"Non-stream collected {len(collected)} events, "
            f"text length={len(full_text)}"
        )
        return CopilotResponseConverter.to_response(full_text, context)

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    def _run_stream(
        self, session, prompt: str, context: AgentRunContext,
        approval_queue: asyncio.Queue,
    ) -> AsyncGenerator:
        """Return an async generator that streams response events.

        If a tool approval is needed the stream yields an
        ``mcp_approval_request`` item and ends with ``response.incomplete``.
        """
        return self._stream_generator(session, prompt, context, approval_queue)

    async def _stream_generator(
        self, session, prompt: str, context: AgentRunContext,
        approval_queue: asyncio.Queue,
    ) -> AsyncGenerator:
        event_queue: asyncio.Queue = asyncio.Queue()
        done = asyncio.Event()

        def on_event(event):
            event_queue.put_nowait(event)
            if event.type == SessionEventType.SESSION_IDLE:
                done.set()

        session.on(on_event)
        await session.send(MessageOptions(prompt=prompt))

        # Race between completion and tool approval
        done_task = asyncio.ensure_future(done.wait())
        approval_task = asyncio.ensure_future(approval_queue.get())

        finished, pending_tasks = await asyncio.wait(
            {done_task, approval_task},
            timeout=300,
            return_when=asyncio.FIRST_COMPLETED,
        )
        for t in pending_tasks:
            t.cancel()

        if approval_task in finished:
            approval_info = approval_task.result()
            # Yield any events collected so far, then the approval request
            collected = _drain_queue(event_queue)
            for evt in CopilotResponseConverter.to_stream_events_with_approval(
                collected, approval_info, context
            ):
                yield evt
            # Park the session for later resumption
            self._pending_sessions[approval_info["request_id"]] = _PendingSession(
                session=session,
                approval_future=approval_info["future"],
                event_queue=event_queue,
                done_event=done,
                approval_request_id=approval_info["request_id"],
                permission_request=approval_info["permission_request"],
            )
            return

        # Normal completion — yield all events
        collected = _drain_queue(event_queue)
        await session.destroy()
        logger.debug(f"Stream collected {len(collected)} events")
        for evt in CopilotResponseConverter.to_stream_events(collected, context):
            yield evt

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _park_session_and_return_approval(
        self, session, event_queue, done_event, approval_info, context
    ):
        """Store the pending session and return an incomplete response with approval request."""
        self._pending_sessions[approval_info["request_id"]] = _PendingSession(
            session=session,
            approval_future=approval_info["future"],
            event_queue=event_queue,
            done_event=done_event,
            approval_request_id=approval_info["request_id"],
            permission_request=approval_info["permission_request"],
        )
        return CopilotResponseConverter.to_approval_request_response(
            approval_info, context
        )

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


def _drain_queue(q: asyncio.Queue) -> list:
    """Drain all items from an asyncio.Queue into a list."""
    items = []
    while not q.empty():
        items.append(q.get_nowait())
    return items


def _collect_text(events: list) -> str:
    """Extract all assistant text from collected Copilot events.

    Prefers ASSISTANT_MESSAGE content, but falls back to assembling
    ASSISTANT_MESSAGE_DELTA chunks if no complete message was received.
    """
    # Try complete messages first
    message_parts = []
    for event in events:
        if event.type == SessionEventType.ASSISTANT_MESSAGE:
            if event.data and hasattr(event.data, "content") and event.data.content:
                message_parts.append(event.data.content)
    if message_parts:
        return "\n".join(message_parts)

    # Fall back to deltas
    delta_parts = []
    for event in events:
        if event.type == SessionEventType.ASSISTANT_MESSAGE_DELTA:
            if event.data and hasattr(event.data, "content") and event.data.content:
                delta_parts.append(event.data.content)
    return "".join(delta_parts)
