# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
# pylint: disable=logging-fstring-interpolation,broad-exception-caught
import asyncio
import os
import uuid
from typing import Any, Dict, List, Optional

from copilot import CopilotClient, MessageOptions, ProviderConfig, SessionConfig
from copilot.generated.session_events import SessionEventType
from copilot.types import PermissionRequest, PermissionRequestResult, ResumeSessionConfig

from azure.ai.agentserver.core.constants import Constants
from azure.ai.agentserver.core.logger import get_logger
from azure.ai.agentserver.core.server.base import FoundryCBAgent
from azure.ai.agentserver.core.server.common.agent_run_context import AgentRunContext

from .models.copilot_request_converter import CopilotRequestConverter
from .models.copilot_response_converter import CopilotResponseConverter

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


def _deny_handler(req: PermissionRequest, _ctx: dict) -> PermissionRequestResult:
    """Deny permission — used on first run to capture tool requests."""
    return PermissionRequestResult(
        kind="denied-no-approval-rule-and-could-not-request-from-user"
    )


def _approve_handler(req: PermissionRequest, _ctx: dict) -> PermissionRequestResult:
    """Approve permission — used on resumed sessions after user approval."""
    logger.info(f"Approved tool: kind={req.get('kind')}")
    return PermissionRequestResult(kind="approved")


class CopilotAdapter(FoundryCBAgent):
    """Adapter that bridges a GitHub Copilot SDK session to an Azure AI Agent Server.

    When ``AZURE_AI_FOUNDRY_RESOURCE_URL`` is set the adapter uses Azure AI
    Foundry models via BYOK with Managed Identity authentication.  Otherwise
    it falls back to the default GitHub Copilot models.

    Tool approval
    ~~~~~~~~~~~~~
    When the Copilot CLI requests permission to execute a tool (file write,
    shell command, etc.) the adapter **denies** the tool immediately so the
    session can complete without hanging, but captures the request details.
    The response includes both the model's text *and* an
    ``mcp_approval_request`` output item, with ``status="incomplete"``.
    The Copilot session is **kept alive** (not destroyed).

    If the caller sends a follow-up request containing an
    ``mcp_approval_response`` item with ``approve=true``, the adapter
    **resumes** the same Copilot session (by session ID) with an
    approve-all permission handler, and re-sends the original prompt.
    The session already has conversation history so the model retries
    the tool call, which is now approved.

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

        session_config = SessionConfig(**config, on_permission_request=_on_permission)
        session = await client.create_session(session_config)

        try:
            collected_events, _ = await _send_and_collect(session, prompt)
            full_text = _collect_text(collected_events)
            logger.debug(
                f"Session collected {len(collected_events)} events, "
                f"text length={len(full_text)}, "
                f"denied tools={len(denied_requests)}"
            )

            if denied_requests:
                # Keep the session alive — store session ID for resume
                session_id = session.session_id
                for req in denied_requests:
                    self._pending_approvals[req["request_id"]] = {
                        "session_id": session_id,
                        "prompt": prompt,
                        "permission_request": req["permission_request"],
                    }
                logger.info(
                    f"Parked session {session_id} with "
                    f"{len(denied_requests)} pending approvals"
                )

                if not context.stream:
                    return CopilotResponseConverter.to_approval_request_response(
                        full_text, denied_requests, prompt, context,
                    )
                return CopilotResponseConverter.to_stream_events_with_approval(
                    collected_events, denied_requests, prompt, context,
                )

            # No tools requested — normal completion, destroy session
            await session.destroy()
            if not context.stream:
                return CopilotResponseConverter.to_response(full_text, context)
            return CopilotResponseConverter.to_stream_events(
                collected_events, context,
            )

        except Exception as e:
            logger.error(f"Error during Copilot agent run: {e}")
            raise

    # ------------------------------------------------------------------
    # Approval follow-up
    # ------------------------------------------------------------------

    async def _handle_approval_response(
        self, approval: Dict[str, Any], context: AgentRunContext,
    ):
        """Resume the parked session after user approval."""
        approved = approval.get("approve", False)
        request_id = approval.get("approval_request_id", "")

        pending = self._pending_approvals.pop(request_id, None)
        if pending is None:
            logger.warning(f"No pending session for approval_request_id={request_id}")
            return CopilotResponseConverter.to_response(
                "Error: no pending tool approval found for the given ID.",
                context,
            )

        if not approved:
            logger.info(f"Tool denied by caller: {request_id}")
            return CopilotResponseConverter.to_response(
                "The requested tool operation was denied.", context,
            )

        # Resume the same Copilot session with an approve-all handler
        session_id = pending["session_id"]
        prompt = pending["prompt"]
        logger.info(f"Resuming session {session_id} with approval")

        client = await self._ensure_client()
        config = self._refresh_token_if_needed()

        resume_config = ResumeSessionConfig(
            on_permission_request=_approve_handler,
        )
        # Inject provider config for BYOK token refresh
        if "provider" in config:
            resume_config["provider"] = config["provider"]

        session = await client.resume_session(session_id, resume_config)

        try:
            # Re-send the same prompt — the session has history so the
            # model knows it previously tried and was denied; now it
            # retries and the tool is approved.
            collected_events, _ = await _send_and_collect(session, prompt)
            await session.destroy()

            full_text = _collect_text(collected_events)
            logger.info(
                f"Resumed session produced {len(collected_events)} events, "
                f"text={full_text[:100]!r}"
            )

            if not context.stream:
                return CopilotResponseConverter.to_response(full_text, context)
            return CopilotResponseConverter.to_stream_events(
                collected_events, context,
            )

        except Exception as e:
            logger.error(f"Error resuming session {session_id}: {e}")
            raise

    # ------------------------------------------------------------------
    # Trace / identity
    # ------------------------------------------------------------------

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


async def _send_and_collect(session, prompt: str):
    """Send a message and collect all events until SESSION_IDLE.

    The Copilot SDK sometimes emits duplicate events (especially after
    ``resume_session``).  We deduplicate by tracking seen
    ``(event_type, content)`` pairs.
    """
    collected = []
    seen = set()
    done = asyncio.Event()

    def on_event(event):
        # Build a dedup key from type + content (if any)
        text = ""
        if event.data and hasattr(event.data, "content") and event.data.content:
            text = event.data.content
        key = (event.type, text)
        if key in seen:
            logger.debug(f"Skipping duplicate event: {event.type}")
            return
        seen.add(key)

        collected.append(event)
        logger.debug(
            f"Event: {event.type} data_type={type(event.data).__name__} "
            f"text={text[:80]!r}"
        )
        if event.type == SessionEventType.SESSION_IDLE:
            done.set()

    session.on(on_event)
    await session.send(MessageOptions(prompt=prompt))
    await asyncio.wait_for(done.wait(), timeout=120)
    return collected, done


def _collect_text(events: list) -> str:
    """Extract assistant text from collected Copilot events.

    Prefers ASSISTANT_MESSAGE content, but falls back to assembling
    ASSISTANT_MESSAGE_DELTA chunks if no complete message was received.
    """
    message_parts = []
    for event in events:
        if event.type == SessionEventType.ASSISTANT_MESSAGE:
            if event.data and hasattr(event.data, "content") and event.data.content:
                message_parts.append(event.data.content)
    if message_parts:
        return "\n".join(message_parts)

    delta_parts = []
    for event in events:
        if event.type == SessionEventType.ASSISTANT_MESSAGE_DELTA:
            if event.data and hasattr(event.data, "content") and event.data.content:
                delta_parts.append(event.data.content)
    return "".join(delta_parts)
