# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
# pylint: disable=logging-fstring-interpolation,broad-exception-caught
import asyncio
import os
from typing import Optional

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


def _build_session_config() -> SessionConfig:
    """Build a SessionConfig from environment variables.

    When ``AZURE_AI_FOUNDRY_RESOURCE_URL`` is set the adapter runs in
    **BYOK mode** against Azure AI Foundry.  A short-lived bearer token
    is obtained via ``DefaultAzureCredential`` (supports Managed Identity,
    Azure CLI, etc.) and injected as the API key because the Copilot SDK
    does not natively support Entra ID / Managed Identity auth.

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

    :param session_config: Override for the Copilot session config.  When
        *None* the config is built automatically from environment variables.
    :type session_config: Optional[SessionConfig]
    """

    def __init__(self, session_config: Optional[SessionConfig] = None):
        super().__init__()
        self._session_config = session_config or _build_session_config()
        self._client: Optional[CopilotClient] = None
        self._credential = None

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

    async def agent_run(self, context: AgentRunContext):
        prompt = CopilotRequestConverter(context.request).convert()
        logger.debug(f"Copilot prompt: {prompt!r}")

        client = await self._ensure_client()
        config = self._refresh_token_if_needed()

        # Auto-approve tool calls — the hosted agent runs in a sandboxed
        # container so file writes and shell commands are safe.
        def _auto_approve(req: PermissionRequest, _ctx: dict) -> PermissionRequestResult:
            logger.info(f"Auto-approving tool: kind={req.get('kind')} intention={req.get('intention', '')}")
            return PermissionRequestResult(kind="approved")

        session_config = SessionConfig(**config, on_permission_request=_auto_approve)
        session = await client.create_session(session_config)
        try:
            if not context.stream:
                return await self._run_non_stream(session, prompt, context)
            return await self._run_stream(session, prompt, context)
        except Exception as e:
            logger.error(f"Error during Copilot agent run: {e}")
            raise

    async def _run_non_stream(self, session, prompt: str, context: AgentRunContext):
        """Execute a non-streaming request and return a single Response."""
        collected_events = []
        done = asyncio.Event()

        def on_event(event):
            collected_events.append(event)
            if event.type == SessionEventType.SESSION_IDLE:
                done.set()

        session.on(on_event)
        await session.send(MessageOptions(prompt=prompt))
        await asyncio.wait_for(done.wait(), timeout=300)
        await session.destroy()

        full_text = _collect_text(collected_events)
        logger.debug(f"Non-stream collected {len(collected_events)} events, text length={len(full_text)}")

        return CopilotResponseConverter.to_response(full_text, context)

    async def _run_stream(self, session, prompt: str, context: AgentRunContext):
        """Execute a streaming request, yielding ResponseStreamEvent objects."""
        collected_events = []
        done = asyncio.Event()

        def on_event(event):
            collected_events.append(event)
            if event.type == SessionEventType.SESSION_IDLE:
                done.set()

        session.on(on_event)
        await session.send(MessageOptions(prompt=prompt))
        await asyncio.wait_for(done.wait(), timeout=300)
        await session.destroy()

        logger.debug(f"Stream collected {len(collected_events)} events")

        # Convert collected events to streaming events
        return CopilotResponseConverter.to_stream_events(collected_events, context)

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
